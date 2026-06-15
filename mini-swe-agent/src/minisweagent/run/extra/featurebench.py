#!/usr/bin/env python3

#TODO: Move evaluation related code to different file.

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.extra.utils.maintainability import analyze
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

from minisweagent.evaluation.featurebench import evaluation
from minisweagent.evaluation.reporting import make_run_report
from minisweagent.evaluation.maintainability_metrics import maintainability_metrics
from minisweagent.run.extra.utils.exceptions import NoFailToPassError

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

#TODO - add to config or something
SANITY_CORRUPT_FOLDER="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/synthetic_chains_claude"
CONTAINER_BASE_PATH="/testbed"

RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "featureBench": "LiberCoders/FeatureBench",
    "custom": os.getenv("CUSTOM_DATA_PATH"), # "/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/synthetic_chains_claude/swe_bench_first_task.csv"
}


_OUTPUT_FILE_LOCK = threading.Lock()

class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()

def get_swebench_docker_image_name(instance: dict, path_local_images: Path | None) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        if path_local_images is not None:
            image_name = f"swebench_sweb.eval.x86_64.{id_docker_compatible}_latest.sif".lower()
        else:
            image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict, path_local_images: Path | None, docker_runtime_config: dict | None = None, skip_git_reset: bool = False) -> Environment:
    image_name = get_swebench_docker_image_name(instance, path_local_images)
    env_config = config.get("environment", {})
    if env_config.get("environment_class") == "singularity":
        if path_local_images is None:
            image_name = "docker://" + image_name
        else:
            image_name = image_name.replace("libercoders/", "libercoders_") + ".sif"
            image_name = path_local_images / image_name
        if docker_runtime_config is not None:
            env_config['runtime_config'] = docker_runtime_config

    env_config["image"] = image_name
    if skip_git_reset:
        env_config['skip_git_reset'] = True
    return get_environment(env_config, default_type="docker")


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

def get_from_preds_file(output_path: Path, instance_id: str):
    """Get model patch from preds file"""
    if not output_path.exists():
        raise FileNotFoundError(f"output_path does not exists - {output_path}")
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            instance_result = output_data.get( instance_id )

            if instance_result:
                return instance_result['model_patch']

    return None

def parse_repo_settings(instance: dict) -> dict:
    """
    Parse repo_settings from instance data.

    Args:
        instance: Instance data as pandas Series

    Returns:
        Dictionary containing repo settings
    """
    repo_settings_str = instance.get("repo_settings", {})
    if not repo_settings_str:
        return {}

    try:
        if isinstance(repo_settings_str, str):
            return json.loads(repo_settings_str)
        elif isinstance(repo_settings_str, dict):
            return repo_settings_str
        else:
            return {}
    except json.JSONDecodeError:
        return {}


def get_docker_runtime_config(repo_settings: dict) -> dict:
    """
    Extract Docker runtime configuration from repo_settings.

    Args:
        repo_settings: Repository settings dictionary

    Returns:
        Dictionary with docker runtime config (need_gpu, shm_size, number_once, env_vars, env_exports)
    """
    docker_specs = repo_settings.get("docker_specs", {})
    run_args = docker_specs.get("run_args", {})
    custom_docker_args = docker_specs.get("custom_docker_args", [])

    # Check whether GPU runtime is requested.
    # Prefer new key cuda_visible_num; fall back to legacy cuda_visible_devices.
    cuda_visible_cfg = run_args.get("cuda_visible_num", run_args.get("cuda_visible_devices", None))
    if isinstance(cuda_visible_cfg, int):
        need_gpu = cuda_visible_cfg > 0
    else:
        need_gpu = bool(cuda_visible_cfg)

    # Parse shm_size
    shm_size = run_args.get("shm_size")

    # Parse number_once (GPU count)
    number_once = run_args.get("number_once", 1)
    if not isinstance(number_once, int) or number_once <= 0:
        number_once = 1

    # Parse environment variables from custom_docker_args (-e and -ee)
    # Ignore -v (volume) and proxy-related -e
    env_vars = {}
    env_exports = []  # For -ee (to be written to .bashrc)

    proxy_keywords = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']

    if isinstance(custom_docker_args, list):
        for arg in custom_docker_args:
            if not isinstance(arg, str):
                continue

            if arg.startswith('-e '):
                # Regular environment variable
                env_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                if '=' in env_part:
                    key = env_part.split('=')[0].strip()
                    # Skip proxy-related variables
                    if key not in proxy_keywords:
                        env_vars[key] = env_part.split('=', 1)[1]

            elif arg.startswith('-ee '):
                # Environment variable to be written to .bashrc
                env_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                if '=' in env_part:
                    key = env_part.split('=')[0].strip()
                    # Skip proxy-related variables
                    if key not in proxy_keywords:
                        env_exports.append(f'export {env_part}')

            # Ignore -v (volume) arguments

    return {
        "need_gpu": need_gpu,
        "shm_size": shm_size,
        "number_once": number_once,
        "env_vars": env_vars,
        "env_exports": env_exports,
    }

def get_test_config_from_repo_settings(repo_settings: dict) -> dict:
    """
    Extract test configuration from repo_settings.

    Args:
        repo_settings: Repository settings dictionary

    Returns:
        Dictionary with test_cmd, timeout_run, timeout_one
    """
    return {
        "test_cmd": repo_settings.get("test_cmd", "pytest -rA -p no:cacheprovider --color=no"),
        "timeout_run": repo_settings.get("timeout_run", 600),
        "timeout_one": repo_settings.get("timeout_one", 10),
        "use_uv": repo_settings.get("use_uv", False),
    }


def setup_environment(instance: dict, instance_dir: Path, env: Environment, init_patch: Path | None = None, files_to_change: list[str] | None = None, progress_manager: RunBatchProgressManager = None) -> None:
    """Setup the environment for the instance.
    1. Restore project from /root/my_repo/
    2. Apply patch (for masking) and directly delete F2P test files
    3. Delete F2P test files 
    4. Reinitialize git
    5. Apply PR_0 and/or PR_1
    """

    instance_id = instance["instance_id"]

    progress_manager.update_instance_status(instance_id, "Restoring project from /root/my_repo/")
    restore_project_command = "rm -rf /testbed/* && cp -r /root/my_repo/* /testbed/ && rm -rf /root/my_repo"
    res = env.execute(restore_project_command)

    print("Restoring project:", res)

    # 2. Apply patch (for masking) and directly delete F2P test files
    progress_manager.update_instance_status(instance_id, "Applying patch (for masking) and directly deleting F2P test files")
    patch_content = _normalize_patch_content(instance["patch"])
    patch_file = Path(instance_dir  / "patch_masking.diff")
    patch_file.write_text(patch_content)
    env.apply_patch(patch_file)
    patch_file.unlink()

    # 3. Delete F2P test files
    progress_manager.update_instance_status(instance_id, "Deleting F2P test files")
    fail_to_pass = instance["FAIL_TO_PASS"]

    if fail_to_pass:
        f2p_tests = fail_to_pass if isinstance(fail_to_pass, list) else [fail_to_pass]
        for f2p_test in f2p_tests:
            # Ensure path starts with /testbed/
            if not f2p_test.startswith('/testbed/'):
                f2p_test_path = f'/testbed/{f2p_test}'
            else:
                f2p_test_path = f2p_test

            logger.info(f"Deleting F2P test file: {f2p_test_path}")
            result = env.execute(f"rm -rf {f2p_test_path}")

            if result['returncode'] != 0:
                logger.error(f"Failed to delete F2P test file: {f2p_test_path} for instance {instance_id}. Result: {result}")
                raise Exception(f"Failed to delete F2P test file: {f2p_test_path}")

    else:
        raise NoFailToPassError(f"No F2P test files to delete for instance {instance_id}")

    # 4. Reinitialize git
    progress_manager.update_instance_status(instance_id, "Reinitializing git")
    reinitialize_git_commands = [
        "rm -rf .git",
        "git init",
        "git config user.email 'fb@bench.com'",
        "git config user.name 'FeatureBench'",
        "git add -A",
        "git commit -m 'Initial commit for FeatureBench evaluation' --allow-empty",
    ]

    reinitialize_git_command = " && ".join(reinitialize_git_commands)
    env.execute(reinitialize_git_command)

    # 5. Apply PR_0 and/or PR_1
    if files_to_change is not None:
        for file_to_change in files_to_change:
            host_file = list( file_to_change.keys() )[0]
            target_path = list( file_to_change.values() )[0]
            env.copy_to_container(host_file, target_path)

    if init_patch is not None:
        env.apply_patch(init_patch)


def _normalize_patch_content(patch_content: str) -> str:
    # Some LLM outputs omit the final newline; `git apply` may report
    # `error: corrupt patch at line ...` when the patch file ends abruptly.
    if patch_content and not patch_content.endswith("\n"):
        return patch_content + "\n"
    return patch_content

def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_only_eval: bool,
    path_local_images: Path | None,
    task_column_name:str = "problem_statement"
) -> None:
    """Process a single FeatureBench instance.

    Level 1 workflow:
    1. Activate conda environment
    2. Restore project from /root/my_repo/
    3. Apply patch (for masking) and directly delete F2P test files
    4. Reinitialize git
    5. Apply agent's patch
    6. Delete any generated F2P files, restore F2P test, and run tests

    Args:
        instance: Instance data (from HuggingFace)
        pred: Prediction data with patch
        container: Docker container
        logger: Logger instance
        log_dir: Directory to save test outputs
        timeout: Test timeout in seconds

    Returns:
        Dictionary with evaluation results
    """

    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    test_output_path = Path( instance_dir / "test_output.txt" )
    patch_file = Path(instance_dir  / "patch.diff")

    if not instance_dir.exists():
        instance_dir.mkdir(parents=True, exist_ok=True)

    # avoid inconsistent state if something here fails and there's leftover previous files
    if not run_only_eval:
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    
    model = get_model(config=config.get("model", {}))
    task = instance[task_column_name] # instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    extra_info = None

    try:
        repo_settings = parse_repo_settings(instance)
        docker_runtime_config = get_docker_runtime_config(repo_settings)
        env = get_sb_environment(config, instance, path_local_images, docker_runtime_config, skip_git_reset=run_only_eval)

        setup_environment(instance, instance_dir, env, init_patch = instance.get('init_patch', None), files_to_change = instance.get('files_to_change', None), progress_manager=progress_manager)

        if not run_only_eval:
            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )
            exit_status, result = agent.run(task)

        else:
            result = get_from_preds_file( output_dir / "preds.json", instance_id)
            if result is None:
                logger.warning(f"{instance_id} prediction does not exists in preds.json")
                return

            progress_manager.update_instance_status(instance_id, "applying patch for eval")
            env.apply_patch(patch_file)

        if len( result.strip() ) > 0:
            # Save patch
            with open(patch_file, "w", encoding="utf-8") as f:
                result = _normalize_patch_content(result)
                f.write(result)

            progress_manager.update_instance_status(instance_id, "Evaluating the patch")
            test_config = get_test_config_from_repo_settings(repo_settings)
            exit_status = evaluation( instance, instance_dir, result, env, test_config )

            progress_manager.update_instance_status(instance_id, "Calculating maintainability metrics")
            maintainability_metrics( instance, instance_dir, env )

    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        traceback.print_exc()
        exit_status, result = type(e).__name__, str(e)
        if exit_status == subprocess.CalledProcessError:
            raise Exception("CalledProcessError. Retrying!")

        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if not run_only_eval:
            save_traj(
                agent,
                instance_dir / f"{instance_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
                print_fct=logger.info,
            )
            update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)

def filter_instances(
        instances: list[dict], *, filter_spec: str, filter_ids : str, slice_spec: str = "", shuffle: bool = False, is_sanity_run : bool = False, use_corruption : bool = False, run_only_eval : bool = False, init_patch_map: str | None = None
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]

    if filter_ids is not None and len(filter_ids) > 0:
        filter_ids = filter_ids.split("|")
        instances = [instance for instance in instances if instance['instance_id'] in filter_ids]


    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")

    return instances

def update_instances(instances: list[dict], is_sanity_run: bool = False, use_corruption: bool = False, init_patch_map: str | None = None) -> list[dict]:
    """Update the instances setup arguments such as init patch"""

    if is_sanity_run:
        with open( Path( SANITY_CORRUPT_FOLDER ) / "instance_file_mapping.json" ) as ofile:
            corruption_file_mapping = json.load(ofile)

        new_corruption_file_mapping = {}
        for key, value in corruption_file_mapping.items():
            new_mapped_values = []
            for _map in value:
                container_file = next( iter( _map ) )
                corrupted_file = _map[ container_file ].replace("^", "")

                new_mapped_values.append( { str( Path( SANITY_CORRUPT_FOLDER ) / corrupted_file) : str( Path( CONTAINER_BASE_PATH ) / container_file)  } )

            new_corruption_file_mapping[ key ] = new_mapped_values
                

        corruption_file_mapping = new_corruption_file_mapping

        instances = [
                {
                    **ex,
                    "files_to_change": corruption_file_mapping[ ex["instance_id"] ] if use_corruption else None
                }
                for ex in instances 
                if ex["instance_id"] in corruption_file_mapping.keys()
        ]

        if len( instances ) == len( corruption_file_mapping.keys() ):
            logger.warning(f"corruption file mapping and instances do not match in length. Instances: {len(instances)}. corruptions: {len(corruption_file_mapping.keys())}")

    if init_patch_map is not None:
        with open(init_patch_map) as ofile:
            init_patch_dict = json.load( ofile )

        instances = [
                  {
                      **ex,
                      "init_patch": Path( init_patch_dict[ ex["instance_id"] ] )
                  }
                  for ex in instances 
                  if ex["instance_id"] in init_patch_dict.keys()
          ]

    return instances

def handle_gold_test(instances: list[dict], output_path: Path) -> None:
    write_dict = {}
    for instance in instances:
        gold_patch = instance['patch']
        instance_id = instance["instance_id"]
        model_name_or_path = "gold"
        write_dict[ instance_id ] = {
                "model_name_or_path" : model_name_or_path,
                "instance_id" : instance_id,
                "model_patch": gold_patch
        }

    with open( output_path / "preds.json", "w") as ofile:
            json.dump( write_dict, ofile, indent=4 )
# fmt: off

#TODO: Remove not used arguments to make it cleaner.
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    filter_ids: str = typer.Option("", "--filter-ids", help="Filter by instance IDs. Delimiter is '|'", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    is_sanity_run : bool = typer.Option( False, "--is-sanity-run", help="Sanity run", rich_help_panel="Sanity run"),
    use_corruption : bool = typer.Option( False, "--use-corruption", help="use corruption", rich_help_panel="use corruption"),
    run_only_eval : bool = typer.Option( False, "--run-only-eval", help="run evals only", rich_help_panel="only run eval and not the agent."),
    generate_final_report: bool = typer.Option( False, "--generate-final-report", help="run final report generation only", rich_help_panel="only run final report."),
    gold : bool = typer.Option( False, "--gold", help="Sanity run gold patch", rich_help_panel="Sanity run"),
    path_local_images: str | None = typer.Option(None, "--path-local-images", help="Path to local images dir", rich_help_panel="Sanity run"),
    init_patch_map: str | None = typer.Option(None, "--init-patch-map", help="Patch to apply to sandbox during init", rich_help_panel="Sanity run"),
    task_column_name: str | None = typer.Option(None, "--task-column-name", help="Task problem statement from the dataset.", rich_help_panel="Sanity run"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")
    path_local_images = Path( path_local_images ) if path_local_images is not None else None

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")

    if subset == "custom":
        instances = list(load_dataset("csv", data_files=dataset_path, split=split))
        for i in instances:
            i["version"] = str( i["version"] )
    else:
        instances = list(load_dataset(dataset_path, split=split))


    instances = update_instances(instances, use_corruption=use_corruption, init_patch_map=init_patch_map)
    instances = filter_instances(instances, filter_spec=filter_spec, filter_ids = filter_ids, slice_spec=slice_spec, shuffle=shuffle, is_sanity_run=is_sanity_run, use_corruption=use_corruption, run_only_eval=run_only_eval, init_patch_map=init_patch_map)

    #TODO: Convert all the print to logger.info()
    logger.info(f"Total Instances: {len(instances)}")

    #TODO: Handle gold dry-run slightly different as we only require to write the preds.json file.
    if gold:
        logger.info("Creating gold test predictions")
        handle_gold_test(instances, output_path)
        

    ######
    
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config = yaml.safe_load(get_config_path(config_spec).read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    if task_column_name is None:
        task_column_name = "problem_statement"

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")
                traceback.print_exc()
                progress_manager.on_uncaught_exception(instance_id, e)

    if generate_final_report:
        logger.info("Creating final report")

        with open( output_path / "preds.json") as ofile:
            predictions = json.load( ofile )

        make_run_report(predictions, instances, output_path)
        return

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager, run_only_eval, path_local_images, task_column_name): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    with open( output_path / "preds.json") as ofile:
        predictions = json.load( ofile )

    logger.info("Creating final report")
    make_run_report(predictions, instances, output_path)


if __name__ == "__main__":
    app()
