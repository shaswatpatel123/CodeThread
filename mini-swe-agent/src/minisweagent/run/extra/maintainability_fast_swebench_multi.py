#!/usr/bin/env python3

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
from pathlib import Path, PurePosixPath

import typer
import yaml
from datasets import load_dataset
from rich.live import Live
from tenacity import retry, stop_after_attempt, wait_fixed
from unidiff import PatchSet

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.extra.utils.exceptions import PatchApplyError, PatchFileNotFound
from minisweagent.run.extra.utils.maintainability import analyze
from minisweagent.utils.log import add_file_handler, logger

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
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "swebenchpro": "ScaleAI/SWE-bench_Pro",

    "custom": os.getenv("CUSTOM_DATA_PATH") # "/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/synthetic_chains/swe_bench_first_task.csv"
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

def get_sb_environment(config: dict, instance: dict, path_local_images: Path | None, multilingual: bool = False) -> Environment:
    image_name = get_swebench_docker_image_name(instance, path_local_images)
    env_config = config.get("environment", {})
    if path_local_images is None:
        image_name = "docker://" + image_name
    else:
        image_name = path_local_images / image_name

    # print("IMAGE NAME:", image_name)
    env_config["image"] = image_name
    env_config['files_to_change'] = instance.get('files_to_change', None)
    env_config['init_patch'] = instance.get('init_patch', None)
    if multilingual:
        env_config['multilingual'] = True
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

def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    return  [x[2:] for x in source_files if x.startswith("a/")]

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_only_eval: bool,
    path_local_images: Path | None,
    oracle_first_pr: bool,
    gold_patch_path: str | None,
    swebench_patch_path: str | None,
    first_pr_patch_path: str | None,
    second_pr_patch_path: str | None,
    multilingual: bool = False,
) -> None:
    """Process a single SWEBench instance."""

    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id

    if not instance_dir.exists():
        instance_dir.mkdir(parents=True, exist_ok=True)

    pr_0_file = Path(instance_dir  / "pr_0_patch.diff")
    if multilingual and instance.get('PR0_Patch'):
        # It might have init_patch if it exists, then set it and write it in instance_dir
        # PR0_Patch
        pr_0_file.write_text( instance['PR0_Patch'] )
    
    gold_patch_file = Path(gold_patch_path)
    swebench_patch_file = Path(swebench_patch_path)
    first_pr_patch_file = Path(first_pr_patch_path)
    second_pr_patch_file = Path(second_pr_patch_path)
    
    language = instance.get("language", "python")

    gold_patch = gold_patch_file / instance_id / "patch.diff"

    swebench_patch = swebench_patch_file / instance_id / "patch.diff"

    first_pr_patch = first_pr_patch_file / instance_id / "patch_generated.diff"
    second_pr_patch = second_pr_patch_file / instance_id / "patch_generated.diff"

    if not gold_patch.exists():
        raise FileNotFoundError(f"Gold patch file not found: {gold_patch}")
    if not swebench_patch.exists():
        raise FileNotFoundError(f"Swebench patch file not found: {swebench_patch}")
    if not first_pr_patch.exists():
        first_pr_patch = first_pr_patch_file / instance_id / "patch.diff"
        if not first_pr_patch.exists():
            raise FileNotFoundError(f"First PR patch file not found: {first_pr_patch}")
    if not second_pr_patch.exists():
        second_pr_patch = second_pr_patch_file / instance_id / "patch.diff"
        if not second_pr_patch.exists():
            raise FileNotFoundError(f"Second PR patch file not found: {second_pr_patch}")

    
    # patch_file = Path(instance_dir  / "patch_generated.diff")
    # complexity_results_file = Path(instance_dir / "complexity_analysis.json")
    modified_functions_file = Path(instance_dir / "modified_functions.json")
    # maintainability_metrics_file = Path(instance_dir / "maintainability_metrics.json")
    gold_maintainability_metrics_file = Path(instance_dir / "gold_maintainability_metrics.json") # HH
    gold_modified_file_path = Path(instance_dir / "gold_modified_file.json")
    swebench_maintainability_metrics_file = Path(instance_dir / "swebench_maintainability_metrics.json") # HA
    swebench_modified_file_path = Path(instance_dir / "swebench_modified_file.json")
    first_pr_maintainability_metrics_file = Path(instance_dir / "first_pr_maintainability_metrics.json") # A
    first_pr_modified_file_path = Path(instance_dir / "first_pr_modified_file.json")
    second_pr_maintainability_metrics_file = Path(instance_dir / "second_pr_maintainability_metrics.json") # AA
    second_pr_modified_file_path = Path(instance_dir / "second_pr_modified_file.json")
    pr_0_maintainability_metrics_file = Path(instance_dir / "pr_0_maintainability_metrics.json") # H
    pr_0_modified_file_path = Path(instance_dir / "pr_0_modified_file.json")



    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    try:
        env = get_sb_environment(config, instance, path_local_images, multilingual)

        print( gold_patch )
        env.apply_patch( gold_patch )

        # Debug: check git state after applying gold patch
        _dbg = env.execute("git ls-files --others --modified --exclude-standard", timeout=60)
        print(f"[DEBUG] {instance_id} after gold_patch apply: returncode={_dbg['returncode']} output={_dbg['output']!r}")

        # Calculate maintainability metrics for gold patch
        gold_maintainability_metrics = env.get_maintainability_metrics(instance_id, language=language)
        gold_modified_files = env.get_modified_functions_mapping(instance_id, language=language)
        with open( gold_maintainability_metrics_file, "w" ) as ofile:
            json.dump( gold_maintainability_metrics, ofile )

        with open(gold_modified_file_path, "w") as ofile:
            json.dump(gold_modified_files, ofile )

        # Remove the changes
        env.reset_changes()
        env.apply_patch( swebench_patch )
        swebench_maintainability_metrics = env.get_maintainability_metrics(instance_id, language=language)
        swebench_modified_files = env.get_modified_functions_mapping(instance_id, language=language)
        with open( swebench_maintainability_metrics_file, "w" ) as ofile:
            json.dump( swebench_maintainability_metrics, ofile )

        with open(swebench_modified_file_path, "w") as ofile:
            json.dump( swebench_modified_files, ofile )

        env.reset_changes()

        # Now apply PR_0 patch (first element of init_patch)
        env.apply_patch(Path(pr_0_file))
        pr_0_maintainability_metrics= env.get_maintainability_metrics(instance_id, language=language)
        pro_0_modified_files = env.get_modified_functions_mapping(instance_id, language=language)
        with open( pr_0_maintainability_metrics_file, "w" ) as ofile:
            json.dump( pr_0_maintainability_metrics, ofile )

        with open(pr_0_modified_file_path, "w") as ofile:
            json.dump( pro_0_modified_files, ofile )

        # commit the changes
        env.git_commit("Changes for PR_0")

        # Now apply PR_1
        print( "first_pr_patch:",  first_pr_patch, flush=True )
        env.apply_patch( first_pr_patch )
        first_pr_maintainability_metrics = env.get_maintainability_metrics(instance_id, language=language)
        first_pr_modified_files = env.get_modified_functions_mapping(instance_id, language=language)
        with open( first_pr_maintainability_metrics_file, "w" ) as ofile:
            json.dump( first_pr_maintainability_metrics, ofile )

        with open(first_pr_modified_file_path, "w") as ofile:
            json.dump( first_pr_modified_files, ofile )
        
        # commit the changes
        env.git_commit("Changes for PR_1")

        # Now apply PR_2
        print("second_pr_patch", second_pr_patch)
        env.apply_patch( second_pr_patch )
        second_pr_maintainability_metrics = env.get_maintainability_metrics(instance_id, language=language)
        second_pr_modified_files = env.get_modified_functions_mapping(instance_id, language=language)

        with open( second_pr_maintainability_metrics_file, "w" ) as ofile:
            json.dump( second_pr_maintainability_metrics, ofile )

        with open(second_pr_modified_file_path, "w") as ofile:
            json.dump( second_pr_modified_files, ofile )

        exit_status = "Completed"

    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        traceback.print_exc()
        exit_status, result = type(e).__name__, str(e)
        if exit_status == subprocess.CalledProcessError:
            raise Exception("CalledProcessError. Retrying!")

    finally:
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
    
    if is_sanity_run:
        # TODO: Read from config

        with open( Path( SANITY_CORRUPT_FOLDER ) / "instance_file_mapping.json" ) as ofile:
            corruption_file_mapping = json.load(ofile)

        new_corruption_file_mapping = {}
        for key, value in corruption_file_mapping.items():
            new_mapped_values = []
            # print(key, value)
            for _map in value:
                # print(_map)
                container_file = next( iter( _map ) )
                corrupted_file = _map[ container_file ].replace("^", "")

                new_mapped_values.append( { str( Path( SANITY_CORRUPT_FOLDER ) / corrupted_file) : str( Path( CONTAINER_BASE_PATH ) / container_file)  } )

            new_corruption_file_mapping[ key ] = new_mapped_values
                

        corruption_file_mapping = new_corruption_file_mapping

        # Adding files_to_change to SingularityEnv, so during __init__ the files can be moved
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
        # init_patch

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

def merge_instances(instances: list[dict], instances_custom: list[dict]) -> list[dict]:

    # Only take "PR_1_problem_statement" and "PR0_Patch"
    instance_id_to_custom_instance = { instance["instance_id"] : instance for instance in instances_custom }
    instance_id_to_instance = { instance["instance_id"] : instance for instance in instances }

    res_instances = []
    # We take only the instances that are in the custom dataset.
    for instance_id in instance_id_to_custom_instance.keys():
        if instance_id in instance_id_to_instance:
            instance = instance_id_to_instance[instance_id]
            instance["PR_1_problem_statement"] = instance_id_to_custom_instance[instance_id]["PR_1_problem_statement"]
            instance["PR0_Patch"] = instance_id_to_custom_instance[instance_id]["PR0_Patch"]
            if "language" in instance_id_to_custom_instance[instance_id]:
                instance["language"] = instance_id_to_custom_instance[instance_id]["language"]
            res_instances.append(instance)
        else:
            raise ValueError(f"Instance {instance_id} is not in the main dataset.")
    return res_instances

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
                      "init_patch": [ Path( init_patch_dict[ ex["instance_id"] ] ) ]
                  }
                  for ex in instances
                  if ex["instance_id"] in init_patch_dict.keys()
          ]

    return instances

# fmt: off
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
    run_chains : bool = typer.Option( False, "--run-chains", help="Sanity run chains", rich_help_panel="Sanity run"),
    gold : bool = typer.Option( False, "--gold", help="Sanity run gold patch", rich_help_panel="Sanity run"),
    path_local_images: str | None = typer.Option(None, "--path-local-images", help="Path to local images dir", rich_help_panel="Sanity run"),
    init_patch_map: str | None = typer.Option(None, "--init-patch-map", help="Patch to apply to sandbox during init", rich_help_panel="Sanity run"),
    task_column_name: str | None = typer.Option(None, "--task-column-name", help="Task problem statement from the dataset.", rich_help_panel="Sanity run"),

    oracle_first_pr: bool = typer.Option( False, "--oracle-first-pr", help="Oracle 1st PR", rich_help_panel="Sanity run"),
    gold_patch_path: str | None = typer.Option(None, "--gold-patch-path", help="Path to gold patch", rich_help_panel="Sanity run"),
    swebench_patch_path: str | None = typer.Option(None, "--swebench-patch-path", help="Path to swebench patch", rich_help_panel="Sanity run"),
    first_pr_patch_path: str | None = typer.Option(None, "--first-pr-patch-path", help="Path to first PR patch", rich_help_panel="Sanity run"),
    second_pr_patch_path: str | None = typer.Option(None, "--second-pr-patch-path", help="Path to second PR patch", rich_help_panel="Sanity run"),

    user_custom: bool = typer.Option(False, "--user-custom", help="Run multilingual evaluation", rich_help_panel="Multilingual evaluation"),
    multilingual: bool = typer.Option(False, "--multilingual", help="Run multilingual evaluation", rich_help_panel="Multilingual evaluation"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")
    path_local_images = Path( path_local_images ) if path_local_images is not None else None

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")

    if user_custom: 
        dataset_path_custom = DATASET_MAPPING.get("custom")
        instances_custom = list(load_dataset("csv", data_files=dataset_path_custom, split="train"))
        for i in instances_custom:
            i["version"] = str( i["version"] )

    instances = list(load_dataset(dataset_path, split=split))


    if user_custom:
        instances = merge_instances(instances, instances_custom)
        
    instances = update_instances(instances, is_sanity_run=is_sanity_run, use_corruption=use_corruption, init_patch_map=init_patch_map)
    instances = filter_instances(instances, filter_spec=filter_spec, filter_ids = filter_ids, slice_spec=slice_spec, shuffle=shuffle, is_sanity_run=is_sanity_run, use_corruption=use_corruption, run_only_eval=run_only_eval, init_patch_map=init_patch_map)

    if gold:
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

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_maintain_{time.time()}.yaml")

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


    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager, run_only_eval, path_local_images, oracle_first_pr, gold_patch_path, swebench_patch_path, first_pr_patch_path, second_pr_patch_path, multilingual): instance[
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


if __name__ == "__main__":
    app()

