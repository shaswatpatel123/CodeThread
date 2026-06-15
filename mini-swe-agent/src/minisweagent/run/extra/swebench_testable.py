#!/usr/bin/env python3

"""Run mini-SWE-agent (TestableSWEAgent) on SWE-bench instances in batch mode.

This is a drop-in replacement for swebench.py that swaps DefaultAgent for
TestableSWEAgent, giving the agent the ability to trigger mid-loop test
execution via ``echo RUN_TESTS_AND_REPORT``.
"""

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
from minisweagent.agents.testable import TestableSWEAgent, get_evaluator
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.extra.utils.maintainability import analyze
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

from minisweagent.evaluation.swebench import evaluation, report
from minisweagent.evaluation.swebench_multilingual import evaluation as evaluation_multilingual
from minisweagent.evaluation.reporting import make_run_report
from minisweagent.evaluation.maintainability_metrics import maintainability_metrics

# Re-use helpers from the standard swebench runner to avoid duplication
from minisweagent.run.extra.swebench import (
    get_swebench_docker_image_name,
    get_sb_environment,
    update_preds_file,
    remove_from_preds_file,
    get_from_preds_file,
    filter_instances,
    update_instances,
    handle_gold_test,
    merge_instances,
    DATASET_MAPPING,
    SANITY_CORRUPT_FOLDER,
    CONTAINER_BASE_PATH,
)

_HELP_TEXT = """Run mini-SWE-agent (TestableSWEAgent) on SWEBench instances.

The agent can request mid-loop test feedback by running:

  echo RUN_TESTS_AND_REPORT

It will receive back a list of still-failing FAIL_TO_PASS tests and any
PASS_TO_PASS regressions, then continue working.
"""

RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingTestableSWEAgent(TestableSWEAgent):
    """TestableSWEAgent with per-step progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_only_eval: bool,
    path_local_images: Path | None,
    task_column_name: str = "problem_statement",
    multilingual: bool = False,
    evaluator_name: str = "swebench",
) -> None:
    """Process a single SWEBench instance with TestableSWEAgent."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    test_output_path = Path(instance_dir / "test_output.txt")
    patch_file = Path(instance_dir / "patch.diff")

    if not instance_dir.exists():
        instance_dir.mkdir(parents=True, exist_ok=True)

    if multilingual and instance.get("PR0_Patch"):
        pr_0_file = Path(instance_dir / "pr_0_patch.diff")
        pr_0_file.write_text(instance["PR0_Patch"])
        old = instance.get("init_patch")
        if old:
            instance["init_patch"] = [pr_0_file, *old]
        else:
            instance["init_patch"] = [pr_0_file]
        print(instance["instance_id"], "pr_0_patch.diff done")

    if not run_only_eval:
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance[task_column_name]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting container")

    agent = None
    extra_info = None

    try:
        env = get_sb_environment(config, instance, path_local_images, multilingual)

        if not run_only_eval:
            env.reset_repo()

            evaluator = get_evaluator(evaluator_name)

            agent = ProgressTrackingTestableSWEAgent(
                model,
                env,
                instance=instance,
                evaluator=evaluator,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )
            exit_status, result = agent.run(task)

        else:
            result = get_from_preds_file(output_dir / "preds.json", instance_id)
            if result is None:
                logger.warning(f"{instance_id} prediction does not exist in preds.json")
                return

        if len(result.strip()) > 0:
            progress_manager.update_instance_status(instance_id, "Evaluating the patch")
            if multilingual:
                exit_status = evaluation_multilingual(instance, instance_dir, result, env)
            else:
                evaluation(instance, instance_dir, result, env)
                exit_status = report(instance, instance_dir, result, test_output_path)

            progress_manager.update_instance_status(instance_id, "Calculating maintainability metrics")
            maintainability_metrics(instance, instance_dir, env)

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
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use", rich_help_panel="Advanced"),
    is_sanity_run: bool = typer.Option(False, "--is-sanity-run", help="Sanity run", rich_help_panel="Sanity run"),
    use_corruption: bool = typer.Option(False, "--use-corruption", help="Use corruption", rich_help_panel="Sanity run"),
    run_only_eval: bool = typer.Option(False, "--run-only-eval", help="Run evals only", rich_help_panel="Eval only"),
    generate_final_report: bool = typer.Option(False, "--generate-final-report", help="Run final report generation only", rich_help_panel="Eval only"),
    gold: bool = typer.Option(False, "--gold", help="Use gold patch", rich_help_panel="Sanity run"),
    path_local_images: str | None = typer.Option(None, "--path-local-images", help="Path to local images dir", rich_help_panel="Advanced"),
    init_patch_map: str | None = typer.Option(None, "--init-patch-map", help="Patch to apply to sandbox during init", rich_help_panel="Advanced"),
    task_column_name: str | None = typer.Option(None, "--task-column-name", help="Task problem statement column", rich_help_panel="Advanced"),
    multilingual: bool = typer.Option(False, "--multilingual", help="Run multilingual evaluation", rich_help_panel="Multilingual"),
    user_custom: bool = typer.Option(False, "--user-custom", help="Load custom dataset via CUSTOM_DATA_PATH", rich_help_panel="Multilingual"),
    evaluator_name: str = typer.Option("auto", "--evaluator", help="Evaluator for mid-loop tests: auto | swebench | multilingual | featbench | swebenchpro. 'auto' picks multilingual when --multilingual is set, otherwise swebench.", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")
    path_local_images = Path(path_local_images) if path_local_images is not None else None

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")

    if user_custom:
        dataset_path_custom = DATASET_MAPPING.get("custom")
        instances_custom = list(load_dataset("csv", data_files=dataset_path_custom, split="train"))
        for i in instances_custom:
            i["version"] = str(i["version"])

    instances = list(load_dataset(dataset_path, split=split))

    if user_custom:
        instances = merge_instances(instances, instances_custom)

    instances = update_instances(instances, is_sanity_run=is_sanity_run, use_corruption=use_corruption, init_patch_map=init_patch_map)
    instances = filter_instances(instances, filter_spec=filter_spec, filter_ids=filter_ids, slice_spec=slice_spec, shuffle=shuffle, is_sanity_run=is_sanity_run, use_corruption=use_corruption, run_only_eval=run_only_eval, init_patch_map=init_patch_map)

    logger.info(f"Total Instances: {len(instances)}")

    if gold:
        logger.info("Creating gold test predictions")
        handle_gold_test(instances, output_path)

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

    if evaluator_name == "auto":
        evaluator_name = "multilingual" if multilingual else "swebench"
    logger.info(f"Using evaluator: {evaluator_name}")

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
        with open(output_path / "preds.json") as ofile:
            predictions = json.load(ofile)
        make_run_report(predictions, instances, output_path)
        return

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    run_only_eval,
                    path_local_images,
                    task_column_name,
                    multilingual,
                    evaluator_name,
                ): instance["instance_id"]
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

    with open(output_path / "preds.json") as ofile:
        predictions = json.load(ofile)

    logger.info("Creating final report")
    make_run_report(predictions, instances, output_path)


if __name__ == "__main__":
    app()
