#!/usr/bin/env python3
"""Run mini-SWE-agent on the foresight anticipation-gap benchmark (batch mode).

Each instance is a PR fix-chain PR1->PR2. The agent solves PR1 from clean intent
(starting at base_commit = pr1_base). Grading then runs two test sets against the
agent's solution and reports the anticipation gap:
  - PR1 F2P%   : did the agent do PR1's stated task (FAIL_TO_PASS at pr1_head).
  - Anticipation F2P% : did it also fix the future edge case (ANTICIPATION_FAIL_TO_PASS at pr2_head).
  - gap = PR1 F2P% - Anticipation F2P%.

Two phases (like swebench/featbench):
  Phase 1 (agent):  writes the agent's git diff to preds.json.
  Phase 2 (--run-only-eval): applies the patch in a .git-intact sandbox and grades
                             via the image-baked run_state.sh (see evaluation/foresight.py).

Gold sanity (--gold, no model): writes each instance's gold PR1 patch as the
prediction; grading should show PR1~100%, anticipation~0%, gap~100%.
"""

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

from minisweagent.evaluation.foresight import evaluate_foresight, make_foresight_report

_HELP_TEXT = "Run mini-SWE-agent on the foresight anticipation-gap benchmark (batch mode)."

DEFAULT_DATASET = "/scratch/spp9399/foresightCoder/data/foresight_dataset.jsonl"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "foresight": DEFAULT_DATASET,
}

# JSON-encoded list fields in the dataset that must be decoded to real lists.
_LIST_FIELDS = [
    "pr1_test_files", "anticipation_test_files",
    "FAIL_TO_PASS", "PASS_TO_PASS", "ANTICIPATION_FAIL_TO_PASS",
]

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """DefaultAgent that reports per-step progress."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def load_instances(dataset_path: str) -> list[dict]:
    """Load instances from a JSONL file.

    Uses a plain reader rather than HF ``load_dataset`` because the dataset has
    mixed-type columns (e.g. clarity_score is sometimes a number, sometimes a
    string) that Arrow's schema inference rejects.
    """
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_instances(instances: list[dict]) -> list[dict]:
    """Decode JSON-string list fields into real lists (idempotent)."""
    for inst in instances:
        for f in _LIST_FIELDS:
            v = inst.get(f)
            if isinstance(v, str):
                s = v.strip()
                inst[f] = json.loads(s) if s.startswith("[") else ([v] if v else [])
            elif v is None:
                inst[f] = []
    return instances


def get_foresight_image(instance: dict, path_local_images: Path | None) -> str | Path:
    """Resolve the SIF/docker image for an instance.

    Local SIFs are named ``<short>.sif`` where short = instance_id before "__"
    (e.g. attrs__1401-1417 -> attrs.sif). Falls back to the row's docker_image.
    """
    short = instance["instance_id"].split("__")[0]
    if path_local_images is not None:
        return path_local_images / f"{short}.sif"
    return "docker://" + instance["docker_image"]


def get_foresight_environment(config: dict, instance: dict, path_local_images: Path | None,
                              skip_git_reset: bool, init_patch: list | None = None) -> Environment:
    env_config = config.get("environment", {})
    env_config.setdefault("environment_class", "foresightSingularity")
    env_config["image"] = get_foresight_image(instance, path_local_images)
    env_config["base_commit"] = instance["base_commit"]
    env_config["skip_git_reset"] = skip_git_reset
    env_config["files_to_change"] = None
    # init_patch (round-1 patch for the round-2 experiment) is seeded on top of
    # base_commit before the .git strip; see ForesightSingularityEnvironment.
    env_config["init_patch"] = init_patch
    return get_environment(env_config, default_type="foresightSingularity")


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    with _OUTPUT_FILE_LOCK:
        data = json.loads(output_path.read_text()) if output_path.exists() else {}
        data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        data = json.loads(output_path.read_text())
        if instance_id in data:
            del data[instance_id]
            output_path.write_text(json.dumps(data, indent=2))


def get_from_preds_file(output_path: Path, instance_id: str):
    if not output_path.exists():
        raise FileNotFoundError(f"output_path does not exist - {output_path}")
    with _OUTPUT_FILE_LOCK:
        data = json.loads(output_path.read_text())
        entry = data.get(instance_id)
        return entry["model_patch"] if entry else None


def _run_continue(agent: "ProgressTrackingAgent", task: str, resume_messages: list) -> tuple[str, str]:
    """Continue-mode round-2: resume a prior trajectory's messages, append the new
    edge-case task as a user turn, and keep stepping. Mirrors DefaultAgent.run() but
    does NOT reset self.messages (so the round-1 conversation is preserved)."""
    from minisweagent.agents.default import NonTerminatingException, TerminatingException
    agent.extra_template_vars |= {"task": task}
    agent.messages = list(resume_messages)
    agent.add_message("user", task)
    while True:
        try:
            agent.step()
        except NonTerminatingException as e:
            agent.add_message("user", str(e))
        except TerminatingException as e:
            agent.add_message("user", str(e))
            return type(e).__name__, str(e)


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_only_eval: bool,
    path_local_images: Path | None,
    task_column_name: str = "problem_statement",
    init_patch_map: dict | None = None,
    task_suffix: str = "",
    resume_traj_dir: Path | None = None,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    patch_file = instance_dir / "patch.diff"
    instance_dir.mkdir(parents=True, exist_ok=True)

    if not run_only_eval:
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    # Round-2: seed the round-1 patch as init_patch (applied on top of base_commit).
    init_patch = None
    if init_patch_map and instance_id in init_patch_map:
        init_patch = [Path(init_patch_map[instance_id])]
    elif init_patch_map:
        # round-2 requested but this instance has no round-1 patch -> nothing to build on
        logger.warning(f"{instance_id}: not in init_patch_map (empty round-1 patch?), skipping")
        progress_manager.on_instance_end(instance_id, "no_round1_patch")
        return

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "starting container")

    agent = None
    model = None
    extra_info = None
    exit_status = "None"
    result = ""

    try:
        # Agent phase strips .git (skip_git_reset=False); grade phase keeps it (True).
        env = get_foresight_environment(config, instance, path_local_images,
                                        skip_git_reset=run_only_eval, init_patch=init_patch)

        if not run_only_eval:
            model = get_model(config=config.get("model", {}))
            agent = ProgressTrackingAgent(
                model, env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )
            task = instance[task_column_name] + task_suffix
            if resume_traj_dir is not None:
                # Continue-mode: resume round-1 messages, then append the edge-case task.
                traj_file = resume_traj_dir / instance_id / f"{instance_id}.traj.json"
                resume_msgs = json.loads(traj_file.read_text()).get("messages", [])
                exit_status, result = _run_continue(agent, task, resume_msgs)
            else:
                exit_status, result = agent.run(task)
        else:
            result = get_from_preds_file(output_dir / "preds.json", instance_id)
            if result is None:
                logger.warning(f"{instance_id}: no prediction in preds.json")
                progress_manager.on_instance_end(instance_id, "no_prediction")
                return
            progress_manager.update_instance_status(instance_id, "applying patch")
            patch_file.write_text(result)
            env.apply_patch(patch_file)

        if run_only_eval and len(result.strip()) > 0:
            progress_manager.update_instance_status(instance_id, "grading (PR1 + anticipation)")
            exit_status = evaluate_foresight(instance, instance_dir, env)

    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        traceback.print_exc()
        exit_status, result = type(e).__name__, str(e)
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
            model_name = model.config.model_name if model is not None else "unknown"
            update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(instances: list[dict], *, filter_spec: str, filter_ids: str,
                     slice_spec: str = "", shuffle: bool = False) -> list[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before = len(instances)
    if filter_spec:
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
    if filter_ids:
        ids = set(filter_ids.split("|"))
        instances = [i for i in instances if i["instance_id"] in ids]
    if (after := len(instances)) != before:
        logger.info(f"Instance filter: {before} -> {after} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    return instances


def handle_gold_test(instances: list[dict], output_path: Path) -> None:
    """Write gold PR1 patches as predictions (sanity baseline, no model)."""
    write_dict = {}
    for inst in instances:
        write_dict[inst["instance_id"]] = {
            "model_name_or_path": "gold",
            "instance_id": inst["instance_id"],
            "model_patch": inst["patch"],
        }
    (output_path / "preds.json").write_text(json.dumps(write_dict, indent=4))


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("foresight", "--subset", help="Dataset name or path to a jsonl", rich_help_panel="Data selection"),
    split: str = typer.Option("train", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice spec, e.g. '0:5'", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    filter_ids: str = typer.Option("", "--filter-ids", help="Filter by instance IDs (delimiter '|')", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Worker threads", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "foresight.yaml", "-c", "--config", help="Config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type", rich_help_panel="Advanced"),
    run_only_eval: bool = typer.Option(False, "--run-only-eval", help="Grade existing predictions only", rich_help_panel="Eval"),
    generate_final_report: bool = typer.Option(False, "--generate-final-report", help="Aggregate reports only", rich_help_panel="Eval"),
    gold: bool = typer.Option(False, "--gold", help="Write gold PR1 patches as predictions (sanity)", rich_help_panel="Eval"),
    path_local_images: str | None = typer.Option(None, "--path-local-images", help="Dir of local SIF images", rich_help_panel="Basic"),
    init_patch_map: str | None = typer.Option(None, "--init-patch-map", help="Round-2: JSON {instance_id: round-1 patch path} seeded on top of base_commit", rich_help_panel="Round2"),
    edgecase_prompt_file: str | None = typer.Option(None, "--edgecase-prompt-file", help="Round-2: file whose text is appended to the problem statement (edge-case hunt prompt)", rich_help_panel="Round2"),
    resume_traj_dir: str | None = typer.Option(None, "--resume-traj-dir", help="Round-2 continue-mode: dir of round-1 results; resumes <iid>/<iid>.traj.json messages", rich_help_panel="Round2"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")
    path_local_images = Path(path_local_images) if path_local_images is not None else None

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = load_instances(dataset_path)
    instances = normalize_instances(instances)
    instances = filter_instances(instances, filter_spec=filter_spec, filter_ids=filter_ids,
                                 slice_spec=slice_spec, shuffle=shuffle)
    logger.info(f"Total instances: {len(instances)}")

    config = yaml.safe_load(get_config_path(config_spec).read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    # Round-2 elicited-anticipation inputs (all optional).
    init_patch_dict = json.loads(Path(init_patch_map).read_text()) if init_patch_map else None
    task_suffix = ("\n\n" + Path(edgecase_prompt_file).read_text()) if edgecase_prompt_file else ""
    resume_dir = Path(resume_traj_dir) if resume_traj_dir else None
    if init_patch_dict is not None:
        logger.info(f"Round-2 mode: init_patch_map has {len(init_patch_dict)} entries"
                    f"{' (continue-mode)' if resume_dir else ' (fresh-mode)'}")

    if gold:
        logger.info("Writing gold PR1 predictions")
        handle_gold_test(instances, output_path)

    # --generate-final-report is a terminal short-circuit ONLY when we're not also
    # grading. If combined with --run-only-eval, fall through to grade first; the
    # grading pass calls make_foresight_report() itself at the end.
    if generate_final_report and not run_only_eval:
        predictions = json.loads((output_path / "preds.json").read_text())
        make_foresight_report(predictions, instances, output_path)
        return

    if not redo_existing and not run_only_eval and (output_path / "preds.json").exists():
        existing = set(json.loads((output_path / "preds.json").read_text()).keys())
        instances = [i for i in instances if i["instance_id"] not in existing]
        logger.info(f"Skipping {len(existing)} existing; running {len(instances)}")

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                iid = futures[future]
                print(f"Error in future for {iid}: {e}")
                traceback.print_exc()
                progress_manager.on_uncaught_exception(iid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, inst, output_path, config, progress_manager,
                                run_only_eval, path_local_images, "problem_statement",
                                init_patch_dict, task_suffix, resume_dir): inst["instance_id"]
                for inst in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    predictions = json.loads((output_path / "preds.json").read_text())
    logger.info("Creating final report")
    make_foresight_report(predictions, instances, output_path)


if __name__ == "__main__":
    app()
