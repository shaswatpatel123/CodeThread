from pathlib import Path
import json

def maintainability_metrics( instance: dict, instance_dir: Path, env ) -> dict:
    maintainability_metrics_file = instance_dir / "maintainability_metrics.json"
    modified_functions_file = instance_dir / "modified_functions.json"

    maintainability_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    modified_functions_file.parent.mkdir(parents=True, exist_ok=True)

    maintainability_metrics = env.get_maintainability_metrics(instance["instance_id"])
    with open( maintainability_metrics_file, "w" ) as ofile:
        json.dump( maintainability_metrics, ofile )

    modified_functions = env.get_modified_functions_mapping(instance["instance_id"])
    with open( modified_functions_file, "w" ) as ofile:
        json.dump( modified_functions, ofile )
