"""
    Generate secondPR.json which maps instance_id to it's repective patch.diff.
    This patch.diff is used to initialize the container.
"""
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        exit("Too less arguments calling script. get_instance_ids.py <path_to_results>")

    base_path = Path( sys.argv[1] )
    synthetic_report_path = base_path / "synthetic_report.json"
    if not synthetic_report_path.exists():
        exit(f"{synthetic_report_path} does not exists!")

    # Read the report and extract the instance_ids
    with open( synthetic_report_path ) as ofile:
        data = json.load( ofile )

    instance_ids = data['success']

    second_pr_json = { k : str( Path(f"{base_path}/{k}/patch.diff").resolve() ) for k in instance_ids }

    with open( base_path / "secondPRMapper.json", "w" ) as ofile:
        json.dump( second_pr_json, ofile )
