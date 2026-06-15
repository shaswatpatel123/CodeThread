"""Check whether sythentic permutation of the first PR changed the PASS2PASS and FAIL2PASS"""

import json
from pathlib import Path

file_path = Path( "../../results/synthetic_claude_first_2_medium" )
subdirectories = [item for item in file_path.iterdir() if item.is_dir()]

report = {}
report['success_all'] = []
report['success_all_reg_and_few_f2p'] = []
report['success_all_reg_and_no_f2p'] = []
report['fail'] = []

for subdir in subdirectories:
    if not (subdir / "report.json").exists():
        continue
    with open( subdir / "report.json" ) as ofile:
        data = json.load( ofile )

        instance_id = list(data.keys())[0]

        if "tests_status" not in data[ instance_id ]:
            report['fail'].append( instance_id )
            continue

        tests_status = data[ instance_id ]["tests_status"]
        p2p = tests_status['PASS_TO_PASS']
        f2p = tests_status['FAIL_TO_PASS']

        if len( p2p['failure'] ) == 0:
            if len( f2p['failure'] ) == 0:
                report['success_all'].append( instance_id )
            elif len( f2p['failure'] ) != 0 and len( f2p['success'] ) != 0:
                report['success_all_reg_and_few_f2p'].append( instance_id )
            elif len( f2p['success'] ) == 0:
                report['success_all_reg_and_no_f2p'].append( instance_id )
        else:
            report['fail'].append( instance_id )


with open(file_path / "synthetic_report.json", "w") as ofile:
    json.dump( report, ofile )

print("Failure:", len(report['fail']) )
print("Success all regression tests and f2p:", len(report['success_all']) )
print("Success all regression tests and few f2p:", len(report['success_all_reg_and_few_f2p']) )
print("Success all regression tests and no f2p", len(report['success_all_reg_and_no_f2p']) )
