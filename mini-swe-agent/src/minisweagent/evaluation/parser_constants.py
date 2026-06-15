import re
from minisweagent.evaluation.python_constants import TestStatus


def parse_log_pytest(log: str, test_spec: dict) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework
    Args:
        log (str): log content

    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")

            test_case = line.split()
            if len(test_case) <= 1:
                continue

            test_status_map[test_case[1]] = test_case[0]
    return test_status_map

def parse_log_pytest_options(log: str, test_spec: dict) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content

    Returns:
        dict: test case to test status mapping
    """

    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}

    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")

            test_case = line.split()

            if len(test_case) <= 1:
                continue

            has_option = option_pattern.search(test_case[1])

            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith("/")
                    and not option.startswith("//")
                    and "*" not in option
                ):
                    option = "/" + option.split("/")[-1]

                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]

            test_status_map[test_name] = test_case[0]
    return test_status_map

def parse_log_django(log: str, test_spec: dict) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content

    Returns:
        dict: test case to test status mapping
    """

    test_status_map = {}
    lines = log.split("\n")
    prev_test = None

    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = (
                TestStatus.PASSED.value
            )

        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]



        pass_suffixes = (" ... ok", " ... OK", " ...  OK")

        for suffix in pass_suffixes:
            if line.endswith(suffix):

                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line

                if line.strip().startswith(
                    "Applying sites.0002_alter_domain_unique...test_no_migrations"
                ):

                    line = line.split("...", 1)[-1].strip()

                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value

                break

        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value

        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value

        if line.startswith("FAIL:"):

            test = line.split()[1].strip()

            test_status_map[test] = TestStatus.FAILED.value

        if line.endswith(" ... ERROR"):

            test = line.split(" ... ERROR")[0]

            test_status_map[test] = TestStatus.ERROR.value

        if line.startswith("ERROR:"):

            test = line.split()[1].strip()

            test_status_map[test] = TestStatus.ERROR.value



        if line.lstrip().startswith("ok") and prev_test is not None:

            # It means the test passed, but there's some additional output (including new lines)

            # between "..." and "ok" message

            test = prev_test

            test_status_map[test] = TestStatus.PASSED.value



    # TODO: This is very brittle, we should do better

    # There's a bug in the django logger, such that sometimes a test output near the end gets

    # interrupted by a particular long multiline print statement.

    # We have observed this in one of 3 forms:

    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"

    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"

    # - "{test_name} ... System check identified no issues (0 silenced).\nok"

    patterns = [

        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",

        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",

        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",

    ]

    for pattern in patterns:

        for match in re.finditer(pattern, log, re.MULTILINE):

            test_name = match.group(1)

            test_status_map[test_name] = TestStatus.PASSED.value

    return test_status_map





def parse_log_pytest_v2(log: str, test_spec: dict) -> dict[str, str]:

    """

    Parser for test logs generated with PyTest framework (Later Version)



    Args:

        log (str): log content

    Returns:

        dict: test case to test status mapping

    """

    test_status_map = {}

    escapes = "".join([chr(char) for char in range(1, 32)])

    for line in log.split("\n"):

        line = re.sub(r"\[(\d+)m", "", line)

        translator = str.maketrans("", "", escapes)

        line = line.translate(translator)

        if any([line.startswith(x.value) for x in TestStatus]):

            if line.startswith(TestStatus.FAILED.value):

                line = line.replace(" - ", " ")

            test_case = line.split()

            if len(test_case) >= 2:

                test_status_map[test_case[1]] = test_case[0]

        # Support older pytest versions by checking if the line ends with the test status

        elif any([line.endswith(x.value) for x in TestStatus]):

            test_case = line.split()

            if len(test_case) >= 2:

                test_status_map[test_case[0]] = test_case[1]

    return test_status_map





def parse_log_seaborn(log: str, test_spec: dict) -> dict[str, str]:

    """

    Parser for test logs generated with seaborn testing framework



    Args:

        log (str): log content

    Returns:

        dict: test case to test status mapping

    """

    test_status_map = {}

    for line in log.split("\n"):

        if line.startswith(TestStatus.FAILED.value):

            test_case = line.split()[1]

            test_status_map[test_case] = TestStatus.FAILED.value

        elif f" {TestStatus.PASSED.value} " in line:

            parts = line.split()

            if parts[1] == TestStatus.PASSED.value:

                test_case = parts[0]

                test_status_map[test_case] = TestStatus.PASSED.value

        elif line.startswith(TestStatus.PASSED.value):

            parts = line.split()

            test_case = parts[1]

            test_status_map[test_case] = TestStatus.PASSED.value

    return test_status_map





def parse_log_sympy(log: str, test_spec: dict) -> dict[str, str]:

    """

    Parser for test logs generated with Sympy framework



    Args:

        log (str): log content

    Returns:

        dict: test case to test status mapping

    """

    test_status_map = {}

    pattern = r"(_*) (.*)\.py:(.*) (_*)"

    matches = re.findall(pattern, log)

    for match in matches:

        test_case = f"{match[1]}.py:{match[2]}"

        test_status_map[test_case] = TestStatus.FAILED.value

    for line in log.split("\n"):

        line = line.strip()

        if line.startswith("test_"):

            if line.endswith(" E"):

                test = line.split()[0]

                test_status_map[test] = TestStatus.ERROR.value

            if line.endswith(" F"):

                test = line.split()[0]

                test_status_map[test] = TestStatus.FAILED.value

            if line.endswith(" ok"):

                test = line.split()[0]

                test_status_map[test] = TestStatus.PASSED.value

    return test_status_map



def parse_log_matplotlib(log: str, test_spec: dict) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """

    test_status_map = {}

    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")

        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()

            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map

def parse_log_pytest_verbose(log: str) -> dict[str, str]:
    """
    Parser for verbose test logs generated with PyTest framework.
    
    Parses pytest verbose output format where test results appear as:
    tests/test_file.py::test_name STATUS [percentage] [percentage]
    
    Args:
        log (str): log content
    Returns:
        dict[str, str]: test case to test status mapping
    """
    test_status_map = {}
    
    if not log or not log.strip():
        return test_status_map
    
    lines = log.split('\n')
    
    # Define the status values to look for
    statuses = [
        TestStatus.PASSED.value,
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
        TestStatus.XFAIL.value,
        TestStatus.SKIPPED.value
    ]
    
    for line in lines:
        line_stripped = line.strip()
        
        # Stop if we hit a separator line (starts with ===)
        # This indicates the end of test execution results and start of detailed output
        if line_stripped.startswith('===') and  ('test session starts' not in line_stripped):
            break
        
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Check if line contains any of the status keywords
        for status in statuses:
            # Look for status keyword surrounded by spaces or at end of relevant part
            if f' {status} ' in line_stripped or f' {status}\t' in line_stripped:
                # Extract test name (everything before the status)
                # Format: test_path::test_name STATUS [percentage] [percentage]
                parts = line_stripped.split(status, 1)
                if len(parts) >= 2:
                    test_name = parts[0].strip()
                    # Only record if it looks like a valid test path (contains ::)
                    # This filters out lines like "tests/test_file.py:8 test_name - message"
                    if '::' in test_name:
                        test_status_map[test_name] = status
                        break  # Found a status, move to next line
    
    return test_status_map

def parse_log_pytest_with_subfailed(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework, handling SUBFAILED duplicates.
    
    When the same test name appears multiple times in SUBFAILED entries,
    adds [1], [2], etc. suffixes to keep them distinct in the result dict.
    Other status types (PASSED, FAILED, ERROR, etc.) do not get numbered suffixes.
    
    Args:
        log (str): log content
    Returns:
        dict[str, str]: test case to test status mapping
    """
    test_status_map = {}
    subfailed_counts = {}  # Track SUBFAILED occurrences per test name

    if not log or not log.strip():
        return test_status_map

    lines = log.split('\n')

    # Find the "short test summary info" section
    summary_start_idx = -1
    for i, line in enumerate(lines):
        if 'short test summary info' in line.lower():
            summary_start_idx = i + 1  # Start from the next line
            break

    # If we didn't find the summary section, return empty result
    if summary_start_idx == -1:
        return test_status_map

    # Parse lines from the summary section until we hit a separator line or end
    for i in range(summary_start_idx, len(lines)):
        line = lines[i].strip()

        # Stop if we hit a separator line (starts with ===)
        if line.startswith('==='):
            break

        # Skip empty lines
        if not line:
            continue

        test_name = None
        status = None
        is_subfailed = False

        # Match PASSED
        if line.startswith(TestStatus.PASSED.value + ' '):
            match = re.match(r'PASSED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.PASSED.value

        # Match FAILED
        elif line.startswith(TestStatus.FAILED.value + ' '):
            match = re.match(r'FAILED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.FAILED.value

        # Match SUBFAILED (subtest failures, treat as FAILED)
        elif line.startswith('SUBFAILED'):
            match = re.match(r'SUBFAILED(?:\(<subtest>\))?\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.FAILED.value
                is_subfailed = True

        # Match ERROR
        elif line.startswith(TestStatus.ERROR.value + ' '):
            match = re.match(r'ERROR\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.ERROR.value

        # Match XFAIL (expected failure)
        elif line.startswith(TestStatus.XFAIL.value + ' '):
            match = re.match(r'XFAIL\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.XFAIL.value

        # Match SKIPPED
        elif line.startswith(TestStatus.SKIPPED.value + ' '):
            # Format: SKIPPED [1] test_example.py:15: reason
            match = re.match(r'SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.SKIPPED.value

        # If we found a test name and status, record it
        if test_name and status:
            # Only add numbered suffix for SUBFAILED entries
            if is_subfailed:
                if test_name in subfailed_counts:
                    subfailed_counts[test_name] += 1
                else:
                    subfailed_counts[test_name] = 1
                # Add suffix [1], [2], etc. for SUBFAILED
                unique_test_name = f"{test_name}[{subfailed_counts[test_name]}]"
            else:
                # For non-SUBFAILED, use test name as-is
                unique_test_name = test_name
            
            test_status_map[unique_test_name] = status

    return test_status_map

def parse_log_pytest_combined(log: str, test_spec: dict = None) -> dict[str, str]:
    """
    Parser that handles both pytest output formats:
    - Verbose (-vv): "test_path::test_name STATUS      [  4%]"  (status embedded, produced by -vv)
    - Summary (-rA): "STATUS test_path::test_name"              (status at start, produced by -rA)

    Runs both parsers and merges results, so it is robust to whichever command was used.
    """
    result = {}
    result.update(parse_log_pytest(log, test_spec))

    statuses = [s.value for s in TestStatus]
    for line in log.split("\n"):
        line_stripped = line.strip()
        for status in statuses:
            if f" {status} " in line_stripped or line_stripped.endswith(f" {status}"):
                parts = line_stripped.split(status, 1)
                if len(parts) >= 2:
                    test_name = parts[0].strip()
                    if "::" in test_name:
                        result[test_name] = status
                        break
    return result

parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER_PY = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
    "huggingface/transformers": parse_log_pytest,
    "pydantic/pydantic": parse_log_pytest_verbose,
    "python/mypy": parse_log_pytest,
    "pypa/setuptools": parse_log_pytest,
    "fastapi/fastapi": parse_log_pytest,
    "pypa/packaging": parse_log_pytest,
    "huggingface/trl": parse_log_pytest,
    "Netflix/metaflow": parse_log_pytest,
    "linkedin/Liger-Kernel": parse_log_pytest,
    "pandas-dev/pandas": parse_log_pytest,
    "Lightning-AI/pytorch-lightning": parse_log_pytest,
    "optuna/optuna": parse_log_pytest,
    "huggingface/accelerate": parse_log_pytest,
    "mesonbuild/meson": parse_log_pytest_with_subfailed,
    "mlflow/mlflow": parse_log_pytest,
    "pypa/hatch": parse_log_pytest,
}

MAP_REPO_TO_PARSER = {
    **MAP_REPO_TO_PARSER_PY
}

