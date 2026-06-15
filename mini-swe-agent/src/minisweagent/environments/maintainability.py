#!/usr/bin/env python3

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from minisweagent.environments.utils.calculate_metrics import calculate_metrics
from minisweagent.run.extra.utils.exceptions import PatchApplyError

# Add src to path to import function_extractor
from .function_extractor import detect_crud_operations, extract_all_functions, filter_python_files


@dataclass
class MaintainabilityEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity executable."""
    files_to_change: list[str] | None = None
    """Path to the patch to apply during init of sandbox"""
    init_patch: Path | None = None

def get_image_name( p : Path ):
    iid = str(p).split("/")[-2]
    id_docker_compatible = iid.replace("__", "_1776_")
    return f"swebench_sweb.eval.x86_64.{id_docker_compatible}_latest.sif".lower()

class MaintainabilityEnvironment:
    def __init__(
        self, *, config_class: type = MaintainabilityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):

        """Maintainability environment. See `MaintainabilityEnvironmentConfig` for kwargs."""

        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"

        files_to_change = kwargs.get( 'files_to_change', None )
        init_patch = kwargs.get( 'init_patch', None )
        
        subprocess.run(
            [self.config.executable, "build", "--fakeroot", "--sandbox", self.sandbox_dir, self.config.image],
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Sometimes apptainer will not be able to bind necessary folders from host to container
        # This will prevent --writeable and will cause in error!
        for p in ["scratch", "projects", "share/apps", "state/partition1"]:
            os.makedirs(os.path.join(self.sandbox_dir, p), exist_ok=True)

        # optional: cache dirs inside container filesystem (only matters if you don't bind them)
        for p in ["tmp", "var/tmp", "root/.cache"]:
            os.makedirs(os.path.join(self.sandbox_dir, p), exist_ok=True)

        self.execute("git checkout -b maintainBench")

        # move corruption files
        # key --- local file path
        # value --- sandbox file path
        if files_to_change is not None:
            self.logger.info( ">>>>>>MOVING PR_0 FILES TO CONTAINER")
            for file_to_change in files_to_change:
                host_file = list( file_to_change.keys() )[0]
                target_path = list( file_to_change.values() )[0]
                tmp_bind = "/tmp/tmpfile.py"
                cmd = [self.config.executable,
                        "exec",
                        "--fakeroot",
                        "--writable",
                        f"--bind={host_file}:{tmp_bind}",
                        self.sandbox_dir,
                        "bash",
                        "-lc",
                        f"cp {tmp_bind} {target_path}"
                ]
                subprocess.run(cmd, check=True)

            for idx, i in enumerate( self.get_modified_file_paths() ):
                with open(i) as ofile:
                    data = ofile.read()

                with open("./ascii_html.py", "w") as ofile:
                    ofile.write(data)
                    

            self.execute("git add .")
            self.execute("git commit -m 'PR_0 files committed'")


        if init_patch is not None:
            self.logger.info( ">>>>>>APPLYING PATCH DURING INIT")
            self.apply_patch(init_patch)

            self.execute("git add .")
            self.execute("git commit -m 'PR_1 files committed'")

    def apply_patch(self, patch_file: Path):
        self.copy_to_container(patch_file, PurePosixPath("/testbed/patchxyz.diff"))

        GIT_APPLY_CMDS = [
            "git apply --verbose",
            "git apply --verbose --reject",
            "patch --batch --fuzz=5 -p1 -i",
        ]

        for git_apply_cmd in GIT_APPLY_CMDS:
            val = self.execute(
                f"{git_apply_cmd} /testbed/patchxyz.diff",
            )

            if val['returncode'] == 0:
                self.execute("rm -rf /testbed/patchxyz.diff")
                return
            self.logger.info(f"Failed to apply patch to container: {git_apply_cmd}, error: {val['output']}")

        raise PatchApplyError()

    def get_modified_file_paths(self) -> list[str]:
        val = self.execute("git ls-files --others --modified --exclude-standard")
        if val['returncode'] == 0:
            modified_files = [ str(self.sandbox_dir / f"testbed/{i}") for i in val['output'].splitlines() ]
        else:
            raise Exception("Issue in finding modified files")

        return modified_files

    def get_modified_functions_mapping(self, instance_id: str) -> dict:
        """
        Extract modified functions with CRUD tracking.
        
        Returns:
            {
                "file.py": [
                    {"name": "Class.method", "crud": "MODIFIED", "info": "SUCCESS"},
                    {"name": "func2", "crud": "ADDED", "info": "SUCCESS"}
                ],
                "new_file.py": [{"name": "FILE_ADDED", "crud": "ADDED", "info": "SUCCESS"}],
                "error_file.py": [{"info": "ERROR", "reason": "Parse error"}]
            }
        """
        result = {}
        repo = instance_id.split("__")[0]
        
        # Get modified files (relative paths)
        val = self.execute("git ls-files --others --modified --exclude-standard")
        if val['returncode'] != 0:
            self.logger.error("Failed to get modified files")
            return result
        
        modified_files = val['output'].splitlines()
        
        # Filter to only Python files, exclude tests, etc.
        filtered_files = filter_python_files(modified_files, repo)
        
        for filepath in filtered_files:
            try:
                # Check if file is newly added
                old_version_result = self.execute(f"git show HEAD:{filepath}")
                
                if old_version_result['returncode'] != 0:
                    # File is new (doesn't exist in HEAD)
                    result[filepath] = [{
                        "name": "FILE_ADDED",
                        "crud": "ADDED",
                        "info": "SUCCESS"
                    }]
                    continue
                
                # Get old version from HEAD
                old_content = old_version_result['output']
                
                # Get new version from working tree
                new_version_result = self.execute(f"cat {filepath}")
                if new_version_result['returncode'] != 0:
                    result[filepath] = [{
                        "info": "ERROR",
                        "reason": f"Failed to read file: {filepath}"
                    }]
                    continue
                
                new_content = new_version_result['output']
                
                # Parse both versions with AST
                try:
                    old_funcs = extract_all_functions(old_content)
                    new_funcs = extract_all_functions(new_content)
                except SyntaxError as e:
                    result[filepath] = [{
                        "info": "ERROR",
                        "reason": f"Syntax error: {str(e)}"
                    }]
                    continue
                
                # Detect CRUD operations
                crud_results = detect_crud_operations(old_funcs, new_funcs)
                
                # Only include files that have actual changes
                if crud_results:
                    result[filepath] = crud_results
                
            except Exception as e:
                result[filepath] = [{
                    "info": "ERROR",
                    "reason": f"Unexpected error: {str(e)}"
                }]
                self.logger.error(f"Error processing {filepath}: {e}")
        
        return result

    def get_maintainability_metrics(self, instance_id: str) -> dict:
        result = {}

        repo = instance_id.split("__")[0]

        # Get modified files (relative paths)
        val = self.execute("git ls-files --others --modified --exclude-standard")
        if val['returncode'] != 0:
            self.logger.error("Failed to get modified files")
            return result
        
        modified_files = val['output'].splitlines()

        print("Modified files", instance_id,  modified_files)
        
        # Filter to only Python files, exclude tests, etc.
        filtered_files = filter_python_files(modified_files, repo)

        print("Modified files (filtered)", instance_id,  filtered_files)

        self.logger.info(f"DEBUG: Processing {len(filtered_files)} filtered files")
        
        for idx, filepath in enumerate(filtered_files):
            self.logger.info(f"DEBUG: Processing file {idx+1}/{len(filtered_files)}: {filepath}")
            try:
                # Check if file is newly added
                self.logger.info(f"DEBUG: Checking if {filepath} exists in HEAD")
                old_version_result = self.execute(f"git show HEAD:{filepath}")
                
                if old_version_result['returncode'] != 0:
                    # File is new (doesn't exist in HEAD)
                    # TODO: only calculate metrics for new_files and set old_file metrics to None
                    self.logger.info(f"DEBUG: {filepath} is a NEW file (not in HEAD)")
                    old_content = None
                else:
                    # Get old version from HEAD
                    self.logger.info(f"DEBUG: {filepath} exists in HEAD, got old content")
                    old_content = old_version_result['output']
                
                # Get new version from working tree
                self.logger.info(f"DEBUG: Reading new content for {filepath}")
                new_version_result = self.execute(f"cat {filepath}")
                if new_version_result['returncode'] != 0:
                    self.logger.error(f"DEBUG: Failed to read {filepath}")
                    result[filepath] = [{
                        "info": "ERROR",
                        "reason": f"Failed to read file: {filepath}"
                    }]
                    continue
                
                new_content = new_version_result['output']
                self.logger.info(f"DEBUG: Got new content for {filepath}, length: {len(new_content)}")

                self.logger.info(f"DEBUG: Calculating metrics for {filepath}")

                metrics = calculate_metrics(new_content, old_content)
                self.logger.info(f"DEBUG: Metrics calculated successfully for {filepath}")

                result[filepath] = [{
                    "info": "SUCCESS",
                    "metrics": metrics
                }]

                self.logger.info(f"DEBUG: Added SUCCESS result for {filepath}")
                
            except Exception as e:
                self.logger.error(f"DEBUG: Exception caught for {filepath}: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
                result[filepath] = [{
                    "info": "ERROR",
                    "reason": f"Unexpected error: {str(e)}"
                }]
                self.logger.error(f"Error processing {filepath}: {e}")

        self.logger.info(f"DEBUG: Finished processing. Result has {len(result)} entries")

        if len(filtered_files) > 0 and len(result) == 0:
            self.logger.error(f"DEBUG: WARNING! Processed {len(filtered_files)} files but result is EMPTY!")
            self.logger.error(f"DEBUG: Filtered files were: {filtered_files}")

        return result
    
    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        # Remove INFO blocks caused by --fakeroot in env where suid mapping is missing (generally on all HPCs)
        # -q flag is quite execution which will supress all INFO blocks but all the warning & errors will propagate
        cmd = [self.config.executable,"-q", "exec", "--fakeroot"]

        # Do not inherit directories and env vars from host
        cmd.extend(["--contain", "--cleanenv"])

        work_dir = cwd or self.config.cwd # config.cwd = /testbed
        
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])

        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Load the conda file and activate the env
        # --cleanenv will flush out conda variables so need to call conda.sh to set them again
        prolog = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
        wrapped = f"{prolog} && {command}"
        # cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", wrapped])

        # For normal exec use the config.timeout but during eval use the timeout provided in the args
        timeout = self.config.timeout if timeout is None else timeout
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        return {"output": result.stdout, "returncode": result.returncode}

    def copy_to_container(self, src: Path , dst: Path | PurePosixPath):
        """
            Copy a file from local to a docker container

            Args:
                src (Path): Source file path
                dst (Path): Destination file path in the container
        """
        if not dst.parent or dst.parent == Path():
            raise ValueError(
                f"Destination path parent directory cannot be empty!, dst: {dst}"
            )

        tmp_bind = "/tmp/tmpfile.py"
        cmd = [self.config.executable,
                "-q",
                "exec",
                "--fakeroot",
                "--writable",
                f"--bind={src}:{tmp_bind}",
                self.sandbox_dir,
                "bash",
                "-lc",
                f"cp '{tmp_bind}' '{dst}'"
        ]
        subprocess.run(cmd, check=True)

    def exec_run_with_timeout(self, cmd, timeout: int | None = 60):
        """
            Run a command in a container with a timeout.

            Args:
                cmd (str): Command to run.
                timeout (int): Timeout in seconds.
        """
        start_time = time.time()
        timed_out = False
        output = ""
        returncode = 1

        try:
            result = self.execute( cmd, timeout=timeout)
            output = result["output"]
            returncode = result["returncode"]

        except subprocess.TimeoutExpired as e:
            timed_out = True
            output = e.stdout if e.stdout else ""
            returncode = -1  # distinguish timeout

        end_time = time.time()
        return output, timed_out, end_time - start_time, returncode

    def cleanup(self):
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
