"""Environment implementations for mini-SWE-agent."""

import copy
import importlib

from minisweagent import Environment

_ENVIRONMENT_MAPPING = {
    "docker": "minisweagent.environments.docker.DockerEnvironment",
    "singularity": "minisweagent.environments.singularity.SingularityEnvironment",
    "local": "minisweagent.environments.local.LocalEnvironment",
    "swerex_docker": "minisweagent.environments.extra.swerex_docker.SwerexDockerEnvironment",
    "maintainability": "minisweagent.environments.maintainability.MaintainabilityEnvironment",
    "firstPRmaintainability": "minisweagent.environments.firstPRmaintainability.FirstPRMaintainabilityEnvironment",
    "maintainability_fast": "minisweagent.environments.maintainability_fast.MaintainabilityFastEnvironment",
    "patchGeneration": "minisweagent.environments.patchGeneration.PatchGenerationEnvironment",
    "singularityMulti": "minisweagent.environments.singularity_swebench_multilingual.SingularityMultiEnvironment",
    "foresightSingularity": "minisweagent.environments.foresight_singularity.ForesightSingularityEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    full_path = _ENVIRONMENT_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown environment type: {spec} (resolved to {full_path}, available: {_ENVIRONMENT_MAPPING})"
        raise ValueError(msg)


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)
