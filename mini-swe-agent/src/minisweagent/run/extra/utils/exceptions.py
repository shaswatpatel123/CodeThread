class PatchApplyError(Exception):
    def __init__(self):
        super().__init__()

class EvaluationTimeout(Exception):
    def __init__(self):
        super().__init__()

class PatchFileNotFound(Exception):
    def __init__(self):
        super().__init__()

class EvaluationError(Exception):
    def __init__(self):
        super().__init__()

class NoFailToPassError(Exception):
    def __init__(self):
        super().__init__()

class NoTestPatchError(Exception):
    def __init__(self):
        super().__init__()

class RepoSettingsError(Exception):
    def __init__(self):
        super().__init__()

class TestRunError(Exception):
    def __init__(self):
        super().__init__()
