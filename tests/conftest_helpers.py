"""Load a module from the repository's tools/ directory by path.

`from tools import x` resolves to whichever `tools` package is installed first,
and this environment has one. Importing by file path is the only way to be sure
a test exercises this repository's copy.
"""

import importlib.util
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_repo_tool(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"driftguard_repo_tools.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
