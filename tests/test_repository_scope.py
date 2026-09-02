import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_scope_gate():
    script = ROOT / "scripts/check_repository_scope.py"
    spec = importlib.util.spec_from_file_location("scope_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.validate() == []
