"""Reject misleading model provenance before loading upstream Python objects."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("export_rgbt_model", Path(__file__).parents[1] / "scripts/export_rgbt_model.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.mark.parametrize("commit,dirty,error", [
    (export.SOURCE_COMMIT, "", None),
    ("wrong-revision", "", "pinned commit"),
    (export.SOURCE_COMMIT, " M models/yolo.py", "local modifications"),
])
def test_source_identity_checked_before_export(monkeypatch, tmp_path, commit, dirty, error):
    replies = iter([commit, dirty])
    monkeypatch.setattr(export.subprocess, "check_output", lambda *a, **kw: next(replies))
    if error:
        with pytest.raises(ValueError, match=error):
            export.validate_repository(tmp_path)
    else:
        assert export.validate_repository(tmp_path) == export.SOURCE_COMMIT
