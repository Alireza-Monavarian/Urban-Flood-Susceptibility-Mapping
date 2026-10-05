# tests/test_config.py
import pytest
from pipeline import config

def test_seed_is_42():
    assert config.GLOBAL_SEED == 42

def test_java_resolves_from_env(monkeypatch, tmp_path):
    fake = tmp_path / "java"; fake.write_text(""); fake.chmod(0o755)
    monkeypatch.setenv("FLOOD_JAVA", str(fake))
    assert config.java_exe() == str(fake)

def test_provenance_has_keys():
    p = config.provenance()
    assert {"git_sha", "config_hash", "python", "created"} <= set(p)

def test_java_raises_on_missing_flood_java(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOOD_JAVA", str(tmp_path / "does_not_exist_java"))
    with pytest.raises(FileNotFoundError):
        config.java_exe()
