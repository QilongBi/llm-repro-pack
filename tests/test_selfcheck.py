from __future__ import annotations

from conftest import confirm_all, do_collect, do_discover, do_package, do_select
from fixtures import COMMIT, OLMO, make_hf_cache, make_selfrun, selfrun_selection
import hash_files
import secret_scan
import selfcheck

FAKE_GROQ = "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def _pkg(tmp_path, extra=()):
    p = make_selfrun(tmp_path)
    hf = make_hf_cache(tmp_path / "hf", OLMO, [COMMIT])
    do_discover(p)
    do_select(p, selfrun_selection(p) + list(extra))
    do_collect(p, OLMO, hf)
    confirm_all(p)
    return p


def test_secret_in_log_fails_selfcheck(tmp_path):
    p = _pkg(tmp_path)
    log = p / "logs" / "run.log"
    log.write_text(log.read_text(encoding="utf-8") + f"api_key={FAKE_GROQ}\n", encoding="utf-8")
    res = do_package(p, tmp_path / "pkg")
    assert not res["selfcheck"]["ok"]
    assert any("secret scan" in e and "groq_key" in e for e in res["selfcheck"]["errors"])


def test_env_var_value_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_API_KEY", "zz-not-a-known-format-0123456789")
    p = _pkg(tmp_path)
    log = p / "logs" / "run.log"
    log.write_text(log.read_text(encoding="utf-8") + "token zz-not-a-known-format-0123456789\n", encoding="utf-8")
    res = do_package(p, tmp_path / "pkg")
    assert any("value_of_env_MY_PROVIDER_API_KEY" in e for e in res["selfcheck"]["errors"])


def test_token_in_environment_never_written(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN_FOR_TEST", FAKE_GROQ)
    res = do_package(_pkg(tmp_path), tmp_path / "pkg")
    assert res["selfcheck"]["ok"], res["selfcheck"]["errors"]
    for f in res["package"].rglob("*"):
        if f.is_file():
            assert FAKE_GROQ.encode() not in f.read_bytes()


def test_tampering_breaks_checksums(tmp_path):
    p = _pkg(tmp_path)
    res = do_package(p, tmp_path / "pkg")
    assert res["selfcheck"]["ok"], res["selfcheck"]["errors"]
    pkg = res["package"]
    (pkg / "outputs" / "labels.csv").write_text("id,label\nt00,negative\n", encoding="utf-8")
    v = hash_files.verify(pkg)
    assert not v["ok"] and "outputs/labels.csv" in v["mismatched"]
    assert not selfcheck.selfcheck(pkg)["ok"]


def test_secret_scan_patterns():
    hits = secret_scan.scan_text(f"key = '{FAKE_GROQ}'\nAuthorization: Bearer abcdefghijklmnop1234\n", [])
    kinds = {k for _, k in hits}
    assert "groq_key" in kinds and "bearer_header" in kinds
    assert secret_scan.scan_text("api_key_env: GROQ_API_KEY\n", []) == []
