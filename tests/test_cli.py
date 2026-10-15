import json

from paleotc.cli import main
from paleotc.config import load_config


def test_cli_runs_all_stages_locally(fake_case):
    assert main([str(fake_case), "--local", "1"]) == 0
    cfg = load_config(fake_case)
    assert len(list(cfg.export_dir.glob("*.nc"))) == 2
    (log,) = (cfg.output_root / "logs").glob("run_*.json")
    report = json.loads(log.read_text())
    assert report["failures"] == [] and report["results"][0]["year"] == 31


def test_cli_reports_failure(fake_case):
    # year 45 has no HR files -> FileNotFoundError captured, exit code 1
    assert main([str(fake_case), "--local", "1", "--years", "45", "45"]) == 1
