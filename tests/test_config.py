import pytest

from paleotc.config import load_config


def test_load_config_types_and_paths(fake_case):
    cfg = load_config(fake_case)
    assert cfg.years == [31]
    assert cfg.export.hours == (0, 6, 12, 18)
    assert cfg.export.lr_year_offset == 70
    assert cfg.cluster is None
    assert cfg.boxed_path(31).name == "tc_boxed.0031.nc"
    assert cfg.stress_path(31).name == "tc_stress.0031.nc"
    assert cfg.export_dir.name == "pop_forcing"


def test_hr_file_found(fake_case):
    cfg = load_config(fake_case)
    assert cfg.hr_file("UBOT", 31).name == "fake.case.cam.h2.UBOT.0031010100-0032010100.nc"


def test_hr_file_missing_raises(fake_case):
    cfg = load_config(fake_case)
    with pytest.raises(FileNotFoundError, match="UBOT"):
        cfg.hr_file("UBOT", 45)
