"""Experiment configuration: YAML file -> frozen dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExportConfig:
    prefix: str
    lr_year_offset: int  # LR-TC model year = HR year + offset
    hours: tuple[int, ...] = (0, 6, 12, 18)


@dataclass(frozen=True)
class ClusterConfig:
    account: str
    queue: str = "casper"
    cores: int = 1
    memory: str = "16GB"
    walltime: str = "02:00:00"
    n_workers: int = 30
    interface: str | None = "ext"
    local_directory: str = "/tmp"
    log_directory: str = "dask-logs"


@dataclass(frozen=True)
class Config:
    case: str
    hr_dir: Path
    track_file: Path
    map_file: Path
    ocn_domain_file: Path
    output_root: Path
    first_year: int
    last_year: int
    export: ExportConfig
    cluster: ClusterConfig | None = None
    hr_stream: str = "h2"
    box_half_width: float = 10.0
    time_chunk: int = 120

    @property
    def years(self) -> list[int]:
        return list(range(self.first_year, self.last_year + 1))

    def hr_file(self, var: str, year: int) -> Path:
        pattern = f"{self.case}.cam.{self.hr_stream}.{var}.{year:04d}01*.nc"
        matches = sorted(self.hr_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected 1 file matching {pattern} in {self.hr_dir}, found {len(matches)}"
            )
        return matches[0]

    def boxed_path(self, year: int) -> Path:
        return self.output_root / "boxed" / f"tc_boxed.{year:04d}.nc"

    def stress_path(self, year: int) -> Path:
        return self.output_root / "stress" / f"tc_stress.{year:04d}.nc"

    @property
    def export_dir(self) -> Path:
        return self.output_root / "pop_forcing"


_PATH_KEYS = ("hr_dir", "track_file", "map_file", "ocn_domain_file", "output_root")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    export = raw.pop("export")
    export["hours"] = tuple(export.get("hours", (0, 6, 12, 18)))
    cluster = raw.pop("cluster", None)
    for key in _PATH_KEYS:
        raw[key] = Path(raw[key])
    return Config(
        export=ExportConfig(**export),
        cluster=ClusterConfig(**cluster) if cluster else None,
        **raw,
    )
