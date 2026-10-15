"""Run the stages for one HR year (one dask task)."""
from __future__ import annotations

import logging

from .config import load_config
from .export import export_year
from .extract import extract_year
from .stress import stress_year

STAGES = ("extract", "stress", "export")
_RUNNERS = {"extract": extract_year, "stress": stress_year, "export": export_year}


def run_year(config_path: str, year: int, stages: tuple[str, ...] = STAGES) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(config_path)
    result = {"year": year}
    for stage in STAGES:
        if stage in stages:
            result[stage] = _RUNNERS[stage](cfg, year)
    return result
