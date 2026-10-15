"""paleotc-run: run pipeline stages for a range of HR years, one dask task per year."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from distributed import as_completed

from .cluster import make_client
from .config import load_config
from .pipeline import STAGES, run_year


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    p.add_argument("--local", type=int, metavar="N", help="use a LocalCluster with N workers")
    a = p.parse_args(argv)

    config_path = str(Path(a.config).resolve())
    cfg = load_config(config_path)
    years = list(range(a.years[0], a.years[1] + 1)) if a.years else cfg.years

    client = make_client(cfg, a.local, n_jobs=len(years))
    try:
        futures = {
            client.submit(run_year, config_path, y, tuple(a.stages), key=f"year-{y}", retries=1): y
            for y in years
        }
        results, failures = [], []
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                results.append(fut.result())
                print(f"year {year}: done", flush=True)
            except Exception as exc:  # report every failed year, keep going
                failures.append({"year": year, "error": repr(exc)})
                print(f"year {year}: FAILED {exc!r}", flush=True)
    finally:
        client.close()

    log_dir = cfg.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stages": a.stages,
        "results": sorted(results, key=lambda r: r["year"]),
        "failures": sorted(failures, key=lambda r: r["year"]),
    }
    (log_dir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
