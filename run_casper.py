"""Run extract_tcbot.py and get_tau.py for a range of HR years on Casper.

Replaces scripts_Li2023/sub-batch.sh and the dask loop in get_tau_dask.ipynb.
dask-jobqueue starts one PBS job (one dask worker) per year; each worker runs
extract_year(), then get_tau_00z.py (6-hourly snapshots) and get_tau.py (daily mean) for its year. Years run in parallel and
results are printed as each year finishes.

Usage (from a Casper login node, a qinteractive session, or JupyterHub):
    python run_casper.py FIRST LAST [config.yaml]
    e.g. python run_casper.py 31 60
"""
import sys
from pathlib import Path

import yaml
from dask.distributed import Client, as_completed
from dask_jobqueue import PBSCluster

from extract_tcbot import extract_year
from get_tau import get_tau_year
from get_tau_00z import get_tau_year as get_tau_00z_year

HERE = Path(__file__).resolve().parent


def run_year(year, cfg):
    """All steps for one year. Returns (TC time steps, 6-hourly tau files, daily tau files)."""
    return extract_year(year, cfg), get_tau_00z_year(year, cfg), get_tau_year(year, cfg)


if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    config_file = sys.argv[3] if len(sys.argv) > 3 else HERE / "config.yaml"
    cfg = yaml.safe_load(open(config_file))
    c = cfg["casper"]
    years = list(range(first, last + 1))

    # one PBS job = one worker with 1 core and enough memory for one year
    cluster = PBSCluster(
        cores=1,
        processes=1,
        memory=c["memory"],
        account=c["account"],
        queue="casper",
        walltime=c["walltime"],
        resource_spec=f"select=1:ncpus=1:mem={c['memory']}",
        interface="ext",
        log_directory=c["log_dir"],
    )
    cluster.scale(jobs=min(len(years), c["max_jobs"]))
    client = Client(cluster)
    print(f"dask dashboard: {client.dashboard_link}")

    # workers start in another directory: send them our scripts so they can import them
    client.upload_file(str(HERE / "extract_tcbot.py"))
    client.upload_file(str(HERE / "get_tau_00z.py"))
    client.upload_file(str(HERE / "get_tau.py"))

    futures = {client.submit(run_year, year, cfg, key=f"year-{year}"): year for year in years}
    failed = []
    for future in as_completed(futures):
        year = futures[future]
        try:
            n_bot, n_00z, n_daily = future.result()
            print(f"year {year}: {n_bot} TC time steps, {n_00z} 6-hourly + {n_daily} daily tau files")
        except Exception as err:
            failed.append(year)
            print(f"year {year}: FAILED {err!r}")

    client.close()
    cluster.close()
    print(f"done; failed years: {failed or 'none'}")
