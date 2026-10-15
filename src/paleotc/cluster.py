"""Dask cluster: PBS jobs on Casper, or a LocalCluster for tests and debugging."""
from __future__ import annotations

from distributed import Client, LocalCluster

from .config import Config


def make_client(cfg: Config, local_workers: int | None = None, n_jobs: int | None = None) -> Client:
    if local_workers:
        return Client(LocalCluster(n_workers=local_workers, threads_per_worker=1))
    from dask_jobqueue import PBSCluster

    c = cfg.cluster
    if c is None:
        raise ValueError("config has no 'cluster' section; use --local N")
    cluster = PBSCluster(
        cores=c.cores,
        processes=1,
        memory=c.memory,
        account=c.account,
        queue=c.queue,
        walltime=c.walltime,
        resource_spec=f"select=1:ncpus={c.cores}:mem={c.memory}",
        interface=c.interface,
        local_directory=c.local_directory,
        log_directory=c.log_directory,
    )
    cluster.scale(jobs=min(c.n_workers, n_jobs or c.n_workers))
    return Client(cluster)
