# PaleoTC worklog

How to resume in a new session: read this file first, then the plan
`docs/superpowers/plans/2026-10-15-tc-wind-stress-pipeline.md` (the simplified version).
Continue from the first unchecked task below. Each task ends with a git commit on branch
`feature/tc-stress-pipeline`, so `git log --oneline` shows what is done.

## Decisions (2026-10-15)

- **Coupler semantics:** the TC stress **replaces** the coupler stress where `TCMASK = 1`, rather than being added (the Li et al. 2023 SourceMod added it).
- **`lr_year_offset`:** 0 for now (LR-TC model year = HR year). Revisit before the LR-TC run.
- **SourceMod build and 5-day test** (Task 7 Step 6): Hui runs this.
- **Execution mode:** native. Claude implements the tasks, then one reviewer checks the whole branch at the end.
- **No further checks of the U10-shortcut error.** Already measured: within 0.9% for 10–70 m/s.
- **Simplified approach (replaces v1):** plain Python scripts that translate the old NCL and notebook, with no package. Intermediate output is one file per TC time step (option 1). The v1 package (commits `55460e4..57eefaa`) is removed in Task 1, but stays in git history.
- The science and grid decisions (h2 stream, gx1eocene weights, air density, cdn, longitude-wrap fix) are in the plan's "Design decisions" table.

## Environment

- Conda env: `/glade/work/huili7/miniconda3/envs/paleotc` (already created; no `pip install` needed)
  Activate with `module load conda; conda activate /glade/work/huili7/miniconda3/envs/paleotc`
- Git branch: `feature/tc-stress-pipeline`
- Outputs: `/glade/derecho/scratch/huili7/PaleoTC/{tc_bot,tc_tau}`

## Task status (simplified plan)

- [x] Task 1: Remove the package; add a plain `config.yaml`
- [x] Task 2: `extract_tcbot.py` (NCL translation) + tests
- [ ] Task 3: `get_tau.py` (notebook replacement) + tests
- [ ] Task 4: `run_casper.py` (dask-jobqueue on Casper)
- [ ] Task 5: Real run for year 31 + `qc_tau.py`, then all 30 years
- [ ] Task 6: get_tau review notes + README
- [ ] Task 7: Port the POP SourceMods (Steps 1–5 Claude; Step 6 Hui)
- [ ] Final whole-branch review

## Log

- 2026-10-15: v1 plan (a package of 9 modules) implemented through Task 9 and committed (`55460e4..57eefaa`, 31 tests passing).
- 2026-10-15: Hui found it too complicated. The plan was rewritten as simple scripts (this version); implementation has not started yet.
- 2026-10-15: v2 Task 1 done: package removed, plain config.yaml added.
- 2026-10-15: v2 Task 2 done: extract_tcbot.py + test_tc.py (4 tests). Awaiting Hui's code review; not yet run on real HR data.
