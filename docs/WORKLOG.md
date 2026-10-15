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
- [x] Task 3: `get_tau.py` (notebook replacement) + tests
- [x] Task 4: `run_casper.py` (dask-jobqueue on Casper)
- [x] Task 5: Real run for year 31 + `qc_tau.py`, then all 30 years
- [~] Task 6: get_tau review notes + README (skipped by Hui, 2026-10-15)
- [x] Task 7: Port the POP SourceMods (Steps 1–5 done; Step 6, build + 5-day test, is Hui's)
- [ ] Final whole-branch review

## Log

- 2026-10-15: v1 plan (a package of 9 modules) implemented through Task 9 and committed (`55460e4..57eefaa`, 31 tests passing).
- 2026-10-15: Hui found it too complicated. The plan was rewritten as simple scripts (this version); implementation has not started yet.
- 2026-10-15: v2 Task 1 done: package removed, plain config.yaml added.
- 2026-10-15: v2 Task 2 done: extract_tcbot.py + test_tc.py (4 tests). Awaiting Hui's code review; not yet run on real HR data.
- 2026-10-15: v2 Task 3 done: get_tau.py (+3 tests, 7 total). Tested extract+get_tau on year 31 Jan 1-3 (12 steps): max |tau| 1.4-2.7 N/m2, no NaN, zero outside boxes. Casper account P93300014 is invalid -> must change casper.account in config.yaml before Task 4/5.
- 2026-10-15: v2 Task 4 done: run_casper.py written; job script checked (not yet submitted). casper.account changed to P93300313 by Hui.
- 2026-10-15: v2 Task 5 done: years 31-60 run on Casper in ~6 min (30 workers, no failures). 38851 tcbot + 38851 tau files (= all track times except the year-61 point), tc_tau 54 GB on scratch. QC year 31: max |tau| 11.5 N/m2 (U10 51.6 vs native 53.6 m/s), no NaN. Pending: Hui's decision on organizing outputs by year (tc_bot yes; tc_tau must stay flat unless SourceMod changes); copy tc_tau off scratch.
- 2026-10-15: tc_bot reorganized into tc_bot/YYYY/ (code + existing 38851 files moved); tc_tau kept flat (POP). QC year 55: 1256 files, max |tau| 14.0 N/m2 (U10 55.9 vs native 58.3 m/s), no NaN; get_tau rerun from new layout reproduces all 1256 files exactly.
- 2026-10-15: Checked the map file against jiangzhu's `map_ne120np4_TO_gx1v6_eocene_{blin,patc,aave}.231017.nc`. Grids are identical and the weights match to 1e-8 (only the ESMF version differs), so it is the same map. Blin vs patch on the strongest year-55 snapshot: max U10 54.3 vs 54.0 m/s, integrated |tau| within 0.3%. Decision: keep `gx1eocene_blin.260410` (matches the LR coupler's ATM2OCN_VMAPNAME). No rerun.
- 2026-10-15: Task 6 skipped. Checked Li's SourceMods against iCESM1.3.1. Li's forcing_ws.F90 = tag + TC blocks (compiles, but it adds the stress, has no TCMASK, and hard-codes the old path). Li's forcing_coupled.F90 must not be used: it is from POP rev 26603 and would drop Jiang Zhu's ROFF_L estuary mods in the LR case. Plan Task 7 fixed: rotate_wind_stress WORK1/WORK2 intent(in)->intent(inout) (the build would fail otherwise). Open item added: with OCN_NCPL=1 only the 00Z snapshot is read and held for the whole day.
- 2026-10-15: Task 7 Steps 1-5 done. sourcemods/orig_iCESM1.3.1/ (tag originals, read-only), sourcemods/orig_LRcase/forcing_coupled.F90 (LR-case base, read-only), sourcemods/src.pop/ (modified). Both modified files compile with ifort 2021.8.0 (oneAPI 2023.0.0 classic, same as the LR build) using the LR case's ocn .mod files and flags; restoring intent(in) gives error #6780, so the intent fix is required. The hard-coded tc_filename matches config.yaml (9730 00Z files exist). Next: Hui's Step 6 (clone LR-cntl -> LR-TC, build, 5-day test), then the final review.
