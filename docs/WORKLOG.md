# PaleoTC worklog

How to resume in a new session: read this file first, then the plan
`docs/superpowers/plans/2026-10-15-tc-wind-stress-pipeline.md`. Continue from the
first unchecked task below. Each task ends with a git commit, so `git log --oneline`
on branch `feature/tc-stress-pipeline` shows what is done.

## Decisions (2026-10-15)

- **Coupler semantics:** the TC stress **replaces** the coupler stress inside the boxes (`TCMASK`), it is not added to it. The Li et al. 2023 SourceMod added it; Task 12 ports the code with replace semantics.
- **`export.lr_year_offset`:** 0 for now (LR-TC model year = HR year). Revisit before the LR-TC run.
- **Task 12 Step 6** (LR-TC build and 5-day test run): Hui runs this, not Claude.
- **Execution mode:** native. Claude implements the tasks in one session, then one reviewer checks the whole branch at the end.
- **No further checks of the U10-shortcut error.** Already measured: within 0.9% for 10–70 m/s.
- Other design decisions (h2 stream, gx1eocene weights, air density, cdn) are in the plan's "Design decisions" table.

## Environment

- Conda env: `/glade/work/huili7/miniconda3/envs/paleotc` (created in Task 1).
  Activate with `module load conda; conda activate /glade/work/huili7/miniconda3/envs/paleotc`
- Git branch: `feature/tc-stress-pipeline`
- Outputs: `/glade/derecho/scratch/huili7/PaleoTC`

## Task status

- [x] Task 1: Branch, environment, scaffold, config
- [x] Task 2: Track parsing
- [x] Task 3: Regridder
- [x] Task 4: TC box mask
- [x] Task 5: Flux formulas
- [x] Task 6: Extraction stage
- [ ] Task 7: Stress stage
- [ ] Task 8: POP export stage
- [ ] Task 9: Pipeline, Casper cluster, CLI
- [ ] Task 10: Real-data run (year 31) + QC, then all 30 years
- [ ] Task 11: get_tau review notes + README
- [ ] Task 12: Port the POP SourceMods (Steps 1–5 Claude; Step 6 Hui)
- [ ] Final whole-branch review

## Log

- 2026-10-15: Plan written and approved, worklog created. Implementation starting with Task 1.
