# GapTrust - when can a trace gap be trusted?

Code and numerical records for a study of **incomplete common-offset GPR profiles**:
if a trace gap is filled by interpolation or by f-k POCS, at what gap width does the
reconstructed section stop recovering the target?

The tested rule is the gap number

    G = W / D ,   D = sqrt(2 * lambda * d + lambda^2 / 4)

where W is the gap width in metres, D the first Fresnel-zone surface aperture of a
diffractor at depth d, and lambda the dominant wavelength.

**Result as reported.** The hypothesis that G collapses the scatter of the failure
width more tightly than W does is  on the full simulation set
(all six coefficient-of-variation ratios are above 0.50). The study is written as
a null result with a bounded supporting test: G reduces the scatter without
collapsing it, and the off-target control shows that failure is caused by loss of
aperture at the target rather than by loss of traces in general.

This repository is the computational record. It is not a tutorial and it does not
reproduce the paper's prose; every number in the manuscript traces to a logged run
under `logs/`.

## Layout

```
gaptrust/              reusable modules (no side effects on import)
    physics.py         wavelength, Fresnel aperture, gap number, Ricker, spreading
    processing.py      time zero, mean removal, Kirchhoff migration, envelope, fills
    models.py          model and void naming, the 26-model design
    style.py           publication style: 174 mm, Okabe-Ito, PDF + 600 dpi LZW TIFF

scripts/
    sim/               run_simulations.py - gprMax 4.0.1 driver (resumable)
    analysis/          pilot, full-set, r/lambda, profile-length, diagnostics
    field/             karst and Aquileia readers, gap sweeps, off-target control
    figures/           one script per figure, Fig. 1-7 and S1-S6
    manuscript/        python-docx assembly of the two Word deliverables

data/                  CSV tables that the figures read (failure points, R tables)
logs/                  one JSON log per run: parameters, versions, inputs, outputs
manifest.csv           release path, SHA-256, and original path for every file
```

`data/` and `logs/` are copied from `Codes/results/`. Nothing in them was edited;
their SHA-256 values in `manifest.csv` refer to the untouched originals.

## Scientific order of the work

1. `scripts/sim/run_simulations.py` builds and runs the 26 B-scans (13 models x 2
   frequencies, seed 20261106) and writes one `.npz` per model. Resumable: models
   with an existing `.npz` are skipped.
2. `scripts/analysis/pilot_analysis.py` evaluates the 6-scan pilot that failed its
   own pre-registered rule.
3. `scripts/analysis/full_analysis_recovery.py` runs the 3 fills x 41 widths sweep
   over every model and writes the recovery tables (the source of Figs. 3-5).
4. `scripts/analysis/rlam_analysis.py` runs the twelve r/lambda models: H3a met,
   H3b not met, outcome "partly supported".
5. `scripts/analysis/test_profile_length.py` and the two `pilot_diagnostic_*.py`
   scripts hold the profile-length test and the pilot diagnostics.
6. `scripts/field/field_analysis.py --edge-guard` processes the three Guilin karst
   lines; `field_analysis_aquileia.py` the three Aquileia profiles on the full
   depth grid; `field_control_offtarget.py` the displaced-gap control of Fig. 7;
   `field_control_summarise.py` reads the on-target ratio from the frozen tables.
7. `scripts/figures/figNN_*.py` render the figures, one script per figure, each
   writing its own JSON log.
8. `scripts/manuscript/build_*.py` assemble the Word files from the ledger numbers.

## Reproducing

Full instructions, including the two conda environments and the gprMax call, are in
`REPRODUCE.md`. Short version:

```
conda env create -f environment.yml
python -m pip install -r requirements.txt
python scripts/sim/run_simulations.py          # hours; skips existing .npz
python scripts/analysis/full_analysis_recovery.py
python scripts/figures/fig01_concept.py        # and the other figNN scripts
```

Everything runs from the repository root or from any directory: each script locates
`gaptrust` from its own file path.

## Verification

Every script in this release was syntax-checked against Python 3.10. Each file's
SHA-256 in `manifest.csv` is the checksum of the file it was copied from, so the
release can be compared with the working tree. The only edits made to the copies
are the import depth (`parents[1]` -> `parents[2]`, because scripts sit one folder
deeper) and one constant `PROJ` that replaces the hard-coded project path in the
scripts that used it. No scientific parameter, threshold or equation was changed.

## Data

- Simulation outputs (`.npz`, several GB) and the raw field archives are **not**
  included. `DATA.md` lists what is needed and where each file is expected.
- Raw field data are third-party property. The Guilin data belong to the author's
  field campaign; the Aquileia SEG-Y belongs to the 2013 study cited in the paper.
  Neither can be redistributed here without permission.

## Status and limits

- POCS on field data drives the recovery ratio above unity (up to 11.35); the paper
  reports it as unusable for this measurement rather than as a success.
- The Guilin line length is an adopted approximation (8.0 m, bracket 7-10 m) and the
  text says so; G is counted in traces because trace spacing was never recorded.
- The classification thresholds G_c = 0.2876 and G_u = 1.0311 are not validated and
  are not applied to field data.
- Fig. S6 is exploratory and lies outside the decision rule.

## Citation and licence

it will be added once manuscript will be published 
# Analysis environment (Python 3.10)
numpy==2.2.6
scipy==1.15.3
matplotlib==3.10.9
Pillow==12.2.0
h5py==3.16.0
python-docx==1.2.0

# simulation, installed separately from the analysis environment:
# gprMax==4.0.1  (gprMax4 env, Python 3.13)
# readgssi==0.0.22  (readgssi env, Python 3.13, needs setuptools<80)
# Reproducing the GapTrust results

## Environments

Four interpreters were used in the original work and are kept separate:

| purpose | interpreter |
|---|---|
| simulation | `anaconda3/envs/gprmax4/Scripts/python.exe`, Python 3.13, gprMax 4.0.1 |
| analysis, figures, Word build | `anaconda3/envs/gprmax/python.exe`, Python 3.10, numpy/scipy/matplotlib/Pillow/python-docx |
| reading GSSI .DZT only | `readgssi_env/Scripts/python.exe`, readgssi 0.0.22, setuptools<80 |
| figure TIFF writing | the gprmax env; the base 3.13 interpreter crashes on LZW TIFF |

`environment.yml` is the analysis environment as recorded in the project; it does not
pin gprMax, h5py or Cython, which are needed only for simulation. Install gprMax
separately (conda or pip), 4.0.1.

## Order of execution

1. Simulation, about 9 min per 250 MHz and 26 min per 400 MHz B-scan on 2 cores:

```
python scripts/sim/run_simulations.py
```

Writes `results/sims/250MHz/*.npz` and `results/sims/400MHz/*.npz`. Resumable.

2. Full recovery sweep (3 fills x 41 gap widths x 26 models):

```
python scripts/analysis/full_analysis_recovery.py
```

Writes `results/data/full_R_recovered.csv`, `failure_points_recovered.csv` and
`results/logs/full_summary_recovered.json`. Takes about 2.5 h on one core; it caches
per case in `results/recovery/cache` and resumes.

3. r/lambda test:

```
python scripts/analysis/rlam_analysis.py
```

4. Field analysis. The raw archives are not in the repository; place them as
   siblings of this folder, named `Karst data/` and `Aquileia Data/`, then:

```
python scripts/field/field_inventory.py
python scripts/field/read_karst_cache.py      # builds the .npz cache, --force to rebuild
python scripts/field/field_analysis.py --edge-guard
python scripts/field/field_analysis_aquileia.py
python scripts/field/field_control_offtarget.py
python scripts/field/field_control_summarise.py
```

`--edge-guard` applies the amendment of ledger section 14r (target at least 10
resampled traces from either end, and the depth box inside the 10-60 ns window).
Running the script without the flag reproduces the first, artefact-affected result.

5. Figures. Each script writes its own JSON log to `results/logs/` and both a PDF
   and a 600 dpi LZW TIFF to `../Figures/main` or `../Figures/supplementary`
   (`style.FIG_ROOT` is the folder beside the repository):

```
python scripts/figures/fig01_concept.py
python scripts/figures/fig02_models.py
...
python scripts/figures/figS6_rlam_scatter.py
```

6. Manuscript. The Word files are assembled in order; each script appends to or
   patches the previous one and expects `python-docx`:

```
python scripts/manuscript/build_main.py
python scripts/manuscript/build2.py
python scripts/manuscript/build3.py
python scripts/manuscript/build4.py
python scripts/manuscript/build5.py
python scripts/manuscript/build_supp.py
```

## Differences from the working copies

- `sys.path` depth: scripts sit one folder deeper here, so `parents[1]` became
  `parents[2]`.
- A constant `PROJ` (the project root, three levels above the script) replaces the
  hard-coded absolute path in `field_inventory.py`, `mslib.py` and the build scripts.
- `scripts/manuscript/` keeps only the assembly steps. The two one-off repair
  scripts (`patch_eq2.py`, `patch_tabcap.py`) that fixed equation placement and
  table caption numbers in a single draft are not included; they are recorded in
  `Paper_Ledger.md` section 14ab.

No parameter, threshold, equation or processing step was altered.

## Determinism

The simulation seed is 20261106 and the no-void twin is generated by the same
model call, so the pairs differ only by the presence of the void. Analyses are
deterministic given the `.npz` inputs; the cache is keyed by frequency and void.
name: gaptrust
channels:
  - conda-forge
dependencies:
  - python=3.13
  - numpy=2.5.3
  - scipy=1.18.1
  - matplotlib=3.11.2
  - pillow
  - h5py
  - cython
  - pip
  - pip:
      # gprMax is distributed on PyPI only. Version 4.0.1 requires Python >= 3.11.
      # On the current machine simulations run in a separate venv; see README.md
      # and ledger sections 14g and 14h.
      - gprMax==4.0.1
# Data needed to rerun everything

## Included

- `data/*.csv`: recovery ratios per model, method and gap width; failure points;
  the field tables for the karst lines, the Aquileia profiles and the off-target
  control.
- `logs/*.json`: one log per run (parameters, package versions, UTC time, outputs).
- `manifest.csv`: SHA-256 of every file at the time it was copied.

## Not included

| item | size | where it must be placed |
|---|---|---|
| `results/sims/250MHz/*.npz`, `results/sims/400MHz/*.npz` | several GB | regenerate with `run_simulations.py`, or requested from the author |
| `Karst data/` (GSSI SIR-4000 .DZT, three lines) | archive | beside the repository root |
| `Aquileia Data/` (MALA 250 MHz SEG-Y) | archive | beside the repository root |
| `results/data/field_cache/*.npz` | about 400 MB | rebuilt by the field scripts on first run |

`field_inventory.py` and `read_karst_cache.py` read `../Karst data`; `field_analysis_aquileia.py`,
`field_control_offtarget.py` and Figs 6-7 read `../Aquileia Data`. The karst analysis and
Figs 6-7 read the cache in `results/data/field_cache`, which `read_karst_cache.py`
builds from the raw .DZT files.
That is why the raw archives go beside the repository, not inside it.

## Redistribution

The Aquileia SEG-Y files belong to the 2013 acquisition described in the site
literature and are not ours to publish. The Guilin profiles were recorded by the
author. Permission for either must come from the author before this repository is
made public with data attached.


