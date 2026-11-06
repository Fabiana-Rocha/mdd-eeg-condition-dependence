# EEG alterations in major depressive disorder: condition dependence and technical confounds

Code for the analyses in *EEG Alterations in Major Depressive Disorder: Formal Multi-Domain
Testing of Condition Dependence and Control of Technical Confounds*

The code is released by stage.

| Stage | Notebook | Status |
|---|---|---|
| 0 | `notebooks/00_duplicate_audit.ipynb` — duplicate records, acquisition parameters, recording dates | Available |
| 1 | `notebooks/01_preprocessing.ipynb` — filtering, acquisition-artifact removal, ICA, windowing | Available |
| 2 | `notebooks/02_features_group_comparison.ipynb` — 1,216 features, HC vs MDD in each condition (Layer 1) | Available |
| 3 | `notebooks/03_condition_dependence.ipynb` — Group × Condition interaction (Layer 2), equivalence (Layer 3), robustness by domain (Layer 4), recording-batch effect | Available |
| 4 | `notebooks/04_stability_sensitivity.ipynb` — stability across window selections, amplitude normalization, analysis without HC_4 (Layer 5) | Available |
| 5 | `notebooks/05_figures.ipynb` — Figures 2–6 and the acquisition-artifact numbers reported in the Results | Available |
| 6 | `notebooks/06_supplementary_analyses.ipynb` — Tables S1–S3, Fig. S1, muscle-activity index, FCR-adjusted intervals | Available |

Feature extraction and Layer 1 statistics are in `src/eeg_features.py`, and Layers 2 and 3 in
`src/condition_tests.py`. Both modules are imported by notebooks 02, 03 and 04, so the
stability analyses use exactly the code of the main analysis.

## Data

The EEG data are the public dataset of Mumtaz (version 2): healthy controls (HC) and
patients with major depressive disorder (MDD), 19 channels of the 10-20 system, linked-ear
reference, 256 Hz, recorded in three conditions: eyes-closed rest (EC), eyes-open rest (EO)
and an auditory oddball task (TASK). The data are not redistributed here; download the EDF
files from the original source and place them, unchanged, in one folder.
https://figshare.com/articles/dataset/EEG_Data_New/4244171/2

## Pipeline

**Stage 0 — Dataset audit** (run on the original files)

- File inventory and check of the acquisition parameters and recording dates in the EDF headers.
- Duplicate detection with SHA-256 at two levels: the complete EDF file, and the signal of
  the 19 EEG channels alone (independent of header and metadata).
- Attribution of a record shared by several identifiers from the recording timestamps.
- Output: the list of excluded records, which is hard-coded in Stage 1 (`EXCLUDED_RECORDS`),
  and `file_inventory.csv`, used in Stage 3 for the recording batch of each control.

**Stage 1 — Preprocessing** (same pipeline for every record)

1. 19 channels of the 10-20 system in a fixed order (exported to `channel_order.json`)
2. Common average reference
3. FIR notch filter at 50 and 100 Hz; FIR band-pass 1–45 Hz
4. Removal of a narrowband acquisition artifact (4.5–6.5 Hz) by generalized
   eigendecomposition; up to two components with λ ≥ 0.425 are removed. The threshold is the
   midpoint of the gap in the λ₁ distribution of all records, set without group labels.
5. ICA (FastICA, 15 components, `random_state=42`) with ICLabel; components labeled as eye,
   heart, muscle, line noise or channel noise with probability > 0.7 are removed
6. Non-overlapping 2-s windows (512 samples)
7. Rejection of windows with peak-to-peak amplitude > 150 µV in any channel. The original
   index of each retained window is saved, so later stages build 10-s blocks only from truly
   consecutive windows.

**Stage 2 — Features and group comparison (Layer 1)**

- Analytic sample: subjects present in all three conditions, with at least 89 valid 2-s
  windows and three 10-s blocks of consecutive windows in every condition.
- Equalization: 2-s windows capped at the 10th percentile of the available windows, 10-s
  blocks at a common number, by random selection without replacement with deterministic
  seeds per subject and condition (base seed 42).
- 1,216 features: spectral (408), time-frequency (480), complexity (96), band ratios (96),
  hemispheric asymmetry (56), connectivity (80).
- Two-sided Mann-Whitney test with Benjamini-Hochberg FDR in each condition (the only
  detection criterion), Cliff's delta with bootstrap 95% CI, Hedges' g; descriptive
  detection pattern; Spearman correlation between window rejection and each feature within
  each group and condition, with FDR.

**Stage 3 — Condition dependence**

- Layer 2: Group × Condition interaction, closed-form F with p-values from 50,000
  permutations of group labels between subjects (main) and from F(2, 2(N−2)); post hoc
  contrasts from a REML linear mixed model; FDR over features.
- Layer 3: TOST equivalence of the three contrasts (margin ±0.30 SD), with sensitivity to
  0.20 and 0.50 SD, smallest attainable margin and minimum detectable effect.
- Layer 4: observed proportions of formal status by domain; participant-level bootstrap of
  the group effect and of its range across conditions; correlation clusters and effective
  dimension.
- Recording-batch effect: Cliff's delta of MDD vs all HC, MDD vs HC recorded outside the
  January 2013 session, and January vs other HC.

**Stage 4 — Stability and sensitivity (Layer 5)**

- Features and Layers 1–3 recomputed with 30 seeds of window and block selection (seed 42
  included); Jaccard index of the significant sets against seed 42, detection rate and
  dispersion of Cliff's delta by feature and family.
- Features and Layers 1–3 recomputed after dividing each record by one global RMS factor;
  feature-by-feature check of scale invariance.
- Layers 1–3 repeated without HC_4, whose TASK record was attributed from metadata.

**Stage 5 — Figures and artifact characterization**

- Figures 2–6 from the outputs of Stages 1–3 (no feature or test is recomputed); the raw EDF
  files are read only for the signal before correction.
- Acquisition artifact: corrected records by recording session, theta PLV before
  correction, theta and alpha power after correction against λ₁, and peaks at multiples of
  f₀ in the corrected data.

**Stage 6 — Supplementary analyses** (added during peer review)

- Windows per participant (Table S1), spectral composition (Table S2) and between-condition
  contrasts of every feature (Table S3 and Supplementary Data S1, from the Layer 2–3 mixed models).
- Muscle-activity index from the raw EDF files (log power 55–75 Hz) and sensitivity of the beta
  difference to it; Cliff's δ intervals adjusted for the false coverage rate; Hedges' g of
  relative beta; individual values (Fig. S1).

## How to run

```bash
pip install -r requirements.txt
```

Run the notebooks in order, setting the folders in the Configuration cell of each:

| Notebook | Reads | Writes |
|---|---|---|
| `00_duplicate_audit` | original EDF files | `OUTPUT_DIR` (audit) |
| `01_preprocessing` | original EDF files | `OUTPUT_DIR` (preprocessed data) |
| `02_features_group_comparison` | preprocessed data | `RESULTS_DIR` |
| `03_condition_dependence` | `RESULTS_DIR`, audit folder | `RESULTS_DIR` |
| `04_stability_sensitivity` | preprocessed data, `RESULTS_DIR` | `RESULTS_DIR/layer5` |
| `05_figures` | original EDF files, preprocessed data, `RESULTS_DIR` | `FIG_DIR` |
| `06_supplementary_analyses` | original EDF files, `RESULTS_DIR`, audit folder | `RESULTS_DIR/supplementary` |

`01_preprocessing` takes a few hours on CPU (Step 0, the artifact scan, takes a few minutes);
`02_features_group_comparison` takes about an hour; `04_stability_sensitivity` repeats that
work once per seed (several hours with 30 seeds; finished seeds are reused if the run is
interrupted). Notebooks 02–06 import the modules in `src/`: set `REPO_DIR` in their first code
cell to the repository root (default `..`, the parent of `notebooks/`). All notebooks also
run on Google Colab; the first code cell installs the dependencies and mounts Google Drive.

## Main outputs

| Stage | File | Content |
|---|---|---|
| 1 | `classical_<COND>/<ID>_windows.npy` | Retained windows, (n_windows, 512, 19), volts |
| 1 | `classical_<COND>/<ID>_window_indices.npy` | Original index of each retained window |
| 1 | `theta_artifact_scan.csv`, `processing_log.json` | Artifact scan and per-record preprocessing log |
| 2 | `features_<COND>.csv` | 1,216 features per subject |
| 2 | `stats_<COND>.csv`, `layer1_summary.csv` | Layer 1 statistics and detection pattern |
| 2 | `sampling_log_<COND>.csv` | Seeds and selected windows and blocks per subject |
| 2 | `parameters_features.json`, `cwt_frequencies.csv`, `connectivity_pairs.csv` | Parameters, CWT frequencies (Table S2), connectivity channel pairs |
| 3 | `layer2_interaction.csv` | Interaction test and post hoc contrasts per feature |
| 3 | `layer3_equivalence.csv`, `layer3_margin_*.csv` | Equivalence tests and margin sensitivity |
| 3 | `descriptive_vs_formal.csv` | Detection pattern versus formal tests |
| 3 | `layer4_*.csv` | Proportions, bootstrap, clusters and effective dimension by domain |
| 3 | `batch_effect_by_feature.csv`, `batch_effect_by_family.csv` | Recording-batch analysis |
| 4 | `layer5/seed_counts_summary.csv`, `layer5/jaccard_vs_seed42.csv` | Counts across seeds and Jaccard index against seed 42 |
| 4 | `layer5/feature_stability.csv`, `layer5/family_stability.csv` | Detection rate and Cliff's delta across seeds |
| 4 | `layer5/rms_summary.csv`, `layer5/without_HC_4_summary.csv` | Amplitude-normalization and HC_4 sensitivity analyses |
| 5 | `fig2`–`fig6` (`.pdf`, `.png`) | Figures of the article |
| 5 | `record_table.csv`, `harmonic_peaks.csv`, `fig6_values*.csv` | Date and theta PLV before correction per record, peaks at multiples of f₀, values of Fig. 6 |
| 6 | `supplementary/tabS_composicao.tex`, `supplementary/tabS_contrastes*.tex`, `supplementary/S3_contrasts.csv` | Tables S2 and S3, Supplementary Data S1 |
| 6 | `supplementary/emg_*.csv`, `supplementary/fcr_*.csv`, `supplementary/figS1_individual_values.*` | Muscle-activity index, FCR-adjusted intervals, Fig. S1 |

## Reproducibility

Random steps use fixed seeds: ICA (`random_state=42`), window and block selection (base seed
42, derived per subject and condition with SHA-256), bootstrap CIs, permutations
(`PERM_SEED`), the Layer 4 bootstrap (`BOOTSTRAP_SEED`) and the Layer 5 seeds
(`numpy.random.SeedSequence(20260930)`, saved in `layer5/seeds.csv`). The reported run used Python
3.13.18, MNE-Python 1.13.2 and NumPy 2.1.3 on Google Colab (CPU); the versions of the other
packages are printed by the Setup cell of each notebook.
