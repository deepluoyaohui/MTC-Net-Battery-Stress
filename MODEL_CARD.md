# Model Card: MTC-Net for Battery-Module Full-Field Stress Reconstruction

## 1. Model overview

**Model name:** Multimodal Temporal–Spatial Coupling Network (MTC-Net)  
**Task:** Reconstruction of two-dimensional battery-module stress fields from multimodal measurements  
**Framework:** PyTorch  
**Source-code repository:** https://github.com/deepluoyaohui/MTC-Net-Battery-Stress  
**Associated manuscript:** *Symbiotic sensory meta-structures enable discrete-sensor-free in situ full-field stress reconstruction and diagnosis of battery modules* (submitted to *Nature Communications*; editorial assessment stage).

MTC-Net is a research model designed to reconstruct a spatial stress distribution from six selected input features. Its components include causal dilated temporal convolution blocks, cross-modal attention, spatial positional encoding, and a Transformer-based spatial decoder. Each prediction contains **41 × 27 = 1,107** spatial stress values.

This document describes the research model and distinguishes the manuscript-reported experiments from the configuration of the currently supplied executable script. It does **not** assert that all manuscript results have been independently reproduced from the public code.

## 2. Intended uses

- Research on battery-module stress-field reconstruction and multimodal sensing.
- Evaluation of stress-field reconstruction under temperature and charge/discharge C-rate conditions represented in the study.
- Reproduction and methodological inspection by editors, reviewers, and researchers, subject to access to the accompanying dataset.

**Out-of-scope uses:** Safety-critical battery-management decisions, operational certification, deployment to different battery chemistries or geometries without additional validation, and prediction outside the studied operating range without independent testing. The model is not a replacement for physical safety testing.

## 3. Inputs and outputs

**Inputs:** Six features selected by the supplied implementation from the experimental feature tables. The source script uses zero-based column indices `[1, 11, 16, 17, 18, 19]` after its data-loading steps. The manuscript describes the integration of mechanical stress, electrochemical, and temperature information; see Section 4.2.1 and Equation (12). The exact mapping of each selected spreadsheet column to its physical quantity and unit should be checked against the released dataset before use.

**Output:** A reconstructed stress field on a **41 × 27** spatial grid. The supplied code labels the inverse-transformed stress values in **kPa**.

**Data organization:** The accompanying research dataset contains feature spreadsheets (`Data input`) and condition/time-step-organized stress-field arrays (`data input2`). Preserve the directory structure when extracting the archive.

## 4. Architecture

The manuscript describes the MTC-Net framework in **Section 4.2.1 and Figure 6**, including:

1. Temporal feature encoding with causal dilated convolution blocks.
2. Cross-modal feature interaction through an attention mechanism.
3. Spatial positional encoding for the output grid.
4. Transformer-based spatial decoding and regression of the full stress field.

The supplied implementation uses an embedding dimension of **128**, **four** attention heads, **two** spatial Transformer layers, a feed-forward dimension of **256**, and an output grid of **41 × 27**. Refer to the source code for the executable definitions and tensor transformations.

## 5. Training and evaluation data

The manuscript describes measurements and reference stress-field construction under multiple temperature and C-rate conditions (Sections 2–3). It evaluates generalization on independently held-out operating conditions in Section 4.2.2.

The supplied code identifies the following held-out temperature–C-rate conditions:

| Temperature | C-rate |
|---|---|
| −5 °C | 0.5 C |
| 10 °C | 1.5 C |
| 35 °C | 1.5 C |

For the supplied implementation, the remaining samples are used for training and validation; the final **10%** of the concatenated training/validation pool is reserved for validation without shuffling. MinMax scalers are fitted on training data only and applied to validation and test data. The exact sample ordering, split membership, and time-step alignment should be verified against the released dataset.

**Data access:** The complete dataset has been deposited as a Zenodo draft for editorial/reviewer access. The manuscript's stated post-publication data-availability approach is access from the corresponding author upon reasonable request. A draft review-access URL may be provided directly to authorized editors/reviewers; the reserved DOI alone is not a substitute for that access URL.

## 6. Training configurations and reproducibility status

**Important: the manuscript-reported protocol and currently supplied executable are not identical.**

| Item | Manuscript-reported experiment | Currently supplied executable |
|---|---|---|
| Maximum training epochs | 300 | 10 |
| Independent runs | 3 | 1 |
| Batch size | 32 | 32 |
| Optimizer | See manuscript, Section 4.2.2 | AdamW |
| Learning rate | See manuscript, Section 4.2.2 | 5 × 10⁻⁵ |
| Weight decay | See manuscript, Section 4.2.2 | 1 × 10⁻⁴ |
| Primary training loss | See manuscript, Equation (17) | Mean squared error |

The 10-epoch, single-run script is provided for code inspection and execution of the implemented pipeline. **Its results must not be presented as a reproduction of the manuscript's 300-epoch, three-run experiment unless that protocol is separately executed and verified.** The repository README documents the script-specific settings, environment, input paths, and output files. Training may require substantial computational resources; a verified end-to-end runtime is not yet available.

## 7. Manuscript-reported performance

The following figures are **reported in the submitted manuscript**, Section 4.2.2 (approximately pp. 22–23), and are **not newly measured results from the current 10-epoch code**:

- For the held-out **−5 °C / 0.5 C** and **10 °C / 1.5 C** conditions, the manuscript reports full-field mean absolute errors (MAEs) of **0.22–0.59 kPa** across the evaluated states of charge, root mean square error (RMSE) **below 0.75 kPa**, and normalized MAE (NMAE) of **0.23%–0.63%**.
- For **35 °C / 1.5 C**, the manuscript reports an MAE of **0.34 kPa** at **20% state of charge (SOC)**. At **100% SOC**, the reported MAE is **1.23 kPa** and RMSE is **1.44 kPa**, with maximum normalized absolute error (MaxNAE) within **2.7%**.

These values are associated with the study's experimental/reference-field evaluation protocol and should not be interpreted as performance guarantees for unseen battery designs or arbitrary operating conditions. Consult the manuscript's **Figures 7–8** and metric definitions for context.

## 8. Software and execution

The repository provides the MTC-Net Python source file, a `README.md`, and `requirements.txt`. Follow the README for dataset layout, environment installation, path configuration, training, evaluation, and exported outputs.

A previously reported local software environment includes **Python 3.11.13**, **PyTorch 2.7.1+cu118**, and **CUDA 11.8** on an **NVIDIA GeForce RTX 3050 Ti Laptop GPU**. The manuscript separately reports a workstation with an **NVIDIA RTX 4090 (24 GB)**. These are different reported environments; the current executable has not been verified on the manuscript workstation.

The README identifies known reproducibility limitations. In particular, check the input-file ordering, time-step pairing, feature-column semantics, and preprocessing assumptions before interpreting quantitative results.

## 9. Limitations and considerations

1. **Operating-domain coverage:** Generalization is supported by the conditions evaluated in the manuscript, not by an exhaustive test of all temperatures, C-rates, battery designs, aging states, or fault conditions.
2. **Reference-field dependence:** Model training and evaluation depend on the study's constructed reference stress fields; uncertainties in those reference fields may affect measured reconstruction accuracy.
3. **Input-data assumptions:** The implementation relies on dataset-specific filenames, feature-column positions, time-step alignment, and spatial-grid dimensions. Incorrect alignment or changes in data format may invalidate outputs.
4. **Training-protocol mismatch:** The public executable's 10-epoch single run is not the same as the manuscript's 300-epoch, three-run protocol. Direct numerical equivalence has not been established.
5. **Reproducibility:** Random seeds, exact split indices, full preprocessing artifacts, and a complete independent rerun of manuscript results have not all been verified for the supplied package.
6. **Safety:** The model is an experimental research tool, not a validated safety monitor or standalone diagnostic system for deployment.

## 10. Ethical and responsible use

The model processes laboratory battery-module measurement data rather than personal information. Users should nevertheless respect any restrictions associated with the experimental dataset, verify predictions against independent measurements where consequential, and avoid presenting exploratory reconstructions as certified safety assessments.

## 11. Citation and contact

When citing the model, refer to the associated manuscript and the repository version actually used. For research data access under the manuscript's stated availability policy, contact the corresponding author(s) identified in the manuscript. No open-source license is asserted by this model card; consult the repository for its current license status and permissions.

## 12. Model-card status

**Status:** Documentation prepared for editorial/reviewer assessment.  
**Model weights:** Availability and provenance of any released checkpoints should be verified separately.  
**Performance provenance:** Manuscript-reported values, not independently re-benchmarked in this document.  
**Last updated:** 2026-10-29.
