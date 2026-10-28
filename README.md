# MTC-Net: Multimodal Stress-Field Reconstruction

## 1. Overview

This repository provides a PyTorch implementation of the Multi-modal Temporal-spatial Coupling Network (MTC-Net) for reconstructing two-dimensional stress fields from six input features. The network includes causal dilated convolution blocks, cross-modal attention, spatial positional encoding, and a Transformer-based spatial decoder. Each output comprises **41 × 27 = 1,107** stress values.

The supplied executable is `MTC_Net_Aligned.py`. It covers data loading, preprocessing, model training, held-out-condition evaluation, and export of reconstructed stress fields. The data traversal logic retains alignment between successfully loaded conditions and their metadata.

## 2. Software and hardware

The following environment was reported for the supplied code:

| Component | Reported configuration |
| --- | --- |
| Operating system | Windows (version not confirmed for this execution environment) |
| Python | 3.11.13 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |

The associated manuscript describes experiments on **Windows 11**, an **Intel Core i9-13900K CPU @ 3.00 GHz**, and an **NVIDIA GeForce RTX 4090 GPU (24 GB VRAM)**. These are manuscript-reported specifications, not a verified execution record for the supplied script.

Required Python packages are `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, and `openpyxl` (for Excel I/O). Install a PyTorch build compatible with the local CUDA setup, followed by the other packages:

```bash
python -m pip install numpy pandas scikit-learn matplotlib tqdm openpyxl
```

For exact environment reproduction, provide the package versions from the environment used for the reported experiments in a separate `requirements.txt` file. Versions other than Python and PyTorch have not been confirmed here.

## 3. Dataset organization and preprocessing

The script uses two input roots, specified by `input_dir` and `array_base_dir`. The expected arrangement is illustrated below (placeholder names are not literal filenames):

```text
data/
├── Data input/
│   └── <condition-specific feature table>.xlsx
└── data input2/
    └── <condition-specific folder>/
        └── Arrays/
            ├── TimeStep_....csv
            └── TimeStep_....xlsx
```

Feature spreadsheets and stress-field folders are matched using temperature and C-rate extracted from their names. The script selects six feature columns at zero-based indices `[1, 11, 16, 17, 18, 19]`, using `df.iloc[1:, ...]`. Stress-field files are read as **41 × 27** arrays, taking the bottom-right region of larger tables. Successfully loaded fields are aligned to feature rows according to their positions in the sorted timestep-file list; unreadable files are skipped without shifting subsequent row positions.

The dataset must preserve the naming convention and the correspondence between feature rows and timestep files. The six feature meanings, units, and complete dataset directory tree should be documented with the distributed dataset.

## 4. Network architecture

| Parameter | Value |
| --- | --- |
| Input features | 6 (split into two 3-feature branches) |
| Embedding dimension (`d_model`) | 128 |
| Attention heads | 4 |
| Spatial Transformer encoder layers | 2 |
| Feed-forward dimension | 256 |
| Output head | 128 → 64 → 1 |
| Spatial grid | 41 × 27 (1,107 locations) |

The implementation projects each three-feature branch into an embedding, processes the embeddings through causal dilated convolution blocks and cross-attention, and decodes spatial tokens into stress values. **Implementation note:** the convolution blocks currently receive a sequence dimension of length one; the executable does not construct a multi-timestep historical input window.

## 5. Data partitioning and training

The following conditions are designated as held-out blind tests:

- 35 °C, 1.5 C
- 10 °C, 1.5 C
- −5 °C, 0.5 C

The remaining loaded samples form the training/validation pool. The final 10% of the concatenated pool is assigned to validation (`shuffle=False`). MinMax normalization parameters are fitted on training samples only and applied to validation and test samples.

**Configuration of the supplied executable:**

| Setting | Value |
| --- | --- |
| Epochs | 10 |
| Training batch size | 32 |
| Optimizer | AdamW |
| Initial learning rate | 5 × 10⁻⁵ |
| Weight decay | 1 × 10⁻⁴ |
| Loss | Mean squared error (MSE) |
| Scheduler | Cosine annealing (`eta_min=1e-6`) |
| Gradient clipping | Maximum norm 0.5 |
| Automatic mixed precision | Disabled |

Validation is computed at epochs 1, 5, and 10 under this configuration. The plotted validation values at other epochs are carried forward rather than separately measured.

**Relationship to the manuscript:** The manuscript describes a maximum of **300 epochs** and **three independent training runs**, with mean results reported. The supplied executable instead performs **one 10-epoch run**. Consequently, results from this executable should not be presented as a reproduction of the manuscript's three-run, 300-epoch protocol without performing and documenting those experiments. The architecture settings listed above agree with the manuscript description.

## 6. Running the code

1. Install Python and the dependencies listed in Section 2.
2. Place the feature tables and stress-field arrays in the directory structure described in Section 3.
3. Edit `input_dir`, `array_base_dir`, `save_dir`, `load_path`, and `output_dir` in `MTC_Net_Aligned.py` to match the local machine. The original script uses absolute Windows paths and two separate output locations.
4. Run the complete pipeline:

```bash
python "MTC_Net_Aligned.py"
```

The script loads and preprocesses the data, trains the network, saves artifacts, reloads data and weights, evaluates held-out conditions, and exports selected stress fields. It is not packaged as a separate inference-only command. Only load `.pt` files from trusted sources because the saved data bundles include serialized Python objects.

## 7. Generated outputs

The default script uses two output roots:

```text
MTC-Net_StressField_Results/
└── processed_tensors.pt

MTC-Net_输出结果/
├── MTC_Net_Best_Weights.pth
├── Loss_Convergence_Curve.png
├── Processed_Data_Bundle.pt
├── Excel_Arrays/
│   ├── Pred_Field_...xlsx
│   ├── True_Field_...xlsx
│   └── Error_Field_...xlsx
└── Stress_Fields/
    └── Field_Comparison_...png
```

The program computes **MAE, RMSE, and R²** on inverse-normalized held-out predictions and reference fields. These metrics are printed to the console; the supplied script does not save a separate metrics summary file. It exports up to **50** selected test samples as predicted, reference, and signed-error Excel matrices, along with **500-DPI** comparison figures showing the reference field, predicted field, and absolute error. Exported matrices are vertically flipped for visualization. The figure labels express stress in **kPa**, consistent with the code; confirm units against the source dataset before publication.

## 8. Reproducibility considerations

- The exact input file naming convention and feature-row/timestep correspondence must be preserved.
- The loader skips the first data row after reading the spreadsheet header; confirm that this matches the distributed input files.
- Timestep filenames are sorted using all digits concatenated from each filename; filenames must therefore follow a convention that preserves the intended order.
- The script does not fix random seeds or implement the manuscript's three-run aggregation.
- Saved data bundles include `scaler_y`, but not `scaler_X`; additional preprocessing metadata is required for inference on arbitrary new raw features.
- Runtime and numerical accuracy depend on the supplied dataset and hardware. No measured runtime or performance values are asserted in this document.

## 9. Notes on the manuscript and supplementary implementation

This Python script is supplied as supporting documentation of the MTC-Net architecture, data-processing pipeline, training procedure, and stress-field reconstruction workflow. It should be distinguished from the complete experimental protocol reported in the manuscript.

The manuscript reports a Windows 11 workstation with an Intel Core i9-13900K CPU and an NVIDIA GeForce RTX 4090 GPU (24 GB VRAM). The environment previously reported for the supplied script uses Python 3.11.13, PyTorch 2.7.1+cu118, CUDA 11.8, and an NVIDIA GeForce RTX 3050 Ti Laptop GPU; its Windows version has not been confirmed.

The model architecture and principal optimizer settings correspond to those described in the manuscript. However, the supplied script is configured for 10 training epochs and one run, whereas the manuscript specifies a maximum of 300 epochs and three independent runs. The supplementary script should therefore be treated as an implementation reference, not as independent verification of the manuscript's full training protocol or numerical results.
