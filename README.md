# Cerebral-Vessel-segmentation (TopCoW / Circle of Willis)

This project is a **PyTorch-based 3D medical imaging pipeline** for:

- **Phase 1 — ROI localization**: predict a **3D bounding box** around the Circle of Willis (CoW) from CTA volumes.
- **Phase 2 — multi-class segmentation**: perform **16-class voxel-wise segmentation** inside the localized region using 3D segmentation networks (3D U-Net [3], V-Net [4], nnU-Net [5], attention variants [6, 7], and experimental architectures).

The repo is **notebook-driven**: the core workflow lives in `Complete Pipeline.ipynb`, and additional segmentation architectures live in `Independent Segmentation Models/`.

---

## What problem this solves

Full head/brain CTA scans are large 3D volumes. Segmenting tiny vessels directly on the full volume is inefficient and error-prone due to:

- **Severe class imbalance** (background vastly outnumbers vessels),
- **High memory cost** for 3D CNNs,
- **Noise / low vessel contrast** in some modalities.

This project addresses it by first predicting a **tight ROI bounding box**, then training segmentation models on **cropped + resized** volumes (dataset format and split strategy aligned with TopCoW [2]).

---

## Repository structure

```text
Cereberal-Vessel-segmentation/
├─ Complete Pipeline.ipynb
├─ README.md
└─ Independent Segmentation Models/
   ├─ 3d-unet-topcow2024.ipynb
   ├─ 3d-vnet-topcow2024.ipynb
   ├─ attention-unet-topcow2024.ipynb
   ├─ attention-vnet-topcow2024.ipynb
   ├─ nnunet-v2-topcow2024.ipynb
   └─ quantum-gate-3d-vnet-topcow-2024.ipynb
```

### `Complete Pipeline.ipynb` (main end-to-end notebook)

This is the **complete pipeline** notebook. It contains:

#### Phase 1 — ROI localization (bounding box regression)

- **Dataset indexing**: pairs each scan with its ROI text label and segmentation mask (where available).
- **Modality-aware intensity normalization**:
  - CT: clip HU range then normalize.
- **Tight crop + resize**: crops the volume around the ROI (with margin), then resizes the crop to a fixed patch size.
  - Tracks `start` and `zoom_factors` so predicted ROIs can be **inverted back** into the original voxel space.
- **Model**: `ROIRegressor3D` (3D CNN encoder + GAP + MLP head) predicting 6 values: \([x, y, z, dx, dy, dz]\) in **normalized patch coordinates**.
- **Loss**: combined regression loss (Smooth L1 + centroid penalty + 3D IoU loss).
- **Training + evaluation**: training loop, early stopping, MAE and IoU validation reporting, plus visualization of predicted vs GT boxes.

#### ROI inference for validation/test scans

- Runs the ROI regressor on validation/test scans.
- Writes predicted ROI coordinates as `.txt` files to a configured output directory (in the notebook it is set to a Kaggle working path).

#### Phase 2 — segmentation data preparation

- **Crops the original image and label volumes consistently** using the ROI (GT ROI for training prep).
- Optional **vessel enhancement** using a Frangi-style vesselness filter (slice-wise) [18].
- Resizes volumes to a fixed segmentation size (e.g., `TARGET_SIZE = (128, 128, 64)`).
  - **Important**: image uses trilinear interpolation (`order=1`), labels use **nearest neighbor** (`order=0`).
- Saves the prepared data to:
  - `seg_train_images/` (images)
  - `seg_train_labels/` (multi-class masks)

#### Phase 2 — segmentation training & comparison

Inside the same notebook, segmentation training includes:

- **Segmentation dataset + dataloaders** (`SegmentationDataset`)
- **Class imbalance handling**: inverse-frequency class weights computed over all labels
- **Loss functions**:
  - CrossEntropy + Dice
  - Focal/Tversky/Dice variants (depending on experiment block)
- **Metrics**: Dice, IoU, precision, recall, F1, specificity, HD95, clDice (approx), Betti-like proxy
- **Architectures / experiment blocks**:
  - 3D U-Net baseline
  - VNet, Attention U-Net, Attention VNet, nnU-Net-like variant, and a Quantum-gated VNet-like model
- **Checkpointing + plots**: saves best checkpoints and attempts to plot/compare metrics across experiments.

> Note: the notebook paths are configured for Kaggle (e.g., `/kaggle/input`, `/kaggle/working`). If you run locally, update those path variables accordingly.

---

## Independent Segmentation Models (model-focused notebooks)

The notebooks in `Independent Segmentation Models/` are **standalone segmentation experiments**, each focusing on one architecture/training recipe. These are useful when you want:

- a cleaner notebook for one model (instead of a single huge comparison notebook),
- faster iteration on architecture/loss/metrics,
- reproducible runs per-model.

Files:

- `3d-unet-topcow2024.ipynb`: baseline 3D U-Net segmentation experiment [3].
- `3d-vnet-topcow2024.ipynb`: VNet-style 3D segmentation experiment [4].
- `attention-unet-topcow2024.ipynb`: attention-gated 3D U-Net variant [7].
- `attention-vnet-topcow2024.ipynb`: attention-augmented VNet variant [6].
- `nnunet-v2-topcow2024.ipynb`: nnU-Net-inspired training/pipeline variant [5].
- `quantum-gate-3d-vnet-topcow-2024.ipynb`: experimental quantum-gated VNet-style model.

---

## How to run (recommended workflow)

### Option A — Run the full pipeline notebook

1. Open `Complete Pipeline.ipynb`.
2. Update path variables near the top:
   - dataset root (images, ROI labels, segmentation labels)
   - checkpoint output locations
   - working output directories
3. Run Phase 1 (ROI regressor) cells:
   - build index → preprocessing → dataset/dataloader → model/loss → training (or load checkpoint) → evaluation.
4. Run ROI inference to produce predicted ROI `.txt` files.
5. Run segmentation prep cells to create `seg_train_images/` and `seg_train_labels/`.
6. Train segmentation models (3D U-Net and/or the comparison blocks).

### Option B — Run one segmentation architecture notebook

1. Prepare segmentation data first (either using the prep section in `Complete Pipeline.ipynb` or your own preprocessing).
2. Open a notebook inside `Independent Segmentation Models/`.
3. Point it to the prepared `seg_train_images/` and `seg_train_labels/` directories and run training/evaluation.

---

## Inputs & outputs (conceptual)

### Typical inputs

- **3D volumes**: `.nii` / `.nii.gz` (CTA)
- **ROI labels**: `.txt` with ROI location + size in voxel coordinates
- **Segmentation masks**: `.nii` / `.nii.gz` integer-labeled masks (multi-class)

### Typical outputs

- **ROI predictions**: `.txt` files with predicted box location/size (voxel coordinates in original space)
- **Prepared segmentation dataset**: cropped/resized `.nii.gz` volumes + corresponding label masks
- **Model checkpoints**: `.pth` files saved when validation score improves
- **Plots/curves**: training/validation curves and metric comparison figures (when enabled)

---

## References 

[1] J. P. Villablanca, R. Jahan, P. Hooshi, S. Lim, G. Duckwiler, A. Patel, J. Sayre, N. Martin, J. Frazee, J. Bentson, and F. Vinuela, “Detection and characterization of very small cerebral aneurysms by using 2D and 3D helical CT angiography,” *AJNR American Journal of Neuroradiology*, vol. 23, pp. 1187–1198, 2002.  
[2] K. Yang et al., “TopCoW challenge,” 2024.  
[3] O. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in *MICCAI*, 2016, pp. 424–432.  
[4] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in *3DV*, 2016, pp. 565–571.  
[5] F. Isensee et al., “nnU-Net: Self-adapting framework for U-Net-based medical image segmentation,” *arXiv preprint arXiv:1809.10486*, 2018.  
[6] X. Liu et al., “Attention V-Net,” *Applied Sciences*, 2022.  
[7] N. Das and S. Das, “Attention U-Net,” *Current Problems in Cardiology*, 2024.  
[8] A. Zeng et al., “ImageCAS dataset for coronary artery segmentation,” *Computerized Medical Imaging and Graphics*, 2023.  
[9] J. Wei et al., “Knowledge-augmented aneurysm detection,” *Radiology*, 2024.  
[10] B. Lengyel et al., “Stroke risk assessment using CoW,” *Journal of Clinical Medicine*, 2024.  
[11] J. Wang et al., “Detection of intracranial aneurysms using CTA,” *Academic Radiology*, 2023.  
[12] P. Shi et al., “Centerline boundary Dice loss,” *arXiv*, 2024.  
[13] Y. Kirchhoff et al., “Skeleton recall loss,” *arXiv*, 2024.  
[14] N. Stucki et al., “Topology-aware segmentation using Betti matching,” *arXiv*, 2024.  
[15] C. Prabhakar et al., “3D vessel graph generation using diffusion,” *arXiv*, 2024.  
[16] L. Li et al., “Universal topology refinement,” *arXiv*, 2024.  
[17] Z. Zhao et al., “DeformCL: Deformable centerline representation,” *arXiv*, 2025.  
[18] G. Yang et al., “Vessel structure extraction using constrained minimal path propagation,” *Artificial Intelligence in Medicine*, vol. 105, 2020.  
[19] A. Hoopes et al., “SynthStrip: Skull-stripping for any brain image,” *NeuroImage*, vol. 260, 2022.  
[20] T. R. Patel et al., “3D deep learning pipeline for cerebral vessel segmentation,” *Neurosurgical Focus*, 2023.  
[21] O. U. Aydin et al., “3D StyleGAN for Circle of Willis,” *medRxiv*, 2024.  
[22] A. M. Ceballos-Arroyo et al., “Vessel-aware aneurysm detection,” in *MICCAI*, 2024.  
[23] W. You et al., “Aneurysm diagnosis using deep learning,” *Journal of NeuroInterventional Surgery*, 2025.  
[24] L. Wang et al., “Lifespan-generalizable skull-stripping model,” *Nature Biomedical Engineering*, 2025.  
[25] A. Koch et al., “Cross-modality CTA synthesis,” *arXiv*, 2024.  

