# 🚀 A Novel CNN Regression-Based Reversible Data Hiding for Adaptive Embedding Without Overhead

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Task](https://img.shields.io/badge/Task-Reversible_Data_Hiding-success)
![Overhead](https://img.shields.io/badge/Overhead-0_Bits-brightgreen)
![Status](https://img.shields.io/badge/Research-State--of--the--Art-purple)

This repository contains the official PyTorch implementation and research validation for the paper: **"A Novel CNN Regression based Reversible Data Hiding for Adaptive Embedding without Overhead"**. 

Our framework introduces an adapted **invertible U-Net (iUNet)** that proactively reshapes cover image histograms through learnable transformations. By preemptively creating zero-frequency bins, our model completely **eliminates communication overhead**, boosts histogram peaks for high-payload multi-bit embedding, and applies reverse-mode enhancement to obscure statistical artifacts and achieve state-of-the-art visual fidelity.

---

## 📊 Executive Summary & Performance Highlights

| Metric / Feature | Dual-Peak Scheme ($\kappa=15$) | Single-Peak Scheme ($\kappa=7$) | SOTA Improvement |
| :--- | :---: | :---: | :---: |
| **Average Embedding Capacity** | **8,140.88 bits** | **7,224.50 bits** | **+3.47% to +354%** higher payload |
| **Average Visual Quality (PSNR)** | **34.47 dB** | **40.26 dB** | **+345%** better than comparable 3-bit SOTA |
| **Structural Similarity (SSIM)** | **0.9499** | **0.9821** | Near-perfect structural preservation |
| **Communication Overhead** | **0 Bits** | **0 Bits** | Eliminates location maps & aux channels |
| **Data Recovery & Reversibility** | **100% Lossless** | **100% Lossless** | Perfect cover & secret bitstream recovery |

---

## 🏗️ System Architecture & Pipeline Visuals

### 1. The Four-Stage End-to-End Pipeline
The proposed Reversible Data Hiding (RDH) framework operates across four distinct stages to guarantee zero overhead and complete reversibility:

```
================================== TRANSMISSION END ==================================
[Cover Image] 
     │
     ▼ (Squeeze: 1 Ch ➔ 4 Ch, H/2 x W/2)
[Stage A: Pre-processing & Zero-Bin Creation via Forward CNN]
     │  ├── Proactive Histogram Shifting (Creates Zero-Frequency Bins)
     │  └── Leftward Histogram Shifting (Consolidates Empty Space)
     ▼
[Stage B: Multi-Bit Message Embedding]
     │  ├── Embed Decimal Values into Left & Right Histogram Peaks (p_i = p_i ± d_i)
     │  └── Quality Enhancement via Reverse-Mode CNN (Obscures Step-Like Artifacts)
     ▼
[Marked Cover Image Transmitted]

=================================== RECEIVING END ===================================
[Marked Cover Image Received]
     │
     ▼
[Stage C: Secret Message Extraction]
     │  ├── Revert Structural Enhancement via Forward CNN
     │  ├── Extract Secret Bits from Left & Right Peaks (d_i = |p_i' - p_i|)
     │  └── Restore Histogram Peak Positions
     ▼
[Stage D: Lossless Cover Recovery]
     │  ├── Rightward Histogram Shifting (Restores Original Distribution)
     │  └── Inverse CNN + Unsqueeze (4 Ch ➔ 1 Ch, H x W)
     ▼
[Recovered Cover Image (100% Lossless) + Extracted Secret Bitstream]
```

### 2. Adapted iUNet Architecture & Reversible Blocks
Our neural network replaces traditional non-invertible operations with mathematically exact bijective transformations:

* **Squeeze / Unsqueeze Downsampling:** Transforms tensor dimensions from $(B, C, H, W) \leftrightarrow (B, 4C, H/2, W/2)$ without losing spatial information.
* **Additive Coupling Layers:** Inspired by NICE and RealNVP, splitting input $\mathcal{X}$ into $\mathcal{X}_1$ and $\mathcal{X}_2$:
  $$\mathcal{Y}_1 = \mathcal{X}_1 + \mathcal{F}(\mathcal{X}_2), \quad \mathcal{Y}_2 = \mathcal{X}_2 + \mathcal{G}(\mathcal{Y}_1)$$
  *Inverse Operation:*
  $$\mathcal{X}_2 = \mathcal{Y}_2 - \mathcal{G}(\mathcal{Y}_1), \quad \mathcal{X}_1 = \mathcal{Y}_1 - \mathcal{F}(\mathcal{X}_2)$$
* **Loss Functions:** Uses a hybrid objective function combining Mean Squared Error ($\mathcal{L}_{MSE}$) with customized forbidden-region penalty losses ($\mathcal{L}_{low}$ and $\mathcal{L}_{high}$) to enforce learnable gamma correction ($\gamma=0.8$) and pixel clamping.

---

## 📈 Comprehensive Quantitative Evaluation

All experiments were conducted on 100 randomly selected $256 	imes 256$ grayscale images from the **ImageNet** test dataset. Below are the detailed benchmarks comparing our proposed framework against leading spatial-domain and high-capacity state-of-the-art (SOTA) techniques.

### Table 1: Internal Performance Across Test Images (Capacity, PSNR, SSIM)
Evaluating the trade-off between the High-Capacity Dual-Peak Scheme ($\kappa=15$) and High-Fidelity Single-Peak Scheme ($\kappa=7$):

| Test Image | Max Capacity ($\kappa=15$) | Max Capacity ($\kappa=7$) | PSNR ($\kappa=15$) | PSNR ($\kappa=7$) | SSIM ($\kappa=15$) | SSIM ($\kappa=7$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image 1** | 7,848 bits | 4,032 bits | 29.63 dB | 35.78 dB | 0.9614 | 0.9860 |
| **Image 2** | 6,534 bits | 3,366 bits | 27.88 dB | 33.52 dB | 0.9579 | 0.9856 |
| **Image 3** | 13,161 bits | 12,648 bits | 43.81 dB | 50.77 dB | 0.9367 | 0.9912 |
| **Image 4** | 8,409 bits | 4,518 bits | 30.11 dB | 36.00 dB | 0.9614 | 0.9843 |
| **Image 5** | 3,678 bits | 1,947 bits | 31.28 dB | 37.41 dB | 0.9138 | 0.9817 |
| **Image 6** | 21,810 bits | 14,946 bits | 37.77 dB | 41.89 dB | 0.9651 | 0.9734 |
| **Image 7** | 4,722 bits | 2,103 bits | 43.88 dB | 50.08 dB | 0.9478 | 0.9762 |
| **Image 8** | 8,658 bits | 4,521 bits | 31.03 dB | 36.64 dB | 0.9375 | 0.9782 |
| **AVERAGE**| **8,140.88 bits** | **7,224.50 bits** | **34.47 dB** | **40.26 dB** | **0.9499** | **0.9821** |

---

### Table 2: Ablation Study — Impact of Post-Embedding CNN Enhancement
Demonstrating how applying the reversible CNN in reverse mode post-embedding dramatically restores visual quality and removes steganalysis vulnerabilities:

| Test Image | PSNR (Before Enhancement) | PSNR (After Enhancement) | PSNR Gain ($\Delta$ dB) | SSIM (Before Enhancement) | SSIM (After Enhancement) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Image 1** | 22.25 dB | **35.78 dB** | **+13.53 dB** | 0.9511 | **0.9614** |
| **Image 2** | 21.51 dB | **33.52 dB** | **+12.01 dB** | 0.9453 | **0.9579** |
| **Image 3** | 26.41 dB | **50.77 dB** | **+24.36 dB** | 0.8392 | **0.9367** |
| **Image 4** | 20.92 dB | **36.00 dB** | **+15.08 dB** | 0.9474 | **0.9614** |
| **Image 5** | 20.13 dB | **37.41 dB** | **+17.28 dB** | 0.9281 | **0.9138** |
| **Image 6** | 19.48 dB | **41.89 dB** | **+22.41 dB** | 0.9025 | **0.9651** |
| **Image 7** | 19.42 dB | **50.08 dB** | **+30.66 dB** | 0.8946 | **0.9478** |
| **Image 8** | 20.42 dB | **36.64 dB** | **+16.22 dB** | 0.9146 | **0.9375** |
| **AVERAGE**| **20.77 dB** | **40.26 dB** | **+19.49 dB (+93.8%)**| **0.9314** | **0.9499** |

---

### Table 3: Communication Overhead Comparison (in Bits)
Proof of complete overhead eradication compared to traditional and multi-stage methods:

| Test Image | Proposed Scheme | Kumar et al. [9] *(3-Bit)* | Fallahpour et al. [10] *(1-Bit)* | Ni et al. [29] *(1-Bit)* | Wang et al. [11] *(1-Bit)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Image 1** | **0** | 2,380 | 8 | 144 | 0 |
| **Image 2** | **0** | 0 | 8 | 16 | 0 |
| **Image 3** | **0** | 1,024 | 8 | 720* *(Failed)* | 0 |
| **Image 4** | **0** | 0 | 8 | 16 | 0 |
| **Image 5** | **0** | 1,268 | 8 | 144 | 0 |
| **Image 6** | **0** | 10,328 | 8 | 16 | 0 |
| **Image 7** | **0** | 0 | 8 | 16 | 0 |
| **Image 8** | **0** | 0 | 8 | 48 | 0 |
| **AVERAGE**| **0 Bits** | **3,538.13 Bits** | **8 Bits** | **140 Bits** | **0 Bits** |

---

### Table 4: State-of-the-Art Capacity Comparison (Total Embedded Bits)
Our dual-peak scheme ($\kappa=15$) achieves the highest overall payload, outperforming 1-bit methods by over **300%** and surpassing leading 3-bit methods:

| Test Image | Proposed ($\kappa=15$) | Proposed ($\kappa=7$) | Kumar et al. [9] *(3-Bit)* | Fallahpour et al. [10] *(1-Bit)* | Ni et al. [29] *(1-Bit)* | Wang et al. [11] *(1-Bit)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image 1** | **7,848** | 4,032 | 7,148 | 1,420 | 982 | 1,126 |
| **Image 2** | **6,534** | 3,366 | 4,632 | 1,047 | 1,047 | 4,632 |
| **Image 3** | 13,161 | **12,648** | 3,258 | 756 | 506 | 506 |
| **Image 4** | **8,409** | 4,518 | 8,328 | 1,951 | 1,404 | 1,404 |
| **Image 5** | **3,678** | 1,947 | 3,822 | 969 | 670 | 670 |
| **Image 6** | **21,810** | 14,946 | 19,416 | 6,614 | 6,587 | 6,603 |
| **Image 7** | 4,722 | **2,103** | 7,225 | 2,005 | 1,408 | 1,424 |
| **Image 8** | **8,658** | 4,521 | 9,114 | 2,079 | 1,498 | 1,546 |
| **AVERAGE**| **8,140.88** | **7,224.50** | **7,867.88** | **2,158.50** | **1,942.29** | **1,790.75** |

---

### Table 5: Visual Quality (PSNR) Comparison Across Methods
While low-capacity 1-bit methods naturally achieve high PSNR due to minor alterations, our method completely shatters the fidelity barrier for **High-Capacity** RDH, beating [9] by **+31.22 dB (+345%)**:

| Test Image | Proposed ($\kappa=15$) | Proposed ($\kappa=7$) | Kumar et al. [9] *(3-Bit)* | Fallahpour et al. [10] *(1-Bit)* | Ni et al. [29] *(1-Bit)* | Wang et al. [11] *(1-Bit)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image 1** | 29.63 dB | **35.78 dB** | 8.82 dB | 52.59 dB | 53.52 dB | 69.06 dB |
| **Image 2** | 27.88 dB | **33.52 dB** | 8.83 dB | 50.04 dB | 69.31 dB | 69.31 dB |
| **Image 3** | 43.81 dB | **50.77 dB** | 8.96 dB | 51.45 dB | 72.35 dB | 72.35 dB |
| **Image 4** | 30.11 dB | **36.00 dB** | 9.65 dB | 50.81 dB | 52.59 dB | 68.05 dB |
| **Image 5** | 37.41 dB | **31.28 dB** | 8.75 dB | 50.30 dB | 51.62 dB | 70.98 dB |
| **Image 6** | 41.89 dB | **37.77 dB** | 8.85 dB | 58.46 dB | 54.83 dB | 61.04 dB |
| **Image 7** | 50.08 dB | **43.88 dB** | 9.31 dB | 49.41 dB | 67.60 dB | 67.60 dB |
| **Image 8** | 36.64 dB | **31.03 dB** | 9.17 dB | 50.20 dB | 51.88 dB | 67.55 dB |
| **AVERAGE**| **34.47 dB** | **40.26 dB** | **9.04 dB** | **51.66 dB** | **52.89 dB** | **68.24 dB** |

---

## 🔬 Qualitative Analysis & Histogram Evolution

A major breakthrough of our model is visible in its effect on the image histogram across the embedding lifecycle (referenced in **Figures 5 through 12** in the manuscript):

1. **Proactive Zero-Bin Creation (Sub-figure b):** Regardless of whether the original image histogram is dark-biased, uniform, or highly peaked, the forward pass of our reversible CNN shifts pixel distributions rightward. This carves out exact zero-frequency vacant bins ($0$ to $2^n-1$) on the dark side, completely eliminating the risk of underflow/overflow during shifting.
2. **Peak Boosting:** As a natural byproduct of histogram compression, peak frequencies are boosted, providing massive contiguous capacity for multi-bit-per-pixel insertion.
3. **Obscuring Step-Like Artifacts (Sub-figures d & f):** Traditional histogram shifting leaves unmistakable flat "plateaus" or step-like distributions where secret data is inserted, making the image vulnerable to steganalysis. Crucially, when we apply our CNN in **reverse mode** after embedding, it redistributes the pixel intensities back toward their natural curves. This effectively **retrievably obscures** the data-hiding artifacts, resulting in smooth, natural-looking histograms with superior anti-detection properties.

---

## 📂 Repository Contents & Notebook Guide

```
├── 📁 model_train_notebook.ipynb  # Training pipeline: Dataset loading, ReversibleBlock definition, custom loss functions
├── 📁 model_test_notebook.ipynb   # Evaluation pipeline: Bidirectional shifting, bitstream embedding/extraction, PSNR/SSIM calculation
├── 📄 SpringerManuscript.pdf      # Complete research paper with mathematical proofs, ablation studies, and full visual appendices
└── 📄 README.md                   # Project documentation and benchmark dashboard
```

### 🧠 Notebook Breakdown:
* **`model_train_notebook.ipynb`**:
  * Implements the `ReversibleBlock`, `InvertibleUNet`, and `SqueezeNetWrapper`.
  * Configures the Adam optimizer ($lpha=10^{-4}, \lambda=10^{-5}$) and trains across ImageNet subsets using the custom threshold-penalty loss functions ($\mathcal{L}_{MSE} + \mathcal{L}_{low} + \mathcal{L}_{high}$).
  * Saves epoch checkpoints for bidirectional and single-peak evaluation.
* **`model_test_notebook.ipynb`**:
  * Loads trained model weights (`.pth`).
  * Executes `test_image_reversibility()`, running the complete 4-stage pipeline on test images.
  * Dynamically computes and outputs quantitative metrics (MSE, PSNR, SSIM, Bit Capacity) and visualizes side-by-side histogram comparisons.

---

## 🚀 Getting Started & Reproduction

### Prerequisites
Ensure you have Python 3.8+ installed along with the required deep learning and visualization libraries:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

### Quickstart Execution
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/CNN-Reversible-Data-Hiding.git
   cd CNN-Reversible-Data-Hiding
   ```
2. **Train the Model (Optional):**
   Open `model_train_notebook.ipynb` in Google Colab or Jupyter Notebook. Set your dataset path (ImageNet subset) and run all cells to train the invertible U-Net from scratch.
3. **Run Evaluation & Benchmarking:**
   Open `model_test_notebook.ipynb`. Verify the checkpoint path (e.g., `checkpoint_epoch_one_side_40.pth`) and point `image_to_test_path` to any sample image (e.g., `ILSVRC2017_test_00005135.JPEG`). Execute the cells to observe real-time embedding, lossless extraction, and automated PSNR/SSIM calculation.

---

## ✒️ Authors & Affiliations

* **Shubham Makhija** *(MAIT, GGSIPU, Delhi, India)* — [shubhammakhija19@gmail.com](mailto:shubhammakhija19@gmail.com)
  * *Contribution:* Conceptualization, Methodology, Validation, and Manuscript Review.
* **Sanjay Kumar** *(SAG, DRDO, Delhi, India)* — [skparsade@gmail.com](mailto:skparsade@gmail.com) *(Corresponding Author)*
  * *Contribution:* Methodology, Formal Analysis, Data Curation, Resource Acquisition, and Original Draft Writing.
* **Gurjit Singh Walia** *(SAG, DRDO, Delhi, India)* — [gurjit.walia@gmail.com](mailto:gurjit.walia@gmail.com)
  * *Contribution:* Formal Analysis, Resource Acquisition, Reviewing, and Editing.
* **Sumit Kumar** *(SAG, DRDO, Delhi, India)* — [sumitkr@hotmail.com](mailto:sumitkr@hotmail.com)

---

## 📜 License & Citation
This repository is released for academic and research purposes. If you use this code, architecture, or benchmark results in your research, please cite our paper:

```bibtex
@article{makhija2025novel,
  title={A Novel CNN Regression based Reversible Data Hiding for Adaptive Embedding without Overhead},
  author={Makhija, Shubham and Kumar, Sanjay and Walia, Gurjit Singh and Kumar, Sumit},
  journal={Preprint submitted to Springer},
  year={2025},
  note={Under Review}
}
```
