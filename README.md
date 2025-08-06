# BODIES_EXPERIMENTS

A Python framework for 3D human‐body reconstruction and avatar generation using the SUPR model. This repository bundles data extraction/preprocessing, model training/evaluation, texture mapping, and a simple GUI to capture photos and generate personalized 3D avatars.

---

## Table of Contents

- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Preparing the SUPR Model](#preparing-the-SUPR-model)  
- [Usage](#usage)  
  - [1. Data Extraction & Preprocessing](#1-data-extraction--preprocessing)  
  - [2. Model Training](#2-model-training)  
  - [3. Evaluation & Testing](#3-evaluation--testing)  
  - [4. Texture Mapping](#4-texture-mapping)  
  - [5. GUI Avatar Generator](#5-gui-avatar-generator)  
- [Configuration](#configuration)  
- [Acknowledgements](#acknowledgements)  
- [License](#license)  
- [Author](#author)  

---

## Features

- **Dataset extraction**: Scripts and utilities for building and preprocessing an SUPR‐based body dataset.  
- **Model training**: End‐to‐end PyTorch training pipeline (`trainer_full.py`) and configurable loss functions.  
- **Evaluation**: Quantitative metrics for shape & measurement accuracy (`evaluator.py`, `measurement_evaluator.py`).  
- **Texture mapping**: Simple pipeline for sampling and applying skin/texture overlays (`texture.py`).  
- **Interactive GUI**: PySimpleGUI front end (`GUI.py`) to capture front/side photos and generate a 3D avatar.  
- **SUPR integration**: Utilities to load and preprocess SUPR parameters for mesh reconstruction.  

---

## Repository Structure

```
.
├── GUI.py                  # PySimpleGUI app: capture images → generate avatar
├── demo.py                 # Example/demo script tying together modules
├── extract_dataset.py      # Ingest raw images/annotations; export SUPR input data
├── trainer_full.py         # Full training pipeline (data loader, model, losses, optimizer)
├── evaluator.py            # Compute reconstruction errors vs. ground truth
├── measurement_evaluator.py# Evaluate anthropometric measurements (height, limb lengths, etc.)
├── texture.py              # Sample and map textures onto reconstructed meshes
├── obj_test.py             # Quick sanity checks on .obj exports
├── test_full.py            # End‐to‐end pipeline tests
├── utils/                  # Utility modules
│   ├── preprocess_SUPR.py  # SUPR‐specific preprocessing (pose & shape parameter handling)
│   ├── torchloader.py      # PyTorch dataset & DataLoader wrappers
│   ├── image_utils.py      # Helpers for image I/O, resizing, normalization
│   ├── model.py            # PyTorch model definitions (encoder, decoder, etc.)
│   ├── losses.py           # Custom loss functions (L2, silhouette, etc.)
│   ├── measures_.py        # Shape & pose evaluation metrics
│   ├── resize.py           # Command‐line tool to resize/crop input photos
│   ├── gpu_test.py         # Quick GPU availability check
│   └── …                    # Other supporting modules
└── LICENSE                 # Apache 2.0
```

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/FrancescoManigrass/BODIES_EXPERIMENTS.git
   cd BODIES_EXPERIMENTS
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate     # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install \
     numpy \
     torch torchvision \
     opencv-python \
     PySimpleGUI \
     scipy \
     trimesh \
     pillow \
     SUPRx
   ```
   > **Note:** adjust versions as needed for your CUDA/PyTorch setup.

---

## Preparing the SUPR Model

1. Download the [SUPR model files](https://SUPR.is.tue.mpg.de/) (e.g. `SUPR_NEUTRAL.pkl`) under a directory of your choice (e.g. `models/SUPR`).
2. In any script that loads SUPR, set the environment variable:
   ```bash
   export SUPR_MODEL_DIR=/path/to/models/SUPR
   ```
   Or modify the path directly in `utils/preprocess_SUPR.py`.

---

## Usage

### 1. Data Extraction & Preprocessing
```bash
python extract_dataset.py \
  --input_dir path/to/raw/images \
  --output_dir path/to/preprocessed \
  --annotations path/to/annotations.json
```

### 2. Model Training
```bash
python trainer_full.py \
  --data_dir path/to/preprocessed \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-4 \
  --save_dir checkpoints/
```

### 3. Evaluation & Testing
```bash
# Quantitative evaluation
python evaluator.py \
  --checkpoint_path checkpoints/latest.pth \
  --test_data path/to/preprocessed

# Measurement accuracy
python measurement_evaluator.py \
  --checkpoint checkpoints/latest.pth \
  --test_data path/to/preprocessed
```

### 4. Texture Mapping
```bash
python texture.py \
  --mesh_dir outputs/meshes \
  --image_dir path/to/input/photos \
  --output_dir outputs/textured_meshes
```

### 5. GUI Avatar Generator
1. Edit the top of `GUI.py` to set:
   ```python
   front_path = "front.jpg"
   side_path  = "side.jpg"
   icon        = "/path/to/app_icon.ico"
   ```
2. Run:
   ```bash
   python GUI.py
   ```
3. Enter your **Name**, **Height**, **Weight**, **Sex**, click **“Scatta Foto”**, then **“Genera Avatar”**.

---

## Configuration

- **Paths:** Modify `front_path`, `side_path`, and `icon` in `GUI.py` to point to your desired files.
- **Hyperparameters:** Tweak learning rate, batch size, loss weights, etc. in `trainer_full.py`.
- **SUPR settings:** Adjust pose/shape parameter ranges in `utils/preprocess_SUPR.py`.

---

## Acknowledgements

This code is based on the methods described in the paper:
```
Pan, Hu, Zhang, et al. "HMMR: Hierarchical Motion Modeling for Humans in the Wild." arXiv preprint arXiv:2205.14347 (2022).
```
See full paper: https://arxiv.org/pdf/2205.14347

---

This code also incorporates methods from the paper:
```
@inproceedings{thota2022estimation,
  title={Estimation of 3D body shape and clothing measurements from frontal-and side-view images},
  author={Thota, Kundan Sai Prabhu and Suh, Sungho and Zhou, Bo and Lukowicz, Paul},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={2631--2635},
  year={2022},
  organization={IEEE}
}
```

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

---

## Author

**Francesco Manigrass**  
– If you use this code, please cite or star the repo!
