# Modeling the Selective and Lossy Nature of Visual Processing Bridges the Discrepancy in Non-Invasive Neural Decoding

## Table of Contents
- [Introduction](#introduction)
- [Repo Architecture](#repo-architecture)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Run](#run)
- [Model Architecture](#model-architecture)
- [Acknowledgement](#acknowledgement)

## Introduction

This is the official implementation for **Modeling the Selective and Lossy Nature of Visual Processing Bridges the Discrepancy in Non-Invasive Neural Decoding**. The manuscript will be accepted for publication in International Journal of Neural Systems 2026. If you find our paper and code interesting or useful, please cite our paper. Contact E-mail: sculyp02@163.com.

Our work studies neural decoding from the perspective that human visual processing is inherently **selective and lossy**. Based on this motivation, the framework introduces an attention-weighted multi-level visual degradation strategy, contextual visual calibration, and a stability-preserving adaptive EEG encoder to improve the alignment between non-invasive neural signals and visual representations.

The implementation is developed based on the **Uncertainty-aware Blur Prior (UBP)** framework, which serves as the main baseline of this work.


## Repo Architecture

```text
SLVP/                                 
├── README.md
├── base                              
│   ├── data_eeg.py    
│   ├── data_meg.py  
│   ├── data_eegcvpr40.py                 
│   ├── eeg_backbone.py                
│   ├── inpating_data.py              
│   └── utils.py                       
├── configs
│   └── eeg
│       └── baseline.yaml
│       └── slvp.yaml
│   └── meg
│       └── baseline.yaml
│       └── slvp.yaml
│   └── eegcvpr40
│       └── baseline.yaml
│       └── slvp.yaml
│                                      
├── data                              
│   ├── things-eeg
│   ├── things-meg
│   └── eegcvpr40                              
├── main.py                                                   
├── scripts                                                   
├── supplementary                      
├── docs
│   └── model_architecture.md          
└── requirements.txt                   
```

## Environment Setup

The main experiments were conducted with:

- Python 3.10.19
- CUDA 12.1
- PyTorch 2.4.1
- cuDNN 9.1.0

Required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The computational-efficiency benchmark reported in the paper was measured under WSL2 using an NVIDIA GeForce RTX 4090 D 24 GB GPU.

## Data Preparation

### 1. Neural decoding datasets

Please download the datasets from their official repositories:

- **THINGS-EEG**: [OSF repository](https://osf.io/anp5v/files/osfstorage)
- **THINGS images**: [OSF repository](https://osf.io/jum2f/files/osfstorage)
- **THINGS-MEG**: [OpenNeuro ds004212](https://openneuro.org/datasets/ds004212/versions/2.0.1)
- **EEGCVPR40**: [official repository](https://github.com/perceivelab/eeg_visual_classification)

We have made all processed data publicly available on Baidu Netdisk:(https://pan.baidu.com/s/1ETjpWX1cIuabKfvpGSyX3g?pwd=cj3b)

For THINGS-EEG, the main experiments use 17 posterior channels and 250 time samples:

```text
Input shape: [B, 17, 250]

Channels:
P7, P5, P3, P1, Pz, P2, P4, P6, P8,
PO7, PO3, POz, PO4, PO8, O1, Oz, O2
```

After preprocessing, the expected directory layout is:

```text
data/
├── things-eeg/
│   ├── Image_feature/
│   ├── Image_set/
│   ├── Image_set_Resize/
│   └── Preprocessed_data_250Hz_whiten/
├── things-meg/
│   ├── Image_feature/
│   ├── Image_set/
│   ├── Image_set_Resize/
│   └── Preprocessed_data_250Hz_whiten/
└── eegcvpr40/
│   ├── Image_feature/
│   ├── Raw_parquet/
│   ├── Image_set_Resize/
│   └── Preprocessed_data_1000Hz/
```

## Run

### THINGS-EEG

A representative intra-subject experiment can be run with:

```bash
python main.py \
  --config configs/eeg/slvp.yaml \
  --dataset eeg \
  --subjects sub-01 \
  --seed 208000 \
  --exp_setting intra-subject \
  --brain_backbone EEGProjectLayerLiteAdapterSafeV32 \
  --vision_backbone RN50 \
  --epoch 50 \
  --lr 1e-4
```

The main optimization settings are:

```text
Optimizer:      AdamW
Learning rate:  1e-4
Weight decay:   0.01
Batch size:     1024
Epochs:         50
```

Run the corresponding scripts under `scripts/` for the complete multi-subject experiments.

## Model Architecture

The proposed **Stability-Preserving Adaptive EEG Encoder** is implemented as:

```python
EEGProjectLayerLiteAdapterSafeV32
```

For THINGS-EEG, it maps:

```text
[B, 17, 250] -> [B, 1024]
```

Its main components are:

```text
Input channel scaling
        ↓
Temporal calibration MLP
        ↓
Core projection: 4250 -> 1024
        ↓
Residual block: 1024 -> 1024
        ↓
LayerNorm
        ↓
Bounded residual adapter: 1024 -> 128 -> 1024
        ↓
Sample-wise residual gate: 1024 -> 128 -> 1
        ↓
Feature reweighting: 1024 -> 256 -> 1024
```

The THINGS-EEG configuration contains approximately **6.341 M parameters**.

A detailed layer-by-layer description, including hidden dimensions, initialization strategy, and forward equations, is provided in:

[`docs/model_architecture.md`](docs/model_architecture.md)


## Acknowledgement

This work builds on and extends the **Uncertainty-aware Blur Prior (UBP)** framework:

- [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior) [CVPR 2025]

We also acknowledge the following datasets and prior neural-decoding works:

- [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758) [THINGS-EEG]

- [THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior](https://pubmed.ncbi.nlm.nih.gov/36847339/) [THINGS-MEG]

- [Deep Learning Human Mind for Automated Visual Classification](https://github.com/perceivelab/eeg_visual_classification) [EEGCVPR40, CVPR 2017]

- [Decoding Natural Images from EEG for Object Recognition](https://github.com/eeyhsong/NICE-EEG) [ICLR 2024]



