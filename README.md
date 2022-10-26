# SourceFilterHiFiGAN
Official Implementation of Source-Filter HiFiGAN (SiFiGAN)

This repo provides official PyTorch implementation of [SiFiGAN](), a fast and pitch controllable high-fidelity neural vocoder.<br>
For more information, please see our [demo](https://chomeyama.github.io/SiFiGAN-Demo/).

## Environment setup

```bash
$ cd SourceFilterHiFiGAN
$ pip install -e .
```

Please refer to the [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) repo for more details.

## Folder architecture
- **egs**:
The folder for projects.
- **egs/namine_ritsu**:
The folder of the [Namine Ritsu](https://www.youtube.com/watch?v=pKeo9IE_L1I) project example.
- **sifigan**:
The folder of the source codes.

The dataset preparation of Namine Ritsu database is based on [NNSVS](https://github.com/nnsvs/nnsvs/).
Please refer to it for the procedure and details.

## Run

In this repo, hyperparameters are managed using [Hydra](https://hydra.cc/docs/intro/).<br>
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation

Make dataset and scp files denoting paths to each audio files according to your own dataset (E.g., `egs/namine_ritsu/data/scp/namine_ritsu.scp`).<br>
List files denoting paths to the extracted features are automatically created in the next step (E.g., `egs/namine_ritsu/data/scp/namine_ritsu.list`).<br>
Note that scp/list files for training/validation/evaluation are needed.

### Preprocessing

```bash
# Move to the project directory
$ cd egs/namine_ritsu

# Extract acoustic features (F0, mel-cepstrum, and etc.)
# You can customize parameters according to sifigan/bin/config/extract_features.yaml
$ sifigan-extract-features audio=data/scp/namine_ritsu_all.scp

# Compute statistics of training data
$ sifigan-compute-statistics feats=data/scp/namine_ritsu_train.list stats=data/stats/namine_ritsu_train.joblib
```

### Training

```bash
# Train a model customizing the hyperparameters as you like
$ sifigan-train generator=sifigan discriminator=univnet train=sifigan data=namine_ritsu out_dir=exp/sifigan
```

### Inference

```bash
# Decode with several F0 scaling factors
$ sifigan-decode out_dir=exp/sifigan checkpoint_steps=400000 f0_factors=[0.5,1.0,2.0]
```

### Monitor training progress

```bash
$ tensorboard --logdir exp
```

## Citation
If you find the code is helpful, please cite the following article.

```

```

## Authors

Development:
[Reo Yoneyama](https://chomeyama.github.io/Profile/) @ Nagoya University, Japan<br>
E-mail: `yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp`

Advisors:<br>
[Yi-Chiao Wu](https://bigpon.github.io/) @ Meta Reality Labs Research, USA<br>
E-mail: `yichiaowu@fb.com`<br>
[Tomoki Toda](https://sites.google.com/site/tomokitoda/) @ Nagoya University, Japan<br>
E-mail: `tomoki@icts.nagoya-u.ac.jp`
