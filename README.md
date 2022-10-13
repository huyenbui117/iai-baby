<div align="center">

# Nuchal transluecency segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This project aims to provide segmentation model for NT segmentation problem

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/CaoHoangTung/iai-baby
cd iai-bayb

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Available experiments
Run each experiment by passing the `experiment` config in the train or eval script
```bash
python src/train.py experiment=[Experiment name]
python src/eval.py experiment=[Experiment name] ckpt_path=[Path to model checkpoint]
```

| Experiment name   |      Description      |  IOU |
|----------|-------------|------:|
| official_resnet_segment |  Simple segmentation with unet+resnet50. | 0.25* |
| official_resnet_segment_with_headmask |    Segmentation with unet+resnet50. Use an additional input channel for the mask of the head region.   |   0.29* |
| official_resnet_segment_localized | Simple segmentation with unet+resnet50. Use code to localize the segmentation region and run the segmentation training |    0.66 (local) |
| official_effnet-b8_segment | Segmentation with unet+effecientnet-b8. |    0.42 |
| official_effnet-b8_segment_localized | Segmentation with unet+effecientnet-b8. Use code to localize the segmentation region and run the segmentation training |    0.64 (local) |
| official_effnet-b8_onlyhead_segment | Segmentation with unet+effecientnet-b8. Only use images that focus on the head region |    0.47* |
| official_effnet-b8_onlybody_segment | Segmentation with unet+effecientnet-b8. Only use images that focus on the body region |    0.36* |
| eval_fromyolo_resnet | Evaluate localized segmentation using unet+resnet. Use bounding box prediction from yolov7 as the input for the segmentation model |    0.53* |
| eval_fromyolo_effnet-b8 | Evaluate localized segmentation using unet+effnet-b8. Use bounding box prediction from yolov7 as the input for the segmentation model |    0.18* |

<b>\* Need re-evaluation</b>
