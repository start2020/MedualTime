# MedualTime: A Dual-Adapter  Language Model for Time Series Multimodal Representation Learning

This code is the official implementation of MedualTime.

## Installation

```
pip install -r requirements.txt
```

## Datasets and Pre-trained LM

Please download the experimental datasets and pre-trained GPT-2 parameters via the following link: https://figshare.com/articles/dataset/Supplementary_Dataset_for_Experiments/27108928

Please put the experimental datasets into the `datasets` folder. The GPT-2 parameters should be placed in `gpt2_pretrained` folder.

## Experiment


For supervised learning experiments, few-shot label transfer experiments, and unsupervised representation learning experiments on PTB-XL dataset, please run the followings:
```
sh supervised_exp.sh 
sh transfer_exp.sh 
sh unsupervised_exp.sh 
```

