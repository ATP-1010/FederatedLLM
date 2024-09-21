# FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations
Code of paper : [FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations](https://arxiv.org/pdf/2409.05976).

You can use this code to fine-tune LLMs with LoRA by datasets.
The LoRA fine-tuning method includes FLoRA, FedIT, and Zero-Padding. You can also use heterogeneous LoRA rank setting in FLoRA and Zero-Padding.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Wizard and MMLU.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
* To run the FLoRA algorithms with global/local schedulers, run:
```

```
To run NON-IID settings, the iid parameter should be 0.
FedHyper-FULL can run both global and local schedulers

* To run the baselines in our paper, here are some examples:
```

```
-----


sh run_wiz.sh
