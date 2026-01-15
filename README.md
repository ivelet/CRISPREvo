# CRIPSR-Evo: Biological Foundation Model for CRISPR Array Detection

## Table of contents
* [Installation and Environment](#installation)
* [Download and Prepare Data](#prepare-data)
* [Reproduce Results](#reproduce-results)
* [Binary Classifier](#binary-classifier)
    * [Train (fine-tune) Evo on CRISPR array data](#binary-train)
    * [Predict CRISPR arryas](#binary-predict)
    * [Evaluate the results](#binary-evaluate)
* [Multi-Class Classifier](#multi-classifier)
    * [Train (fine-tune) Evo on CRISPR array data](#multi-train)
    * [Predict CRISPR arryas](#multi-predict)
    * [Evaluate the results](#multi-evaluate)


## Installation and Environment
Note that Evo requires flash-attn, which must be installed on an Ampere, Ada, or Hopper GPU (e.g., A100, RTX 3090, RTX 4090, H100).
Also, due to known issues with the installation of Flash Attention, it is recommended to install PyTorch and Flash Attention first before other packages in the environment.
If there are issues installing the pip packages when setting up the conda environment from environemnt.yml, activate the conda environment and then install them separately using pip while inside the environment.
```bash
conda env create -f environment/environment.yml
conda activate crispr-evo
pip install -r environment/requirements.txt
```

## Download and Prepare Data
```bash 
bash evo/scripts/download_all_datasets.sh
bash evo/scripts/prepare_all_data.sh
```

## Reproduce Results
CRISPR-evo was fine-tuned on a single H200. To reproduce any results from the manuscript, as well as to run any additional inference and evaluation on our model without needing to run training again, we provide the fine-tuned model.

To reproduce all results using the provided fine-tuned model checkpoint, run the following:

```bash
bash evo/scripts/reproduce_inference_results.sh
```

To reproduce all results from scratch including fine-tuning, run the following:

```bash 
bash evo/scripts/reproduce_all_results.sh
```

## Binary Classifier 
### Train (fine-tune) Evo on CRISPR array data
```bash 
bash evo/scripts/bin/train_job.sh
```
### Predict CRISPR arryas
```bash 
bash evo/scripts/bin/infer_job.sh
```
### Evaluate the results
```bash 
bash evo/scripts/bin/eval_job.sh
```
## Multi-Class Classifier 
### Train (fine-tune) Evo on CRISPR array data

```bash 
bash evo/scripts/multi/train_job.sh
```
### Predict CRISPR arryas
```bash 
bash evo/scripts/multi/infer_job.sh
```
### Evaluate the results
```bash 
bash evo/scripts/multi/eval_job.sh
```

