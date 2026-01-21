# CRIPSR-Evo: Biological Foundation Model for CRISPR Array Detection Without Metagenomic Assembly

Accurate identification of CRISPR arrays is essential for studying prokaryotic adaptive immunity, yet existing tools struggle with short-read sequencing
data and arrays containing degenerate repeats. These limitations restrict CRISPR analysis in metagenomic and fragmented genomic datasets. We
present a foundation model-based approach for CRISPR array detection that addresses both these challenges. We fine-tune a large genomic foundation
model using the Parameter-Efficient Fine-Tuning (PEFT) method, Low-Rank Adaptation (LoRA) to perform per-nucleotide classification of DNA
sequences into repeat, spacer, and non-array regions directly from raw input nucleotide sequences. We develop two model variants for different
sequence context lengths. The long-context model supporting sequences of up to 8,192 nucleotides achieves 98.16% test accuracy and identifies
degenerate repeat candidates missed by similarity-based CRISPR detection tools, with 92.5% of candidates aligning significantly to their array
consensus repeats. The short-context model supports sequences of up to 150 nucleotides, optimized for Illumina reads, reaches 90.03% accuracy
and enables direct analysis of individual reads without assembly. On simulated metagenomic data, it achieves a spacer recall of 49.12% and recovers
12.57% of spacers that are otherwise not detected by dedicated metagenomic CRISPR array detection methods which require metagenomic assembly.
Together, these results demonstrate that genomic foundation models provide a robust and complementary paradigm for CRISPR array detection.

## Table of contents
* [Installation and Environment](#installation)
* [Download and Prepare Data](#prepare-data)
* [Reproduce Results](#reproduce-results)
* [CRISPR Cas Bona Fide Array](#bona-fide)
    * [Train (fine-tune) Evo on CRISPR array data](train)
    * [Predict CRISPR arryas](#predict)
    * [Evaluate the results](#mevaluate)
* [Metagenomic Analysis](#meta)


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

### Download all datasets
```bash 
wget
unzip data
```

### Download fine-tuned model
To save time, you may use the following to download the fine-tuned model used in the manuscript to avoid fine-tuning an Evo model from scratch.
```bash
wget 
unzip models
```
### Prepare all data
```bash 
python3 evo/src/prepare_data.py
```

<!-- ## Reproduce Results
CRISPR-evo was fine-tuned on a single H200. To reproduce any results from the manuscript, as well as to run any additional inference and evaluation on our model without needing to run training again, we provide the fine-tuned model.

To reproduce all results using the provided fine-tuned model checkpoint, run the following:

```bash
bash evo/scripts/reproduce_inference_results.sh
```

To reproduce all results from scratch including fine-tuning, run the following:

```bash 
bash evo/scripts/reproduce_all_results.sh -->
```

## CRISPR Cas Bona Fide Array Prediction

### Train (fine-tune) Evo on CRISPR array data
Use the following to download a pre-trained Evo model and then fine-tune it using LoRA on bona-fide sequences. Note that this was done on a single H200 GPU for >3 hours. To skip this step and save time on fine-tuning, you may run inference on the fine-tuned model downloaded.
```bash 
python3 evo/src/multi/train.py
```
### Predict CRISPR arryas
Run inference on the fine-tuned CRISPREvo model and evaluate the results on the bona-fide test sequences
```bash 
python3 evo/src/multi/infer.py $1 $2
python3 evo/src/multi/eval_metrics.py $1
```
## Metagenomic Analysis 
CRISPREvo can perform metagenomic analysis without the need for assembly. After downloading the fine-tuned model or fine-tuning it from scratch, run the following to infer the model on simulated metagenomic reads. Note that there are 10 million sequences in total which can take a long time to run. You may specify the number of sequences to randomly sample from this dataset as done below to run inference on 100,000 randomly selected sequences.

```bash
python3 evo/src/pred_meta_sim_small_reads.py
```

