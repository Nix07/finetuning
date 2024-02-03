This repository contains the code used for the experiments in the paper [Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking]().

We study how fine-tuning affects the internal mechanisms implemented in language models. As a case study, we explore the property of entity tracking in Llama-7B, and in its fine-tuned variants - Goat-7B and Vicuna-7B.
Our findings suggest that fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the model.

 ## DCM
[experiment_2/DCM.py](experiment_2/DCM.py), An automated method for identifying the components of a model responsible for specific semantic properties.

## CMAP
[experiment_3/cmap_utils.py](experiment_3/cmap_utils.py), A new approach for patching activations across models to reveal improved mechanisms. 
<p align="center">
<img src="cmap_vis.png" style="width:80%;"/>
</p>

## Setup

To get all the dependencies run:
```bash
conda env create -f environment.yaml
```
## How to cite
