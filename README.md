## Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking

This repository contains the code used for the experiments in the paper [Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking](https://openreview.net/forum?id=8sKcAWOf2D).

We study how fine-tuning affects the internal mechanisms implemented in language models. As a case study, we explore the task of entity tracking in Llama-7B, and in its fine-tuned variants - Vicuna-7B, Goat-7B, Float-7B.

<p align="center">
<img src="circuit_flow.png" style="width:80%;"/>
</p>

**Our findings suggest that fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the model.**

Please check [finetuning.baulab.info](finetuning.baulab.info) for more information.

## Methods
In order to discover the underlying mechanism for performing entity tracking task, we employed: 1) [Path Patching](https://arxiv.org/abs/2211.00593) ([experiment_1/path_patching.py](experiment_1/path_patching.py)) and 2) [Desiderata-based Component Masking](https://arxiv.org/abs/2307.03637) ([experiment_2/DCM.py](experiment_2/DCM.py)). Both the methods are implemented using [baukit](baukit/nethook.py), which can be easily adopted for other tasks.


Moreover, in order to uncover the reason behind the performance enhancement in fine-tuned models employing the same mechanism, we have introduced a novel approach called CMAP (Cross-Model Activation Patching). This method involves patching activations across models to elucidate the enhanced mechanisms. The notebook [experiment_3/cmap.ipynb](experiment_3/cmap.ipynb) provides a demonstration on how to execute the complete experiment.

<p align="center">
<img src="cmap_vis.png" style="width:80%;"/>
</p>

## Setup

To get all the dependencies run:
```bash
conda env create -f environment.yaml
```
## How to cite
