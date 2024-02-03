This repository contains the code used for the experiments in the paper [	Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking].

We study how fine-tuning affects the internal mechanisms implemented in language models. As a case study, we explore the property of entity tracking in Llama-7B, and in its fine-tuned variants - Goat-7B and Vicuna-7B.
Our findings suggest that fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the model.

This repository provides an implementation of the main two methods we use to support our findings:  
(i) DCM, which automatically detects model components responsible for specific semantics.
(ii) CMAP, a new approach for patching activations across models to reveal improved mechanisms. 

## Setup

To get all the dependencies run:
```bash
conda env create -f environment.yaml
```
## How to cite
