# CND-IDS: Continual Novelty Detection for Intrusion Detection Systems

This repo contains the code for the paper:

Sean Fuhrman, Onat Gungor, Tajana Rosing. "CND-IDS: Continual Novelty Detection for Intrusion Detection Systems" in DAC 2025. 

[arXiv link](https://arxiv.org/abs/2502.14094)

## Prerequisites

Install required packages: `pip install -r requirements.txt`

### Datasets

Create a `data` directory and add datasets as subdirectories. Each dataset should contain `x.npy` and `y.npy` files for features and labels, respectively.
`y.npy` should match the labels specified in utils.py's dictionaries. More details on how the datasets are loaded can be found in utils.py.

## Running the Code

you can launch experiments with:
```bash
python run_experiments.py
```
This script allows you to customize configuration options such as:
- Feature extractor 
- Anomaly scoring method 
- Dataset 
- Training parameters like epochs, batch size, number of experiments

You can modify these settings in the `experiment` dictionary at the top of `run_experiments.py`.

To replicate CND-IDS, you can use the following configurations:
``` 
 {
    "feature_extractor": ["CND_IDS"],
    "anomaly_scoring": ["PCA"],
    ...
 }
```
