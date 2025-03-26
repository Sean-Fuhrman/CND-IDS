#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
'''
Experiment options:

*Feature Extractor: None, "ADCN", "AE", "EaM"
default: None, implemented in main

*anomaly Score Method: "PCA", "AE", "KMeans", "LOF", "OCSVM", "IF"
default: "PCA". implemented in main

*Metrics: "trivial val 3 Std (F1)", "trivial val 2 Std (F1)",  "best f1 (F1)", "roc auc", "pr auc"
default: ["best f1 (F1)"], implemented in get_metrics_scores of metrics

Normalize: True /False
default: True

*Dataset: "XIIoT", "EdgeIIoT", "WUST", "MQTT", "UNSW", "CICIDS17", "CICIDS18"
No default val, implemented in utils

num_experiments: int
default: 1

load_model: True / False   ---- If true will attempt to load model, instead of training it (Currently implemented for: ADCN) 
default: False

train_epochs: int
default: 10 -- applies to both ADCN and AE

ADCN_label_mode: "random", ---- More options in datastream

* indicates it can be an array, for multiple experiments in one run
'''

experiment = {
    'feature_extractor': ["CND_IDS"],
    'anomaly_score_method': ["PCA"],
    'metrics': ["trivial val 3 Std (F1)", "trivial val 2 Std (F1)",  "trivial val 1 Std (F1)", "best f1 (F1)", "roc auc", "pr auc"],
    'dataset': ["UNSW","XIIoT","WUST", "CICIDS17"],
    'num_experiments': 3,
    'train_epochs': 10,
    'normalize': True,
    'batch_size': 64,
    }

import logger_config
logger_config.init_logs(experiment)

import main
import logging
import traceback

logger = logging.getLogger()

import torch

if torch.torch.cuda.is_available():
    device = "cuda" 
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("time_results", exist_ok=True)

logger.info("Using Device: %s", device)
try:
    main.run_experiments(experiment, device)
except Exception as e:
    logger.error(traceback.format_exc())


#
# %%
