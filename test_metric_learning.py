#%%
import AnomolyDetectors.PCA as PCA
import FeatureExtractors.CFE as CFE
import FeatureExtractors.CNDIDS.sampler
import FeatureExtractors.CNDIDS.K_Means
import numpy as np
import torch
import utils

import logging

logging.basicConfig(level=logging.INFO)
datastream = utils.get_benchmark("XIIoT")
experience_num = 0
device = "cuda"
def pca_test(X_normal, X_anomalous, init_normal, init_val):
    pca = PCA.PCA_model()
    pad = []
    pnm = []
    pad3 = []
    pnm3 = []
    for i in range(10):
        #Randomize the data
        torch.manual_seed(i)
        np.random.seed(i)
        X_normal = X_normal[torch.randperm(X_normal.size(0))][:40000]
        X_anomalous = X_anomalous[torch.randperm(X_anomalous.size(0))][:40000]
        
        pca.fit(init_normal)
        val_score = pca.predict(init_val)

        normal_scores = pca.predict(X_normal)
        anomalous_scores = pca.predict(X_anomalous)

        second_std = np.percentile(val_score, 95)
        percent_anomalous_detected = np.sum(anomalous_scores > second_std) / len(anomalous_scores)
        percent_normal_missed = np.sum(normal_scores > second_std) / len(normal_scores)
        pad.append(percent_anomalous_detected)
        pnm.append(percent_normal_missed)
        
        third_std = np.percentile(val_score, 99)
        percent_anomalous_detected = np.sum(anomalous_scores > third_std) / len(anomalous_scores)
        percent_normal_missed = np.sum(normal_scores > third_std) / len(normal_scores)
        pad3.append(percent_anomalous_detected)
        pnm3.append(percent_normal_missed)
                    
    pad = np.array(pad)
    pnm = np.array(pnm)
    print("2nd Percent Anomalous Detected: ", pad.mean())
    print("2nd Percent Normal Missed: ", pnm.mean())
    
    pad3 = np.array(pad3)
    pnm3 = np.array(pnm3)
    print("3rd Percent Anomalous Detected: ", pad3.mean())
    print("3rd Percent Normal Missed: ", pnm3.mean())

def test_extractor(extractor, cheating=False):
    X = datastream.train_experiences[experience_num][0]


    if cheating:
        X_normal = X[datastream.train_experiences[experience_num][1] == 0]
        X_anomalous = X[datastream.train_experiences[experience_num][1] == 1]
        extractor.fit(X_normal, X_anomalous, device)
    else:
        extractor.fit(X, device)
    
    init_normal = datastream.init_normal
    init_val = datastream.init_val
    X_test = datastream.test_experiences[experience_num][0]
    y = datastream.test_experiences[experience_num][1]
    X_normal = X_test[y == 0]
    X_anomalous = X_test[y == 1]
    
    X_normal = extractor(X_normal).detach().cpu()
    print(X_normal)
    X_anomalous = extractor(X_anomalous).detach().cpu()
    print(X_anomalous)
    init_normal = extractor(init_normal).detach().cpu()
    init_val = extractor(init_val).detach().cpu()

    pca_test(X_normal, X_anomalous, init_normal, init_val)
#%%

from FeatureExtractors import Met2

extractor = Met2.Met2(datastream)
test_extractor(extractor) 

#%%

from FeatureExtractors import Met_Con

extractor = Met_Con.Met_Con(nFeatures=datastream.nInput, nLatent=96)
test_extractor(extractor) 

#%%

from FeatureExtractors import Met

extractor = Met.Met(nFeatures=datastream.nInput, nLatent=30)
#With Cheating (knowing x_normal and x_anomalous) in training
test_extractor(extractor, cheating=True) 
# %%


from FeatureExtractors import PassThroughExtractor

extractor = PassThroughExtractor.PassThroughExtractor(datastream, {"batch_size": 100, "memory_mode": None})

test_extractor(extractor) #0.51, 0.05
#%%
from FeatureExtractors import CFE

extractor = CFE.CFE(datastream, {"batch_size": 100, "memory_mode": None}, device)

test_extractor(extractor) #0.4876680616151724 , 0.047352107102816066
#%%