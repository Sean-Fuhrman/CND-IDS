#%%
import utils
from AnomolyDetectors import AE, PCA
from AutonomousDCN import ADCNbasic, ADCNmainloop
from AutonomousDCN.model import simpleMPL
import torch
import pickle as pkl
import pandas as pd
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging

logging.basicConfig(level=logging.INFO)

#%%
strategy = "AE"
dataset = "WUST"
percent_val = 0.01
datastream = utils.get_benchmark(dataset)

normal_label = utils.get_normal_attack(dataset)

nNodeInit  = 96
nOutput   = 500
nHidden   = 1000
epoch = 1
latent_dim = 64

ADCNnet = ADCNbasic.ADCN(datastream.nOutput, nInput=nOutput, nHiddenNode = nNodeInit, desiredLabels=[0,1])
ADCNnet.ADCNcnn = simpleMPL(datastream.nInput, nHidden, nOutput)

df = pd.DataFrame(columns=['Novelty Strategy', 'Train Experience', 'Test Experience', 'Accuracy', 'F1', 'Threshold Method', 'Threshold'])

for i in range(datastream.nExperiences):
    #load with pickle
    with open(f'./models/ADCN_{dataset}_{i}.pkl', 'rb') as f:
        ADCNnet = pkl.load(f)
    

    init_val = datastream.init_normal[:int(len(datastream.init_normal)*percent_val)]
    init_normal = datastream.init_normal[int(len(datastream.init_normal)*percent_val):]
    encoded_x = ADCNnet.encode(init_normal)
    encoded_val = ADCNnet.encode(init_val).cpu().detach().numpy()

    novelty_model = AE.AE(nFeatures=encoded_x.shape[1], nLatent=latent_dim).to('cuda')
    novelty_model.fit(encoded_x, epochs=20, device='cuda')

    # novelty_model=PCA_model.PCA_model()
    # novelty_model.fit(encoded_x.cpu().detach().numpy())
    

    thresholding_methods = {
        'top k': utils.top k,
        'trivial_percentile_val': utils.trivial_percentile_val,
        'best_f_score': utils.best_f_score
    }

    #Get ROC AUC for each experience
    import numpy as np
    for j,experience in enumerate(datastream.test_experiences):
        X, Y = experience
        Y = Y.cpu().detach().numpy()
        encoded_x = ADCNnet.encode(X).cpu().detach().numpy()
        if isinstance(novelty_model, AE.AE):
            encoded_x = torch.tensor(encoded_x).to('cuda')
            encoded_val = torch.tensor(encoded_val).to('cuda')
        predictions = novelty_model.predict(encoded_x)
        val_predictions = novelty_model.predict(encoded_val)
        contamination = 1- ((Y == normal_label).sum() / len(Y))
        if isinstance(novelty_model, AE.AE):
            predictions = predictions.cpu().detach().numpy()
            val_predictions = val_predictions.cpu().detach().numpy()

        for method in thresholding_methods:
            if method == 'top k':
                t = thresholding_methods[method](predictions, contamination)
            elif method == 'trivial_percentile_val':
                t = thresholding_methods[method](val_predictions, 99.7)
            elif method == 'best_f_score':
                t = thresholding_methods[method](predictions, Y)

            labels = (predictions > t).astype(int)
            f1 = utils.f1_score(Y, labels)
            acc = (labels == Y).sum() / len(Y)
            df = df.append({'Novelty Strategy': strategy, 'Train Experience': i, 'Test Experience': j, 'Accuracy': acc, 'F1': f1, 'Threshold Method' : method, 'Threshold': t}, ignore_index=True)

df.to_csv(F'./results/strategy=ADCN-{strategy}_dataset={dataset}.csv')
# %%

