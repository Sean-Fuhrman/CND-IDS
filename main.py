#%%
import torch
import utils
import logging
import pandas as pd
import numpy as np
import random
import pickle as pkl
import metrics 
from functools import partial
from AutonomousDCN import ADCNbasic, ADCNmainloop
from AutonomousDCN.model import simpleMPL
from FeatureExtractors.PassThroughExtractor import PassThroughExtractor
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger()

def run_experiments(experiment, device):
    #Array formatted settings: feature_extractor, anomaly_score_method, dataset
    logger.info("Beginning experiment with settings: %s", experiment)
    for dataset in experiment['dataset']:
        for f_e in experiment['feature_extractor']:
            for a_s_m in experiment['anomaly_score_method']:
                if f_e != "CFE":
                    memory_modes = [None]
                else:
                    memory_modes = experiment['memory_mode']
                for memory_mode in memory_modes:
                    if f_e == None and a_s_m == None:
                        continue
                    config = extract_config(experiment, f_e, a_s_m, dataset, memory_mode)

                    columns = ['Experiment Number', 'Train Experience', 'Test Experience']
                    results_df = pd.DataFrame(columns=columns)
                    logger.info("Created results df with columns: %s", columns)

                    num_experiments = config['num_experiments']


                    for i in range(num_experiments):
                        #Set seed for reproducibility
                        random.seed(i)
                        np.random.seed(i)
                        torch.manual_seed(i)
                        logger.info("Loading dataset: %s", dataset)
                        datastream = utils.get_benchmark(dataset, normalize=experiment['normalize'])
                        utils.log_datastream(config['dataset'], datastream)
                        logger.info("Beginning experiment %d/%d", i+1, num_experiments)
                        df = run_single_experiment(config, datastream, device)
                        #Add experiment number to results
                        df['Experiment Number'] = i
                        results_df = pd.concat([results_df, df])
                        logger.info("Finished experiment %d/%d", i+1, num_experiments)

                    #Get rid of columns with empty values
                    results_df = results_df.dropna(axis=1, how='all')
                    #Save results based on config
                    results_filename = "Feature Extractor=" + str(f_e) + "_Outlier Detector=" + str(a_s_m) + "_Dataset=" + str(dataset)
                    if config['memory_mode'] is not None:
                        results_filename += "_Memory Mode=" + config['memory_mode']
                    results_filename = results_filename.replace(":", "-").replace("/", "-").replace("\\", "-").replace("=", "-").replace("[", "").replace("]", "").replace("'", "")
                    results_df.to_csv(f"results/{results_filename}.csv")

def extract_config(experiment, f_e, a_s_m, d, memory_mode):
    config = {}
    config['normalize'] = experiment.get('normalize', True)
    config['num_experiments'] = experiment.get('num_experiments', 1)
    config['metrics'] = experiment.get('metrics', ['best f1 (F1)'])
    config['load_model'] = experiment.get('load_model', False)
    config['train_epochs'] = experiment.get('train_epochs', 10)

    if (f_e == "ADCN"):  
        config['ADCN_clustering_loss'] = experiment.get('ADCN_clustering_loss', False)
        config['ADCN_label_mode'] = experiment.get('ADCN_label_mode', 'random')

    if (f_e == "ADCN" and a_s_m == None):
        config['metrics'] = ['F1', 'ACC']
    
    config['feature_extractor'] = f_e
    config['anomaly_score_method'] = a_s_m
    config['dataset'] = d
    config['memory_mode'] = memory_mode

    #Check that ADCN and AE are normalized
    if config['feature_extractor'] == "ADCN" or config['feature_extractor'] == "AE":
        config['normalize'] = True
        logger.warning("ADCN and AE strategies require normalized data. Setting normalize to True")
        
    #Add other settings here
    for key in experiment.keys():
        if key not in config.keys():
            config[key] = experiment[key]
        
    return config

def run_single_experiment(config,datastream, device):
    df = pd.DataFrame(columns=['Train Experience', 'Test Experience'] + config['metrics'])

    logger.info("Created single experiment dataframe with columns: %s", df.columns.array)
    f_e = config['feature_extractor']
    if f_e == None:
        
        f_e = PassThroughExtractor(datastream, config)
        logger.info("Using PassThroughExtractor")
    elif f_e == "ADCN":
        pass #ADCN is handled within it's run function
        logger.info("Using ADCN")
    elif f_e == "AE":
        from FeatureExtractors.AE_Exactor import AE_Extractor
        f_e = AE_Extractor(nFeatures=datastream.nInput, nLatent=96, train_epochs=config['train_epochs'])
        logger.info("Using AE")
    elif f_e == "EaM":
        from FeatureExtractors.EaM import EaM
        f_e = EaM(nFeatures=datastream.nInput, nLatent=96)
        logger.info("Using EaM")
    elif f_e == "VaDE":
        logger.info("Using VaDE")
    elif f_e == "CFE":
        from FeatureExtractors.CFE import CFE
        f_e = CFE(datastream, config, device)
    elif f_e == "CFEM":
        from FeatureExtractors.CFEM import CFE
        f_e = CFE(datastream, config, device)
    elif f_e == "MetCon":
        from FeatureExtractors.Met_Con import Met_Con
        f_e = Met_Con(nFeatures=datastream.nInput, batch_size=config['batch_size'], nLatent=96)
    elif f_e == "Met2":
        from FeatureExtractors.Met2 import Met2
        f_e = Met2(datastream)
    elif f_e == "Met2LwF":
        from FeatureExtractors.Met2LwF import Met2LwF
        f_e = Met2LwF(datastream)
    elif f_e == "CFEMet2":
        from FeatureExtractors.CFEMet2 import CFEMet2
        f_e = CFEMet2(datastream, config, device)
    elif f_e == "Met2LwFBig":
        from FeatureExtractors.Met2LwFBig import Met2LwF
        f_e = Met2LwF(datastream)
    elif f_e == "Met2LwFRecon":
        from FeatureExtractors.Met2LwFRecon import Met2LwF
        f_e = Met2LwF(datastream, train_epochs=config['train_epochs'])
    elif f_e == "LwFRecon":
        from FeatureExtractors.LwFRecon import Met2LwF
        f_e = Met2LwF(datastream, train_epochs=config['train_epochs'])
    elif f_e == "Met2Recon":
        from FeatureExtractors.Met2Recon import Met2Recon
        f_e = Met2Recon(datastream, train_epochs=config['train_epochs'])
    else:
        raise ValueError(f"Feature extractor {f_e} is invalid")


    a_s_m = config['anomaly_score_method']
    if a_s_m == None:
        if f_e == "ADCN":
            df = run_ADCN(datastream, config, df)
            return df
        elif f_e == "VaDE":
            df = run_VaDE(datastream, config, df)
            config['metrics'] = ['F1', 'ACC']
            return df
        else:
            raise ValueError("Anomaly score method must be specified if feature extractor is not ADCN")
        
    if a_s_m == "PCA":
        from AnomolyDetectors.PCA import PCA_model
        a_s_m = PCA_model()
    elif a_s_m == "AE":
        from AnomolyDetectors.AE import AE
        a_s_m = AE(nFeatures=datastream.nInput, nLatent=10, train_epochs=10)
    elif a_s_m == "Random":
        from AnomolyDetectors.Random import Random
        a_s_m = Random()
    elif a_s_m == "IF":
        from sklearn.ensemble import IsolationForest
        a_s_m = IsolationForest()
        config['metrics'] = ['F1', 'ACC']
    elif a_s_m == "LOF":
        from sklearn.neighbors import LocalOutlierFactor
        a_s_m = LocalOutlierFactor(novelty=True, contamination=0.001)
        config['metrics'] = ['F1', 'ACC']
    elif a_s_m == "OCSVM":
        from sklearn.svm import OneClassSVM
        a_s_m = OneClassSVM(kernel='linear')
        config['metrics'] = ['F1', 'ACC']
    elif a_s_m == "KMeans":
        from AnomolyDetectors.K_Means import K_Means
        a_s_m = K_Means(datastream, f_e)
        config['metrics'] = ['F1', 'ACC']
    elif a_s_m == "DIF":
        from AnomolyDetectors.DIF import DIF
        a_s_m = DIF()
    elif a_s_m == "SLAD":
        from AnomolyDetectors.SLAD import SLAD
        a_s_m = SLAD()
    elif a_s_m == "ICL":
        from AnomolyDetectors.ICL import ICL
        a_s_m = ICL()
    else:
        raise ValueError(f"Anomaly score method {a_s_m} is invalid")


    if f_e == "ADCN":
        df = fit_and_test_ADCN(a_s_m, datastream, config, df)
    else:
        df = fit_and_test(f_e, a_s_m,datastream, config, df, device)

    return df

def run_ADCN(datastream, config, df):
    nNodeInit  = 96
    nOutput   = 500
    nHidden   = 1000
    epoch = 1
    ADCN_clustering_loss = True
    
    train_time_df = pd.DataFrame(columns=['Train Experience', 'Time'])
    inference_time_df = pd.DataFrame(columns=['Train Experience', 'Inference Time'])
    
    def test_ADCN(df, test_experiences, train_time_df, inference_time_df, model, train_experience_num):
            logger.info(f"Testing ADCN for train experience {train_experience_num}")      
            
            train_end_time = time.time()
            train_time_df.loc[len(train_time_df.index)] = [train_experience_num, train_end_time - test_ADCN.train_start_time]
            
            inference_start_time = time.time()
            for j, test_experience in enumerate(test_experiences):
                logger.info("Testing Experience %i", j)
                X_test = test_experience[0]
                Y_test = test_experience[1]

                labels = model.predict(X_test)
                
                f1 = metrics.f1_score(Y_test, labels)
                acc = (labels == Y_test).sum() / len(Y_test)
                

                df.loc[len(df.index)] = [train_experience_num, j, f1, acc.item()]
            
            inference_end_time = time.time()
            inference_time_df.loc[len(inference_time_df.index)] = [train_experience_num, inference_end_time - inference_start_time]
            test_ADCN.train_start_time = time.time()
            return df

    test_ADCN.train_start_time = time.time()
    model = ADCNbasic.ADCN(datastream.nOutput, nInput=nOutput, nHiddenNode = nNodeInit, desiredLabels=[0,1])
    model.ADCNcnn   = simpleMPL(datastream.nInput, nHidden, nOutput)
    ADCNdatastream = datastream.get_ADCN_datastream(config["ADCN_label_mode"])
    callback = partial(test_ADCN, df, datastream.test_experiences, train_time_df, inference_time_df)
    model, performanceHistory0, allPerformance0  = ADCNmainloop.ADCNmainMT(model, ADCNdatastream, callback,df, noOfEpoch = epoch, clusteringLoss=ADCN_clustering_loss, device = device)       
    df = test_ADCN(df, datastream.test_experiences,train_time_df, inference_time_df,model, datastream.nExperiences-1)\
    #Save train and inference time dataframes
    train_time_df.to_csv(f"time_results/train_time_{config['feature_extractor']}_{config['anomaly_score_method']}_{config['dataset']}.csv")
    inference_time_df.to_csv(f"time_results/inference_time_{config['feature_extractor']}_{config['anomaly_score_method']}_{config['dataset']}.csv")
                             
    return df

def fit_and_test_ADCN(a_s_m, datastream, config, df):
    if config["load_model"]:
        logger.error("ADCN model loading not implemented yet")
        exit()
    else:
        nNodeInit  = 96
        nOutput   = 500
        nHidden   = 1000
        epoch = 1
        def test_novelty_methods(df, test_experiences, model, train_experience_num):
            logger.info(f"Testing ADCN for train experience {train_experience_num}")
            # for name, novelty_model in novelty_models.items():
            print("Init normal size: ", datastream.init_normal.size())
            encoded_x = model.encode(datastream.init_normal)
            #Save model
            logger.info(f"Saving model for train experience {train_experience_num}")
            # #Save ADCNnet with pickle
            # with open(f'./models/ADCN_no_clustering_loss_{config["dataset"]}_{train_experience_num}.pkl', 'wb') as f:
            #     pkl.dump(model, f)

            a_s_m.fit(encoded_x.cpu())  
            encoded_val = model.encode(datastream.init_val)
            scores_val = a_s_m.predict(encoded_val.cpu())

            for j, test_experience in enumerate(test_experiences):
                logger.info("Testing Experience %i", j)
                X_test = test_experience[0]
                Y_test = test_experience[1]

                encoded_X_test = model.encode(X_test)

                scores = a_s_m.predict(encoded_X_test.cpu())

                metric_scores = metrics.get_metric_scores(config['metrics'], Y_test, scores,scores_val)
                
                logger.info("Metrics: %s", config['metrics'])
                logger.info("Metric Scores %s", metric_scores)

                df.loc[len(df.index)] = [train_experience_num, j] + metric_scores
            return df
        callback = partial(test_novelty_methods, df, datastream.test_experiences)
        model = ADCNbasic.ADCN(datastream.nOutput, nInput=nOutput, nHiddenNode = nNodeInit, desiredLabels=[0,1])
        model.ADCNcnn   = simpleMPL(datastream.nInput, nHidden, nOutput)
        ADCNdatastream = datastream.get_ADCN_datastream(config["ADCN_label_mode"])
        model, performanceHistory0, allPerformance0  = ADCNmainloop.ADCNmainMT(model, ADCNdatastream, callback,df, noOfEpoch = epoch, clusteringLoss=False, device = device)
        df = test_novelty_methods(df, datastream.test_experiences, model, datastream.nExperiences-1)
        return df

def fit_and_test(f_e, a_s_m, datastream, config, df, device):
    logger.info("Fitting and testing feature extractor: %s and anomaly score method: %s", f_e, a_s_m)
    
    train_time_df = pd.DataFrame(columns=['Train Experience', 'Time'])
    inference_time_df = pd.DataFrame(columns=['Train Experience', 'Time'])
    
    for i, experience in enumerate(datastream.train_experiences):
        logger.info("--------- Beginning train experience %d ----------", i)
        #Check if feature extractor is PassThroughExtractor object and experience is not 0
        if isinstance(f_e, PassThroughExtractor) and i != 0:
            logger.info("Feature Extractor is None and experience is not 0. Skipping")
            break
        X = experience[0]
        y = experience[1]
        logger.info("X size: %s, y size: %s", X.size(), y.size())

        start_time = time.time()
        logger.info("Fitting feature extractor")
        f_e.fit(X, device)

        logger.info("Encoding init normal data")
        encoded_init = f_e(datastream.init_normal)
        
        logger.info("Fitting anomaly score method")
        a_s_m.fit(encoded_init)
        logger.info("Fit anomaly score method")
        end_time = time.time()
        
        train_time_df.loc[len(train_time_df.index)] = [i, end_time - start_time]
        logger.info("Train Time: %f", end_time - start_time)
        
        start_time = time.time()
        test_df = test_experiences(f_e, a_s_m, datastream, config)
        end_time = time.time()
        
        inference_time_df.loc[len(inference_time_df.index)] = [i, end_time - start_time]
        logger.info("Inference Time: %f", end_time - start_time)
        test_df['Train Experience'] = i
        df = pd.concat([df, test_df])
        
    logger.info("Train Time Statistics: %f, %f", train_time_df['Time'].mean(), train_time_df['Time'].std())
    logger.info("Inference Time Statistics: %f, %f", inference_time_df['Time'].mean(), inference_time_df['Time'].std())
    #Save train and inference time dataframes
    train_time_df.to_csv(f"time_results/train_time_{config['feature_extractor']}_{config['anomaly_score_method']}_{config['dataset']}.csv")
    inference_time_df.to_csv(f"time_results/inference_time_{config['feature_extractor']}_{config['anomaly_score_method']}_{config['dataset']}.csv")
    return df
        
def test_experiences(f_e, a_s_m, datastream, config):
    logger.info("--------- Testing all experiences ---------")

    df = pd.DataFrame(columns=['Test Experience'] + config['metrics'])
    val_encoded = f_e(datastream.init_val)
    scores_val = a_s_m.predict(val_encoded)
    logger.info("Scores Val Statisitcs: %f, %f", scores_val.mean(), scores_val.std())

    for j, test_experience in enumerate(datastream.test_experiences):
        logger.info("Testing Experience %i", j)
        X_test = test_experience[0]
        Y_test = test_experience[1]

        encoded_X_test = f_e(X_test)
        
        # encoded_X_test_anomaly = encoded_X_test[Y_test == 1]
        # encoded_X_test_normal = encoded_X_test[Y_test == 0]
        
        # dist = torch.mean(torch.cdist(encoded_X_test_anomaly, encoded_X_test_normal))
        # logger.info("Anomaly vs Normal Distance: %f", dist)
        scores = a_s_m.predict(encoded_X_test)
        logger.info("Scores: %s", scores)
        if config['metrics'] == ['F1', 'ACC']:
            scores = (scores + 1) / 2 #Convert -1, 1 to 0, 1
            metric_scores = [metrics.f1_score(Y_test, scores), ((Y_test == scores).sum() / len(Y_test)).item()]
        else:
            metric_scores = metrics.get_metric_scores(config['metrics'], Y_test, scores, scores_val)
        
        logger.info("Metrics: %s", config['metrics'])
        logger.info("Metric Scores %s", metric_scores)

        df.loc[len(df.index)] = [j] + metric_scores
    return df
#%%