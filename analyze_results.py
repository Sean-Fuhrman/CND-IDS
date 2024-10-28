#%%
import torch
import pandas as pd
import os
# Plot all results

def get_mean_std(list):
    if 'N/A' in list:
        return 'N/A', 'N/A'
    mean = sum(list) / len(list)
    std = (sum([(x - mean) ** 2 for x in list]) / len(list)) ** 0.5
    return mean, std

def analyze_results(results_df, metric):
    if 'Experiment Number' in results_df.columns:
        num_experiments = int(results_df['Experiment Number'].max()) + 1
    else:
        num_experiments = 1
        results_df['Experiment Number'] = 0

    if 'Threshold Method' in results_df.columns:
        results_df = results_df[results_df['Threshold Method'] == 'trivial_percentile_val']
        print('Using best f score threshold method')
    ress = []
    bwts = []
    fwts = []
    for i in range(num_experiments):
        experiment_df = results_df[results_df['Experiment Number'] == i]
        res, bwt, fwt = analyze_experiment(experiment_df, metric)
        ress.append(res)
        bwts.append(bwt)
        fwts.append(fwt)
    res_mean, res_std = get_mean_std(ress)
    bwt_mean, bwt_std = get_mean_std(bwts)
    fwt_mean, fwt_std = get_mean_std(fwts)
    return res_mean, res_std, fwt_mean, fwt_std, bwt_mean, bwt_std, 


def analyze_experiment(experiment_df,metric):
    nExperiences = int(experiment_df['Train Experience'].max())
    if nExperiences == 0:
        res = experiment_df[metric].mean()
    else:
        res = experiment_df[experiment_df['Train Experience'] == nExperiences-1][metric].mean()

    bwt = 0
    for i in range(0, nExperiences):
        res_i_i = experiment_df[(experiment_df['Train Experience'] == i) & (experiment_df['Test Experience'] == i)][metric].values[0]
        res_T_i = experiment_df[(experiment_df['Train Experience'] == nExperiences) & (experiment_df['Test Experience'] == i)][metric].values[0]
        bwt += res_i_i - res_T_i
    bwt /= (nExperiences - 1)

    #Fwt is the average of test experience j on train experience i s.t. j > i
    fwt = 0
    for i in range(0, nExperiences):
        for j in range(i+1, nExperiences):
            res_i_j = experiment_df[(experiment_df['Train Experience'] == i) & (experiment_df['Test Experience'] == j)][metric].values[0]
            fwt += res_i_j
    if nExperiences == 0:
        fwt = 'N/A'
        bwt = 'N/A'
    else:
        fwt /= (nExperiences * (nExperiences - 1) / 2)

    return res, bwt, fwt

final_df = pd.DataFrame(columns=["Feature Extractor", "Outlier Detector", "Dataset", "Normalized", "Thresholding Method", "Result Mean", "Result Std"])
fwt_df = pd.DataFrame(columns=["Feature Extractor", "Outlier Detector", "Dataset", "Normalized", "Thresholding Method", "Result Mean", "Result Std"])
bwt_df = pd.DataFrame(columns=["Feature Extractor", "Outlier Detector", "Dataset", "Normalized", "Thresholding Method", "Result Mean", "Result Std"])
metrics = set()
for csv in os.listdir(f'./results/'):
    if 'csv' not in csv:
        continue
    config = {}
    info = csv.split('_')
    results_df = pd.read_csv(f'./results/'+csv)
    values = []
    row = {}
    for i in info[::-1]:
        if "=" not in i:
            continue
        key, value = i.split('=')
        row[key] = value
        
    metrics = row.pop('Metrics')[1:-5].split(',')
    for metric in metrics:
        metric = metric.strip().replace("'","")
        curr_row = row.copy()
        res_mean, res_std, fwt_mean, fwt_std, bwt_mean, bwt_std = analyze_results(results_df, metric)
        curr_row["Thresholding Method"] = metric
        curr_row["Result Mean"] = res_mean
        curr_row["Result Std"] = res_std
        final_df.loc[len(final_df.index)] = curr_row
        curr_row["Metric"] = metric + " FWT"
        curr_row["Result Mean"] = fwt_mean
        curr_row["Results Std"] = fwt_std
        fwt_df.loc[len(final_df.index)] = curr_row
        curr_row["Metric"] = metric + " BWT"
        curr_row["Result Mean"] = bwt_mean
        curr_row["Results Std"] = bwt_std
        bwt_df.loc[len(final_df.index)] = curr_row
# final_df = final_df[final_df['Thresholding Method'] == 'best f1 (F1)']
final_df.sort_values(by=['Dataset','Thresholding Method','Result Mean'], inplace=True, ascending=False)
fwt_df.sort_values(by=['Dataset','Thresholding Method','Result Mean'], inplace=True, ascending=False)
bwt_df.sort_values(by=['Dataset','Thresholding Method','Result Mean'], inplace=True, ascending=False)

final_df.to_csv('final_results.csv')


#%%