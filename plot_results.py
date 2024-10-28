#%%
import pandas as pd
import os

novelty_metrics = ["trivial val 3 Std (F1)", "trivial val 2 Std (F1)", "trivial val 1 Std (F1)", "best f1 (F1)", "roc auc", "pr auc"]
cl_metrics = ["F1"]

novelty_detectors_df = pd.DataFrame(columns=['Novelty Detector', 'Dataset', 'Memory Mode', 'Trivial Val 3 F1 Mean', 'Trivial Val 3 F1 Std', 'Trivial Val 2 F1 Mean', 'Trivial Val 2 F1 Std', 'Trivial Val 1 F1 Mean', 'Trivial Val 1 F1 Std', 'Best F1 F1 Mean', 'Best F1 F1 Std', 'ROC AUC Mean', 'ROC AUC Std', 'PR AUC Mean', 'PR AUC Std'])
continual_learning_df = pd.DataFrame(columns=['Novelty Detector', 'Dataset','F1 Mean', 'F1 Std'])

def analyze_novelty_detector(results_df, feature_extractor):
    results = []
    #If train experience > 0, then we have continual learning
    max_train_experience = results_df['Train Experience'].max()
    if max_train_experience > 0:
        # select train experience == test experience
        results_df = results_df[results_df['Train Experience'] == results_df['Test Experience']]
    for metric in novelty_metrics:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        results += [mean, std]
    return results

def analyze_cl_detector(results_df, feature_extractor):
    results = []
    #If train experience > 0, then we have continual learning
    max_train_experience = results_df['Train Experience'].max()
    if max_train_experience > 0:
        # select train experience == test experience
        results_df = results_df[results_df['Train Experience'] == results_df['Test Experience']]
    for metric in cl_metrics:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        results += [mean, std]
    return results

for csv in os.listdir(f'./results/'):
    if 'csv' not in csv:
        continue
    config = {}
    info = csv.split('_')
    results_df = pd.read_csv(f'./results/'+csv)
    values = []
    row = {}
    for i in info[::-1]:
        if "-" not in i:
            continue
        key, value = i.split('-')
        row[key] = value
    if row['Outlier Detector'] not in ['None', 'KMeans']:
        novelty_detector = row['Outlier Detector']
        if row['Feature Extractor'] != 'None':
            novelty_detector = row['Feature Extractor'] + "-" + novelty_detector
        dataset = row['Dataset'].split('.')[0]
        memory_mode = row.get('Memory Mode', 'None').split('.')[0]
        results = analyze_novelty_detector(results_df, row['Feature Extractor'])
        new_row = [novelty_detector, dataset,memory_mode] + results
        columns = novelty_detectors_df.columns
        novelty_detectors_df.loc[len(novelty_detectors_df)] = new_row
    else:
        results = analyze_cl_detector(results_df, row['Feature Extractor'])
        novelty_detector = row['Feature Extractor']
        dataset = row['Dataset'].split('.')[0]
        if row['Outlier Detector'] != 'None':
            novelty_detector += "-" + row['Outlier Detector']
        new_row = [novelty_detector, dataset] + results
        continual_learning_df.loc[len(continual_learning_df)] = new_row
            
#%%
novelty_detectors_df.sort_values(by=['Dataset', 'Novelty Detector', 'Memory Mode'], inplace=True)
novelty_detectors_df.to_csv('novelty_detectors.csv', index=False)
# %%

import matplotlib.pyplot as plt
import seaborn as sns

Datasets = novelty_detectors_df['Dataset'].unique()
plotting_metrics = ['Trivial Val 3 F1 Mean', 'Trivial Val 2 F1 Mean', 'Trivial Val 1 F1 Mean', 'Best F1 F1 Mean', 'ROC AUC Mean', 'PR AUC Mean']
novelty_detectors_df['Memory Mode'] = novelty_detectors_df['Memory Mode'].replace('None', '')
novelty_detectors_df['Memory Mode'] = novelty_detectors_df['Memory Mode'].replace('none', '')
novelty_detectors_df['Memory Mode'] = novelty_detectors_df['Memory Mode'].replace('perfect', '-PM')
novelty_detectors_df['Strategy'] = novelty_detectors_df['Novelty Detector'] + novelty_detectors_df['Memory Mode']
continual_learning_df['Strategy'] = continual_learning_df['Novelty Detector']
# selected_metric = 'Best F1 F1 Mean'
# selected_metric = 'PR AUC Mean'
selected_metric = 'Trivial Val 3 F1 Mean'
# selected_metric = 'Trivial Val 2 F1 Mean'
# selected_metric = 'Trivial Val 1 F1 Mean'
continual_learning_df[selected_metric] = continual_learning_df['F1 Mean']
new_df = pd.concat([novelty_detectors_df, continual_learning_df])
print(new_df['Strategy'].unique())
# strategy_list = ['CFE-PCA', 'CFE-PCA-PM']
# new_df = new_df[new_df['Strategy'].str.contains('CFE')]
new_df.sort_values(by=['Dataset', 'Strategy', selected_metric], inplace=True)
new_df.to_csv('new_df.csv', index=False)

for dataset in Datasets:
    # create a bar plot for each dataset
    # if dataset in ["MQTT", "EdgeIIoT", "CICIDS18"]:
    #     continue
    if dataset != 'WUST' and dataset != 'XIIoT' and dataset != 'UNSW':
        continue
    plt.figure(figsize=(10, 5))
    sns.barplot(data=new_df[new_df['Dataset'] == dataset], x='Strategy', y=selected_metric)
    #Space out the x-axis labels
    
    
    plt.title(f'{dataset} - {selected_metric}')
    #Print the top 5 strategies
    dataset_df = new_df[new_df['Dataset'] == dataset].sort_values(by=selected_metric, ascending=False)
    
    for i, row in dataset_df.iterrows():
        print(row['Strategy'], row[selected_metric])
       
    plt.xticks(rotation=90)
# for dataset in Datasets:
#     print(dataset)
#     print(novelty_detectors_df[novelty_detectors_df['Dataset'] == dataset].groupby('Novelty Detector').mean())

# fig, ax = plt.subplots(len(Datasets), len(plotting_metrics), figsize=(len(Datasets) * 5,len(plotting_metrics) * 5))
# novelty_detectors_df['Memory Mode'] = novelty_detectors_df['Memory Mode'].replace('None', '')
# novelty_detectors_df['Memory Mode'] = novelty_detectors_df['Memory Mode'].replace('perfect', '-PM')
# novelty_detectors_df['Strategy'] = novelty_detectors_df['Novelty Detector'] + novelty_detectors_df['Memory Mode']

# for i, dataset in enumerate(Datasets):
#     for j, metric in enumerate(plotting_metrics):
#         ax[i, j].set_title(f'{dataset} - {metric}')
#         sns.barplot(data=novelty_detectors_df[novelty_detectors_df['Dataset'] == dataset], x='Strategy', y=metric, ax=ax[i, j])
#         # ax[i, j].set_xticklabels(ax[i, j].get_xticklabels(), rotation=90)

# #Give each strategy a different color, based on it's name
# strategies = novelty_detectors_df['Strategy'].unique()
# colors = sns.color_palette("pastel6", len(strategies))
# color_dict = dict(zip(strategies, colors))
# for i, dataset in enumerate(Datasets):
#     for j, metric in enumerate(plotting_metrics):
#         for k, strategy in enumerate(strategies):
#             ax[i, j].get_children()[k].set_color(color_dict[strategy])

# plt.tight_layout()
# plt.show()
# %%
