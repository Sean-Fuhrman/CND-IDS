#%%
import pandas as pd
import os

selected_novelty_detectors = ['PCA', 'DIF','OCSVM', "LOF"]
# selected_novelty_detectors = []
#"trivial val 3 Std (F1)", "trivial val 2 Std (F1)",  "trivial val 1 Std (F1)", "best f1 (F1)", "roc auc", "pr auc"
# selected_metric = 'trivial val 2 Std (F1)'
# selected_metric = 'trivial val 3 Std (F1)'
# selected_metric = 'best f1 (F1)'
selected_metric = 'pr auc'

selected_datasets = ['CICIDS17','UNSW', 'WUST', 'XIIoT']

novelty_detectors_df = pd.DataFrame(columns=['Novelty Detector', 'Dataset', selected_metric + ' Mean', selected_metric + ' Std'])

def analyze_novelty_detector(results_df):
    results = []
    #If train experience > 0, then we have continual learning
    max_train_experience = results_df['Train Experience'].max()
    if selected_metric not in results_df.columns:
        results_df[selected_metric] = results_df['F1']
    if max_train_experience > 0:
        # select train experience == test experience
        results_df = results_df[results_df['Train Experience'] == results_df['Test Experience']]
    
    mean = results_df[selected_metric].mean()
    std = results_df[selected_metric].std()
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
    if row['Outlier Detector'] in selected_novelty_detectors and row['Feature Extractor'] == 'None' and row['Dataset'].split('.')[0] in selected_datasets:
        novelty_detector = row['Outlier Detector']
        dataset = row['Dataset'].split('.')[0]
        results = analyze_novelty_detector(results_df)
        new_row = [novelty_detector, dataset] + results
        columns = novelty_detectors_df.columns
        novelty_detectors_df.loc[len(novelty_detectors_df)] = new_row

other_methods = [
    # {
    # 'Outlier Detector': 'PCA',
    # 'Feature Extractor': 'CFE',
    # 'Strategy' : 'CFE+PCA'
    # }, 
    # {
    #     'Outlier Detector': 'PCA',
    #     'Feature Extractor': 'Met2',
    #     'Strategy' : 'Met2+PCA'
    # }, 
    # {
    #     'Outlier Detector': 'ICL',
    #     'Feature Extractor': 'Met2LwFRecon',
    #     'Strategy' : 'Met2LwFRecon+ICL',
    # },
    {
        'Outlier Detector': 'PCA',
        'Feature Extractor': 'Met2LwFRecon',
        'Strategy' : 'CND-IDS'
    },
    # {
    #     'Outlier Detector': 'PCA',
    #     'Feature Extractor': 'Met2LwFReconOld',
    #     'Strategy' : 'Met2LwFReconOld+PCA'
    # }
    # {
    #     'Outlier Detector': 'PCA',
    #     'Feature Extractor': 'Met2LwFReconBalanced',
    #     'Strategy' : 'Met2LwFReconBalanced+PCA'
    # }
]
for our_method in other_methods:
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
        if row['Outlier Detector'] == our_method['Outlier Detector'] and row['Feature Extractor'] == our_method['Feature Extractor'] and row['Dataset'].split('.')[0] in selected_datasets:
            dataset = row['Dataset'].split('.')[0]
            results = analyze_novelty_detector(results_df)
            new_row = [our_method['Strategy'], dataset] + results
            columns = novelty_detectors_df.columns
            novelty_detectors_df.loc[len(novelty_detectors_df)] = new_row

print(novelty_detectors_df)
#%%
novelty_detectors_df.to_csv(f'./plots/novelty_detectors_{selected_metric}.csv', index=False)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

Datasets = novelty_detectors_df['Dataset'].unique()

dataset_name_map = {
    'CICIDS17': 'CICIDS2017',
    'UNSW': 'UNSW-NB15',
    'WUST': 'WUSTL-IIoT',
    'XIIoT': 'XIIoTID',
}

#Rename the datasets
novelty_detectors_df['Dataset'] = novelty_detectors_df['Dataset'].map(dataset_name_map)
Datasets = novelty_detectors_df['Dataset'].unique()

fig, axs = plt.subplots(1,len(Datasets), figsize=(9, 3.3))
if len(Datasets) == 1:
    axs = [axs]
# color_palette = ['tab:red','tab:purple','tab:blue', 'tab:orange','tab:green']
# Strategy_order = ['LOF','OCSVM','DIF','PCA', 'CND-IDS',]
color_palette = ['tab:blue', 'tab:orange','tab:green']
Strategy_order = ['DIF','PCA','CND-IDS',]
for i, dataset in enumerate(Datasets):
    dataset_df = novelty_detectors_df[novelty_detectors_df['Dataset'] == dataset]
    #Remove strategies from strategy_order that are not in the dataset
    Strategy_order = [strategy for strategy in Strategy_order if strategy in dataset_df['Novelty Detector'].unique()]
    dataset_df['Novelty Detector'] = pd.Categorical(dataset_df['Novelty Detector'], categories=Strategy_order, ordered=True)
    sns.barplot(hue='Novelty Detector', y=selected_metric + ' Mean', data=dataset_df, ax=axs[i], palette=color_palette, edgecolor='black', ci=None)
    axs[i].set_title(f'{dataset}')
    if i == 0:
        axs[i].set_ylabel("PR-AUC Mean (%)")
    else:
        axs[i].set_ylabel('')
    # axs[i].set_xlabel('Novelty Detectors')
    #get rid of the legend
    axs[i].get_legend().remove()
    axs[i].set_ylim([0, 0.9])
    #Set the range of the y-axis to be what the value of 'Random' strategy is
    # axs[i].set_ylim([dataset_df[dataset_df['Novelty Detector'] == 'Random'][selected_metric + ' Mean'].min(), 1])
    # if dataset == 'CICIDS17':
    #     axs[i].set_ylim([0.05, 0.7])
    # elif dataset == 'UNSW':
    #     axs[i].set_ylim([0.3, 0.8])
    # elif dataset == 'WUST':
    #     axs[i].set_ylim([0, 0.9])
    # elif dataset == 'XIIoT':
    #     axs[i].set_ylim([0.2, 0.9])
    
#put the legend outside the plot
legend_args = {
    'bbox_to_anchor': (-0.4, -0.06),
    'ncol': len(novelty_detectors_df['Novelty Detector'].unique()),
    'title': 'Novelty Detectors',
}
plt.legend(**legend_args)
# plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig(f'./plots/novelty_detectors_{selected_metric}.pdf', bbox_inches='tight')
plt.show()
    
# %%
