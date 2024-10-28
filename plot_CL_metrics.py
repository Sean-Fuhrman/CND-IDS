#%%
import pandas as pd
import os


# selected_CL_methods = [
#     # {
#     #     'Outlier Detector': 'ICL', 
#     #     'Feature Extractor': 'Met2LwFRecon',
#     #     'Strategy': 'Met2LwFRecon + ICL',
#     # },
#      {
#         'Outlier Detector': 'PCA', 
#         'Feature Extractor': 'Met2LwFRecon',
#         'Strategy': 'CND-IDS',
#     },
#     #   {
#     #     'Outlier Detector': 'PCA', 
#     #     'Feature Extractor': 'Met2Recon',
#     #     'Strategy': 'No LCL',
#     # },
#     #  {
#     #     'Outlier Detector': 'DIF',
#     #     'Feature Extractor': 'Met2LwFRecon',
#     #     'Strategy': 'Met2LwFRecon + DIF',
#     #  },
#     # {
#     #     'Outlier Detector': 'SLAD',
#     #     'Feature Extractor': 'Met2LwFRecon',
#     #     'Strategy': 'Met2LwFRecon + SLAD',
#     # },
#     {
#         'Outlier Detector': 'KMeans',
#         'Feature Extractor': 'AE',
#         'Strategy': 'LwF',
#     },
#     # {
#     #     'Outlier Detector': 'ICL',
#     #     'Feature Extractor': 'Met2LwFRecon',
#     #     'Strategy': 'ICL+Met2LwFRecon',
#     # }
#     # {
#     #     'Outlier Detector': 'PCA',
#     #     'Feature Extractor': 'CFE',
#     #     'Strategy': 'CFE+PCA',
#     # }
#     {
#         'Outlier Detector': 'None',
#         'Feature Extractor': 'ADCN',
#         'Strategy': 'ADCN',
#     }
# ]

selected_CL_methods = [
    {
        'Outlier Detector': 'PCA', 
        'Feature Extractor': 'Met2LwFRecon',
        'Strategy': 'CND-IDS',
    },
      {
        'Outlier Detector': 'PCA', 
        'Feature Extractor': 'LwFRecon',
        'Strategy': 'No LCS',
    },
]
#"trivial val 3 Std (F1)", "trivial val 2 Std (F1)",  "trivial val 1 Std (F1)", "best f1 (F1)", "roc auc", "pr auc"
# selected_metric = 'trivial val 2 Std (F1)'
selected_metric = 'best f1 (F1)'
# selected_metric = 'pr auc'

selected_datasets = ['CICIDS17','UNSW', 'WUST', 'XIIoT']

cl_df = pd.DataFrame(columns=['Novelty Detector', 'Dataset', 'AVG', 'FWT', 'BWT'])

def analyze_cl_detector(results_df):
    if selected_metric in results_df.columns:
        results_df['F1'] = results_df[selected_metric]
        
    nExperiences = int(results_df['Train Experience'].max())
    if nExperiences == 0:
        res = results_df['F1'].mean()
    else:  
        res_df = results_df[(results_df['Train Experience'] == nExperiences)]
        res = res_df['F1'].mean()

    bwt = 0
    count = 0
    for i in range(0, nExperiences):
        for j in range(i):
            if i == j:
                continue
            curr = results_df[(results_df['Train Experience'] == i) & (results_df['Test Experience'] == j)]['F1'].values[0]
            prev = results_df[(results_df['Train Experience'] == i-1) & (results_df['Test Experience'] == j)]['F1'].values[0]
            bwt += curr - prev
            count += 1
    bwt /= count
    #Fwt is the average of test experience j on train experience i s.t. j > i
    fwt = 0
    for i in range(0, nExperiences):
        for j in range(i+1, nExperiences):
            res_i_j = results_df[(results_df['Train Experience'] == i) & (results_df['Test Experience'] == j)]['F1'].values[0]
            fwt += res_i_j
    if nExperiences == 0:
        fwt = 'N/A'
        bwt = 'N/A'
    else:
        fwt /= (nExperiences * (nExperiences - 1) / 2)

    return [res, fwt, bwt]

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
    if 'Memory Mode' in row.keys() and row['Memory Mode'] != 'None':
        continue
    for selected_CL_method in selected_CL_methods:
        if row['Outlier Detector'] == selected_CL_method['Outlier Detector'] and row['Feature Extractor'] == selected_CL_method['Feature Extractor'] and row['Dataset'].split('.')[0] in selected_datasets:
            results = analyze_cl_detector(results_df)
            new_row = [selected_CL_method['Strategy'], row['Dataset'].split('.')[0]] + results
            columns = cl_df.columns
            cl_df.loc[len(cl_df)] = new_row


# %%

cl_df.to_csv(f'plots/cl_metrics_{selected_metric}_ablation.csv')
# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1, 3, figsize=(18,4))

metrics = ['AVG', 'FWT', 'BWT']
color_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

dataset_name_map = {
    'CICIDS17': 'CICIDS2017',
    'UNSW': 'UNSW-NB15',
    'WUST': 'WUSTL-IIoT',
    'XIIoT': 'XIIoTID',
}

metric_name_map = {
    'AVG': 'AVG',
    'FWT': 'FwdTrans',
    'BWT':  'BwdTrans',
}

#Rename the datasets
cl_df['Dataset'] = cl_df['Dataset'].apply(lambda x: dataset_name_map[x])

#Rename the metrics
cl_df.rename(columns=metric_name_map, inplace=True)
metrics = ['AVG', 'FwdTrans', 'BwdTrans']

for i, metric in enumerate(metrics):
    sns.barplot(x='Dataset', y=metric, hue='Novelty Detector', data=cl_df, ax=axs[i], palette=color_palette, edgecolor='black')
    # axs[i].set_title(metric)
    axs[i].set_ylabel(metric )
    axs[i].set_xlabel('Dataset')
    axs[i].get_legend().remove()

#put the legend outside the plot
legend_args = {
    'bbox_to_anchor': (-0.3, -0.14),
    'ncol': len(cl_df['Novelty Detector'].unique()),
    'title': 'Continual Learning Strategies',
}
plt.legend(**legend_args)
#add padding between the plots
plt.subplots_adjust(wspace=0.2)
plt.savefig(f'plots/cl_metrics_{selected_metric}.pdf', bbox_inches='tight')
plt.show()
# %%
