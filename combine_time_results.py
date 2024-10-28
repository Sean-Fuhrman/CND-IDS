#%%
import os
import pandas as pd
train_df = pd.DataFrame(columns=['Dataset', 'Strategy', 'Average Time'])
inference_df = pd.DataFrame(columns=['Dataset', 'Stategy', 'Average Time'])

for file in os.listdir('time_results/'):
    if not file.endswith('.csv'):
        continue
    info = file.split('_')
    if len(info) < 4:
        continue
    df_type = info[0]
    dataset = info[-1].split('.')[0]
    strategy = info[2] + '-' + info[3]
    df = pd.read_csv(f'time_results/{file}')
    if 'Time' not in df.columns:
        if 'Inference Time' in df.columns:
            df.rename(columns={'Inference Time': 'Time'}, inplace=True)
        else:
            continue
    avg_time = df['Time'].mean()
    if df_type == 'train':
        train_df.loc[len(train_df)] = [dataset, strategy, avg_time]
    else:
        inference_df.loc[len(inference_df)] = [dataset, strategy, avg_time]
        
train_df.to_csv('time_results/train_time_results.csv', index=False)
inference_df.to_csv('time_results/inference_time_results.csv', index=False)
        
# %%

