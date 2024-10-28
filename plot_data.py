#%%
import utils
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

# Set up logging to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#%%



def plot_datastream(datastream, labels, name):
    multiclass_train_labels = datastream.multiclass_train_labels

    init_normal = datastream.init_normal

    df = pd.DataFrame(columns=['Experience', 'Attack Num', 'Attack Type', 'Count', 'Percentage of Experience'])

    print(init_normal.shape)

    for i, experience_labels in enumerate(multiclass_train_labels):
        Y = experience_labels
        unique, counts = np.unique(Y, return_counts=True)
        total = sum(counts)
        for j, count in zip(unique, counts):
            df.loc[len(df.index)]= [i, j, labels[j], count, count/total]
        

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Experience", y="Count", hue="Attack Type", data=df)

    #Create directory if it does not exist
    import os
    if not os.path.exists(f"./plots/dataset-experiences/{name}"):
        os.makedirs(f"./plots/dataset-experiences/{name}")

    plt.title(f"{name}-Training Set: Attacks per Experience")
    #Save df
    df.to_csv(f"./plots/dataset-experiences/{name}/{name}_Attacks_Per_Experience.csv")

    #move legend outside of plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f"./plots/dataset-experiences/{name}/{name}_Attacks_Per_Experience.pdf")
#%%

datastream = utils.get_WUST_benchmark(4, [[0],[1],[2],[3]])
labels = utils.WUST_ATTACK_TO_LABEL

total_train = 0
total_test = 0
for i, train_experience in enumerate(datastream.train_experiences):
    X, Y = train_experience
    total_train += len(Y)
    print(f"Experience {i} Train: {len(Y)}")
    
for i, test_experience in enumerate(datastream.test_experiences):
    X, Y = test_experience
    total_test += len(Y)
    print(f"Experience {i} Test: {len(Y)}")

print(f"Total Train: {total_train}")
print(f"Total Test: {total_test}")
print(f"Total init_normal: {len(datastream.init_normal)}")
plot_datastream(datastream, labels, "WUST")

#%%
datastream = utils.get_MQTT_benchmark(5, [[0],[1],[2],[4], [5]])
labels = utils.MQTT_ATTACK_TO_LABEL

plot_datastream(datastream, labels, "MQTT")
#%%
datastream = utils.get_EdgeIIoT_benchmark(n_experiences=5)
labels = utils.EDGE_ATTACK_TO_LABEL

plot_datastream(datastream, labels, "EdgeIIoT")
# %%
datastream = utils.get_XIIoT_benchmark(n_experiences=5)
labels = utils.X_ATTACK_TO_LABEL

total_train = 0
total_test = 0
for i, train_experience in enumerate(datastream.train_experiences):
    X, Y = train_experience
    total_train += len(Y)
    print(f"Experience {i} Train: {len(Y)}")
    
for i, test_experience in enumerate(datastream.test_experiences):
    X, Y = test_experience
    total_test += len(Y)
    print(f"Experience {i} Test: {len(Y)}")

print(f"Total Train: {total_train}")
print(f"Total Test: {total_test}")
print(f"Total init_normal: {len(datastream.init_normal)}")
plot_datastream(datastream, labels, "XIIoT")

#%%

datastream = utils.get_benchmark("CICIDS17")
labels = utils.CICIDS17_ATTACK_TO_LABEL

total_train = 0
total_test = 0
for i, train_experience in enumerate(datastream.train_experiences):
    X, Y = train_experience
    total_train += len(Y)
    print(f"Experience {i} Train: {len(Y)}")
    
for i, test_experience in enumerate(datastream.test_experiences):
    X, Y = test_experience
    total_test += len(Y)
    print(f"Experience {i} Test: {len(Y)}")

print(f"Total Train: {total_train}")
print(f"Total Test: {total_test}")
print(f"Total init_normal: {len(datastream.init_normal)}")
plot_datastream(datastream, labels, "CICIDS17")
#%%

datastream = utils.get_benchmark("UNSW")
labels = utils.UNSW_ATTACK_TO_LABEL


total_train = 0
total_test = 0
for i, train_experience in enumerate(datastream.train_experiences):
    X, Y = train_experience
    total_train += len(Y)
    print(f"Experience {i} Train: {len(Y)}")
    
for i, test_experience in enumerate(datastream.test_experiences):
    X, Y = test_experience
    total_test += len(Y)
    print(f"Experience {i} Test: {len(Y)}")

print(f"Total Train: {total_train}")
print(f"Total Test: {total_test}")
print(f"Total init_normal: {len(datastream.init_normal)}")
plot_datastream(datastream, labels, "UNSW")

#%%

def ADCN_datastream_to_csv(ADCN_datastream, name):
    train_df = pd.DataFrame(columns=['Experience', 'Attack', 'Labeled', 'Count', 'Percentage of Class in Experiences'])

    for i in range(0 , ADCN_datastream.nTask):
        unlabeledData = ADCN_datastream.unlabeledData[i]
        unlabeledLabel = ADCN_datastream.unlabeledLabel[i]
        assert len(unlabeledData) == len(unlabeledLabel)

        unlabeled_ones = torch.sum(unlabeledLabel).item()
        unlabeled_zeros = len(unlabeledLabel) - unlabeled_ones

        labeledData = ADCN_datastream.labeledData[i]
        labeledLabel = ADCN_datastream.labeledLabel[i]
        assert len(labeledData) == len(labeledLabel)

        labeled_ones = torch.sum(labeledLabel).item()
        labeled_zeros = len(labeledLabel) - labeled_ones

        total_ones = labeled_ones + unlabeled_ones
        total_zeros = labeled_zeros + unlabeled_zeros

        if total_ones == 0:
            total_ones = 1
        if total_zeros == 0:
            total_zeros = 1

        train_df.loc[len(train_df.index)] = [i,'Attack', 'No',unlabeled_ones, unlabeled_ones/total_ones]
        train_df.loc[len(train_df.index)] = [i,'Normal', 'No',unlabeled_zeros, unlabeled_zeros/total_zeros]

     
        train_df.loc[len(train_df.index)] = [i,'Attack', 'Yes', labeled_ones, labeled_ones/total_ones]
        train_df.loc[len(train_df.index)] = [i,'Normal', 'Yes', labeled_zeros, labeled_zeros/total_zeros]

    train_df.to_csv(f"./results/dataset-experiences/{name}/{name}_ADCN_Attacks_Per_Experience.csv", float_format='%.6f')

    test_df = pd.DataFrame(columns=['Experience', 'Attack', 'Labeled', 'Count', 'Percentage of Experience'])

    for i in range(0 , ADCN_datastream.nTask):
        dataTest = ADCN_datastream.unlabeledDataTest[i]
        labelTest = ADCN_datastream.unlabeledLabelTest[i]
        assert len(dataTest) == len(labelTest)

        ones = torch.sum(labelTest).item()
        zeros = len(labelTest) - ones

        test_df.loc[len(test_df.index)] = [i,'Attack', 'No',ones, ones/len(labelTest)]
        test_df.loc[len(test_df.index)] = [i,'Normal', 'No',zeros, zeros/len(labelTest)]

    test_df.to_csv(f"./results/dataset-experiences/{name}/{name}_ADCN_Test_Attacks_Per_Experience.csv")

datastream = utils.get_WUST_benchmark(n_experiences=3)

ADCN_datastream = datastream.get_ADCN_datastream()

ADCN_datastream_to_csv(ADCN_datastream, "WUST")

#%%
datastream = utils.get_MNIST_benchmark(n_experiences=5)

ADCN_datastream = datastream.get_ADCN_datastream()

ADCN_datastream_to_csv(ADCN_datastream, "MNIST")

#%%
datastream = utils.get_MQTT_benchmark(n_experiences=5)

ADCN_datastream = datastream.get_ADCN_datastream()

ADCN_datastream_to_csv(ADCN_datastream, "MQTT")

#%%


datastream = utils.get_EdgeIIoT_benchmark(n_experiences=5)

ADCN_datastream = datastream.get_ADCN_datastream()

ADCN_datastream_to_csv(ADCN_datastream, "EdgeIIoT")

#%%

datastream = utils.get_XIIoT_benchmark(n_experiences=5)

ADCN_datastream = datastream.get_ADCN_datastream()

ADCN_datastream_to_csv(ADCN_datastream, "XIIoT")


# %%
