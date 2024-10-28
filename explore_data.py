#%% 
import numpy as np
#Check if Wust contains Nans

X = np.load('./data/WUST/x.npy')
Y = np.load('./data/WUST/y.npy')

print(np.isnan(X).any())
print(np.isnan(Y).any())

#%%
import utils

# Check if wust datastream contains Nans
datastream = utils.get_WUST_benchmark(n_experiences=5)
#%%
for i in range(5):
    print(np.isnan(datastream.train_experiences[i][0]).any())
    print(np.isnan(datastream.train_experiences[i][1]).any())
    print(np.isnan(datastream.test_experiences[i][0]).any())
    print(np.isnan(datastream.test_experiences[i][1]).any())
#%%
# Check if ADCN datastream contains Nans

datastream = datastream.get_ADCN_datastream()
#%%
for i in range(5):
    print(np.isnan(datastream.labeledData[i]).any())
    print(np.isnan(datastream.unlabeledData[i]).any())
    print(np.isnan(datastream.labeledLabel[i]).any())
    print(np.isnan(datastream.unlabeledLabel[i]).any())
# %%
