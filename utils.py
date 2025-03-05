#%%
import torch
import numpy as np
# from torchvision import datasets, transforms #Check what this does. 
from sklearn.model_selection import train_test_split
import sklearn
from datastream import datastream
import logging
import pandas as pd
from sklearn.metrics import precision_recall_curve
from metrics import RocAUC, PrAUC

DATA_PATH = '/Users/seant/Desktop/Current Programs/new-ucl-ids-1/data/'

EDGE_NORMAL_ATTACK = 7
EDGE_ATTACK_TO_LABEL = {0: 'Backdoor', 1: 'DDoS_HTTP', 2: 'DDoS_ICMP', 3: 'DDoS_TCP', 4: 'DDoS_UDP', 5: 'Fingerprinting', 6: 'MITM', 7: 'Normal', 8: 'Password', 9: 'Port_Scanning', 10: 'Ransomware', 11: 'SQL_injection', 12: 'Uploading', 13: 'Vulnerability_scanner', 14: 'XSS'}

X_NORMAL_ATTACK = 11
X_ATTACK_TO_LABEL = {0: 'BruteForce', 1: 'C&C', 2: 'Dictionary', 3: 'Discovering_resources', 4: 'Exfiltration', 5: 'Fake_notification', 6: 'False_data_injection', 7: 'Generic_scanning', 8: 'MQTT_cloud_broker_subscription', 9: 'MitM', 10: 'Modbus_register_reading', 11: 'Normal', 12: 'RDOS', 13: 'Reverse_shell', 14: 'Scanning_vulnerability', 15: 'TCP Relay', 16: 'crypto-ransomware', 17: 'fuzzing', 18: 'insider_malcious'}  

MQTT_NORMAL_ATTACK = 3
MQTT_ATTACK_TO_LABEL = {0: 'bruteforce', 1: 'dos', 2: 'flood', 3: 'Normal', 4: 'malformed', 5: 'slowite'}

MNIST_NORMAL_ATTACK = torch.tensor([0,1])
MNIST_ATTACK_TO_LABEL = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

WUST_NORMAL_ATTACK = 4
WUST_ATTACK_TO_LABEL = {0: 'Backdoor', 1: 'CommInj', 2: 'DoS', 3: 'Reconn', 4: 'normal'}

USNW_NORMAL_ATTACK = 6
UNSW_ATTACK_TO_LABEL = {0: 'Analysis', 1: 'Backdoor', 2: 'DoS', 3: 'Exploits', 4: 'Fuzzers', 5: 'Generic', 6: 'Normal', 7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms'}


CICIDS18_NORMAL_ATTACK = 0
CICIDS18_ATTACK_TO_LABEL = {0: 'Benign', 1: 'Bot', 2: 'Brute Force -Web', 3: 'Brute Force -XSS', 4: 'DDOS attack-HOIC', 5: 'DDOS attack-LOIC-UDP', 6: 'DDoS attacks-LOIC-HTTP', 7: 'DoS attacks-GoldenEye', 8: 'DoS attacks-Hulk', 9: 'DoS attacks-SlowHTTPTest', 10: 'DoS attacks-Slowloris', 11: 'FTP-BruteForce', 12: 'Infilteration', 13: 'SQL Injection', 14: 'SSH-Bruteforce'}

CICIDS17_NORMAL_ATTACK = 0
CICIDS17_ATTACK_TO_LABEL = {0: 'BENIGN', 1: 'Bot', 2: 'DDoS', 3: 'DoS GoldenEye', 4: 'DoS Hulk', 5: 'DoS Slowhttptest', 6: 'DoS slowloris', 7: 'FTP-Patator', 8: 'Heartbleed', 9: 'Infiltration', 10: 'PortScan', 11: 'SSH-Patator', 12: 'Web Attack-Brute Force', 13: 'Web Attack-Sql Injection', 14: 'Web Attack-XSS'}

{0: 'Backdoor', 1: 'CommInj', 2: 'DoS', 3: 'Reconn', 4: 'normal'}

logger = logging.getLogger()

def get_benchmark(dataset, normalize=True):
    if dataset == "EdgeIIoT":
        return get_EdgeIIoT_benchmark(5, None, normalize)
    elif dataset == "XIIoT":
        return get_XIIoT_benchmark(5, None,normalize)
    elif dataset == "MQTT":
        return get_MQTT_benchmark(5, [[0],[1],[2],[4], [5]], normalize=normalize)
    elif dataset == "WUST":
        return get_WUST_benchmark(4, [[0],[1],[2],[3]], normalize=normalize)
    elif dataset == "UNSW":
        return get_UNSW_benchmark(5, None, normalize=normalize)
    elif dataset == "CICIDS18":
        return get_CICIDS18_benchmark(5, None, normalize=normalize)
    elif dataset == "CICIDS17":
        return get_CICIDS17_benchmark(5, None, normalize=normalize)
    else:
        raise ValueError("Unknown dataset")

def get_EdgeIIoT_benchmark(n_experiences=5, class_order=None,  normalize=False):
    # Load the dataset
    X, Y = load_data('EdgeIIoT')
    # Convert to tensors
    X = torch.tensor(X)
    Y = torch.argmax(torch.tensor(Y), dim=1)

    #Remove 10% of normal data for initialization
    normal_mask = torch.isin(Y, EDGE_NORMAL_ATTACK)
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(EDGE_ATTACK_TO_LABEL)):
            if i == EDGE_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="EdgeIIoT")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="EdgeIIoT")

    return datastream(train_experiences, test_experiences, init_normal, 95, name="EdgeIIoT", normalize=normalize)

def get_XIIoT_benchmark(n_experiences=5, class_order=None,  normalize=False):
    # Load the dataset
    X, Y = load_data('XIIoT')
    
    # Convert to tensors
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).int()

    #Remove 10% of normal data for initialization
    normal_mask = torch.isin(Y, X_NORMAL_ATTACK)
    print("Number of Normal:",normal_mask.sum())
    print("Number of Attack:", (~normal_mask).sum())
    print("Total:", Y.size(0))
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(X_ATTACK_TO_LABEL)):
            if i == X_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="XIIoT")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="XIIoT")
    
    return datastream(train_experiences, test_experiences, init_normal, 56, name="XIIoT", normalize=normalize)

def get_UNSW_benchmark(n_experiences=5, class_order=None, normalize=False):
    X, Y = load_data('UNSW')

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).int()

    normal_mask = torch.isin(Y, USNW_NORMAL_ATTACK)
    print("Number of Normal:",normal_mask.sum())
    print("Number of Attack:", (~normal_mask).sum())
    print("Total:", Y.size(0))
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(UNSW_ATTACK_TO_LABEL)):
            if i == USNW_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="UNSW")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="UNSW")
    return datastream(train_experiences, test_experiences, init_normal, 36, name="UNSW", normalize=normalize)

def get_CICIDS18_benchmark(n_experiences=5, class_order=None, normalize=False):
    X, Y = load_data('CIC-IDS-2018')

    X = torch.tensor(np.nan_to_num(X)).float()
    Y = torch.tensor(Y).int()

    normal_mask = torch.isin(Y, CICIDS18_NORMAL_ATTACK)
    print("Number of Normal:",normal_mask.sum())
    print("Number of Attack:", (~normal_mask).sum())
    print("Total:", Y.size(0))
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(CICIDS18_ATTACK_TO_LABEL)):
            if i == CICIDS18_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="CICIDS18")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="CICIDS18")
    return datastream(train_experiences, test_experiences, init_normal, 70, name="CICIDS18", normalize=normalize)

def get_CICIDS17_benchmark(n_experiences=5, class_order=None, normalize=False):
    X, Y = load_data('CIC-IDS-2017')
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).int().squeeze()

    normal_mask = torch.isin(Y, CICIDS17_NORMAL_ATTACK)
    print("Number of Normal:",normal_mask.sum())
    print("Number of Attack:", (~normal_mask).sum())
    print("Total:", Y.size(0))
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(CICIDS17_ATTACK_TO_LABEL)):
            if i == CICIDS17_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="CICIDS17")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="CICIDS17")
    return datastream(train_experiences, test_experiences, init_normal, 78, name="CICIDS17", normalize=normalize)

def get_WUST_benchmark(n_experiences=5, class_order=None, normalize=False):
    X, Y = load_data('WUST')
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).int()

    normal_mask = torch.isin(Y, WUST_NORMAL_ATTACK)
    print("Number of Normal:",normal_mask.sum())
    print("Number of Attack:", (~normal_mask).sum())
    print("Total:", Y.size(0))
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(WUST_ATTACK_TO_LABEL)):
            if i == WUST_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="WUST")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="WUST")
    return datastream(train_experiences, test_experiences, init_normal, 41, name="WUST", normalize=normalize) 

def get_MQTT_benchmark(n_experiences=5, class_order=None, normalize=False):
    X, Y = load_data('MQTT')
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).int()

    normal_mask = torch.isin(Y, MQTT_NORMAL_ATTACK)
    sparse_normal = torch.rand_like(Y, dtype=torch.float32) < 0.1
    normal_mask = normal_mask & sparse_normal

    init_normal = X[normal_mask]

    X = X[~normal_mask]
    Y = Y[~normal_mask]

    if class_order is None:
        class_order = [ [] for _ in range(n_experiences) ] 
        for i in range(len(MQTT_ATTACK_TO_LABEL)):
            if i == MQTT_NORMAL_ATTACK:
                continue
            class_order[i % n_experiences].append(i)
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_experiences = create_split_experiences(X_train, Y_train, class_order, n_experiences, name="MQTT")
    test_experiences = create_split_experiences(X_test, Y_test, class_order, n_experiences, name="MQTT")
    return datastream(train_experiences, test_experiences, init_normal, 33, name="MQTT", normalize=normalize) 

def load_data(folder_name):
    X = np.load(DATA_PATH + folder_name + '/x.npy')
    Y = np.load(DATA_PATH + folder_name + '/y.npy')
    return X, Y

def create_split_experiences(X, Y, class_order, n_experiences, name="Edge-IIoT"):    
    normal_attack = get_normal_attack(name)
    normal_mask = torch.isin(Y, normal_attack)

    X_normal = X[normal_mask]
    Y_normal = Y[normal_mask]
    shuffle_idx = torch.randperm(X_normal.size(0))
    X_normal = X_normal[shuffle_idx]
    Y_normal = Y_normal[shuffle_idx]
    X_normal = torch.chunk(X_normal, n_experiences)
    Y_normal = torch.chunk(Y_normal, n_experiences)

    experiences = []
    for classes, X_n, Y_n in zip(class_order, X_normal, Y_normal):
        mask = np.isin(Y, classes)
        X_exp = X[mask]
        Y_exp = Y[mask]
        X_exp = torch.cat((X_exp, X_n))
        Y_exp = torch.cat((Y_exp, Y_n))
        shuffle_idx = torch.randperm(X_exp.size(0))
        X_exp = X_exp[shuffle_idx]
        Y_exp = Y_exp[shuffle_idx]
        experiences.append((X_exp, Y_exp))

    return experiences

def get_normal_attack(name):
    if name == "EdgeIIoT":
        return EDGE_NORMAL_ATTACK
    elif name == "XIIoT":
        return X_NORMAL_ATTACK
    elif name == "MQTT":
        return MQTT_NORMAL_ATTACK
    elif name == "MNIST":
        return MNIST_NORMAL_ATTACK
    elif name == "WUST":
        return WUST_NORMAL_ATTACK
    elif name == "UNSW":
        return USNW_NORMAL_ATTACK
    elif name == "CICIDS18":
        return CICIDS18_NORMAL_ATTACK
    elif name == "CICIDS17":
        return CICIDS17_NORMAL_ATTACK
    else:
        raise ValueError("Unknown dataset")
    
def get_attack_to_label(name):
    if name == "EdgeIIoT":
        return EDGE_ATTACK_TO_LABEL
    elif name == "XIIoT":
        return X_ATTACK_TO_LABEL
    elif name == "MQTT":
        return MQTT_ATTACK_TO_LABEL
    elif name == "MNIST":
        return MNIST_ATTACK_TO_LABEL
    elif name == "WUST":
        return WUST_ATTACK_TO_LABEL
    elif name == "UNSW":
        return UNSW_ATTACK_TO_LABEL
    elif name == "CICIDS18":
        return CICIDS18_ATTACK_TO_LABEL
    elif name == "CICIDS17":
        return CICIDS17_ATTACK_TO_LABEL
    else:
        raise ValueError("Unknown dataset")

def deleteRowTensor(x, index, mode = 1):
    if mode == 1:
        x = x[torch.arange(x.size(0))!=index] 
    elif mode == 2:
        # delete more than 1 row
        # index is a list of deleted row
        allRow = torch.arange(x.size(0)).tolist()
        
        for ele in sorted(index, reverse = True):  
            del allRow[ele]

        remainderRow = torch.tensor(allRow).long()

        x = x[remainderRow]
    
    return x

def log_datastream(dataset, datastream):
    logger.info("Loaded %s datastream with: %d input features, %d output features, %d experiences,", dataset, datastream.nInput, datastream.nOutput, datastream.nExperiences)
    logger.info("Init normal shape: %s", str(datastream.init_normal.shape))
    for i, train_experience in enumerate(datastream.train_experiences):
        logger.info("Train experience %d X shape: %s", i, str(train_experience[0].shape))
        logger.info("Train experience %d Y shape: %s", i, str(train_experience[1].shape))

    for i, test_experience in enumerate(datastream.test_experiences):
        logger.info("Test experience %d X shape: %s", i, str(test_experience[0].shape))
        logger.info("Test experience %d Y shape: %s", i, str(test_experience[1].shape))

    for i, train_labels in enumerate(datastream.multiclass_train_labels):
        attacks = train_labels.unique().tolist()
        attack_to_label = get_attack_to_label(dataset)
        attack_labels = [attack_to_label[attack] for attack in attacks]
        logger.info("Attacks in experience %d: %s", i, attack_labels)
        logger.info("Percetage of normal in experience %d: %f", i, (train_labels == get_normal_attack(dataset)).sum().item() / len(train_labels))
