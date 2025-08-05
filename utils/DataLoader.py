from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import os
import bisect
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from scipy.sparse.linalg import svds
import sys

NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool,drop_last=True):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

class Data_noNext:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        
class Data_withSampleInterval:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,sampleInterval:np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.sampleInterval = sampleInterval
        
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        

def get_train_data(dataset_name: str, val_ratio: float, test_ratio: float,interval:str):

    graph_df = pd.read_csv('./processed_data_pretrain/{}/{}/ml_train_{}.csv'.format(dataset_name,interval, dataset_name))

    edge_raw_features = np.zeros((len(graph_df['ts'])+1, 1))
    node_raw_features = np.zeros((len(np.unique(np.concatenate((graph_df['u'], graph_df['i']), axis=0)))+1, 1))
    
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)        
    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'    

    
    unique_nodes = np.unique(np.concatenate((graph_df.u.values, graph_df.i.values)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    src_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.u.values], dtype=np.longlong)
    dst_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.i.values], dtype=np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    sampleInterval = graph_df.sampleInterval.values
    
    is_sorted = np.all(node_interact_times[:-1] <= node_interact_times[1:])
    if not is_sorted:
        node_interact_times = np.sort(node_interact_times)
        sorted_indices = np.argsort(node_interact_times)
        src_node_ids = src_node_ids[sorted_indices]
        dst_node_ids = dst_node_ids[sorted_indices]
        edge_ids = graph_df.idx.values.astype(np.longlong)[sorted_indices]
        labels = graph_df.label.values[sorted_indices]
        sampleInterval = graph_df.sampleInterval.values[sorted_indices]

    train_data = Data_withSampleInterval(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,sampleInterval=sampleInterval, edge_ids=edge_ids, labels=labels)

    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, train_data

def get_pretrain_train_data(dataset_name: str, val_ratio: float, test_ratio: float,interval:str):

    if dataset_name.startswith('mixed_'):
        graph_df = pd.read_csv('./processed_data_pretrain/{}/ml_train_{}.csv'.format(dataset_name, dataset_name))
    else:
        graph_df = pd.read_csv('./processed_data_pretrain/{}/{}/ml_train_{}.csv'.format(dataset_name,interval, dataset_name))

    

    edge_raw_features = np.zeros((len(graph_df['ts'])+1, 1))
    node_raw_features = np.zeros((len(np.unique(np.concatenate((graph_df['u'], graph_df['i']), axis=0)))+1, 1))
    
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)        
    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'    

    unique_nodes = np.unique(np.concatenate((graph_df.u.values, graph_df.i.values)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    src_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.u.values], dtype=np.longlong)
    dst_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.i.values], dtype=np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    sampleInterval = graph_df.sampleInterval.values
    
    is_sorted = np.all(node_interact_times[:-1] <= node_interact_times[1:])
    if not is_sorted:
        node_interact_times = np.sort(node_interact_times)
        sorted_indices = np.argsort(node_interact_times)
        src_node_ids = src_node_ids[sorted_indices]
        dst_node_ids = dst_node_ids[sorted_indices]
        edge_ids = graph_df.idx.values.astype(np.longlong)[sorted_indices]
        labels = graph_df.label.values[sorted_indices]
        sampleInterval = graph_df.sampleInterval.values[sorted_indices]

    train_data = Data_withSampleInterval(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,sampleInterval=sampleInterval, edge_ids=edge_ids, labels=labels)

    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, train_data

def get_pretrain_train_data_mixedSI(dataset_name: str, val_ratio: float, test_ratio: float,interval:str):
    if dataset_name.startswith('mixed_'):
        graph_df = pd.read_csv('./processed_data_pretrain/{}/ml_train_{}.csv'.format(dataset_name, dataset_name))
    else:
        graph_df = pd.read_csv('./processed_data_pretrain/{}/{}/ml_train_{}.csv'.format(dataset_name,interval, dataset_name))

    edge_raw_features = np.zeros((len(graph_df['ts'])+1, 1))
    node_raw_features = np.zeros((len(np.unique(np.concatenate((graph_df['u'], graph_df['i']), axis=0)))+1, 1))
    
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)        
    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'    

    unique_nodes = np.unique(np.concatenate((graph_df.u.values, graph_df.i.values)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    src_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.u.values], dtype=np.longlong)
    dst_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.i.values], dtype=np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    sampleInterval = graph_df.sampleInterval.values
    
    is_sorted = np.all(node_interact_times[:-1] <= node_interact_times[1:])
    if not is_sorted:
        node_interact_times = np.sort(node_interact_times)
        sorted_indices = np.argsort(node_interact_times)
        src_node_ids = src_node_ids[sorted_indices]
        dst_node_ids = dst_node_ids[sorted_indices]
        edge_ids = graph_df.idx.values.astype(np.longlong)[sorted_indices]
        labels = graph_df.label.values[sorted_indices]
        sampleInterval = graph_df.sampleInterval.values[sorted_indices]

    train_data = Data_withSampleInterval(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,sampleInterval=sampleInterval, edge_ids=edge_ids, labels=labels)

    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, train_data

def get_pretrain_finetune_data(dataset_name: str, val_ratio: float, test_ratio: float,interval:str):
    if dataset_name.startswith('mixed_'):
        graph_df = pd.read_csv('./processed_data_pretrain/{}/ml_pretrain_{}.csv'.format(dataset_name, dataset_name))
    else:
        graph_df = pd.read_csv('./processed_data_pretrain/{}/{}/ml_pretrain_{}.csv'.format(dataset_name,interval, dataset_name))

    edge_raw_features = np.zeros((len(graph_df['ts'])+1, 1))
    node_raw_features = np.zeros((len(np.unique(np.concatenate((graph_df['u'], graph_df['i']), axis=0)))+1, 1))
    
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)        
    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'    

    
    unique_nodes = np.unique(np.concatenate((graph_df.u.values, graph_df.i.values)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    src_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.u.values], dtype=np.longlong)
    dst_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.i.values], dtype=np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    sampleInterval = graph_df.sampleInterval.values
    
    is_sorted = np.all(node_interact_times[:-1] <= node_interact_times[1:])
    if not is_sorted:
        node_interact_times = np.sort(node_interact_times)
        sorted_indices = np.argsort(node_interact_times)
        src_node_ids = src_node_ids[sorted_indices]
        dst_node_ids = dst_node_ids[sorted_indices]
        edge_ids = graph_df.idx.values.astype(np.longlong)[sorted_indices]
        labels = graph_df.label.values[sorted_indices]
        sampleInterval = graph_df.sampleInterval.values[sorted_indices]

    train_data = Data_withSampleInterval(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,sampleInterval=sampleInterval, edge_ids=edge_ids, labels=labels)

    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, train_data


def get_pretrain_test_data(dataset_name: str, val_ratio: float, test_ratio: float,interval:str):

    if dataset_name.startswith('mixed_'):
        graph_df = pd.read_csv('./processed_data_pretrain/{}/ml_test_{}.csv'.format(dataset_name, dataset_name))
    else:
        graph_df = pd.read_csv('./processed_data_pretrain/{}/{}/ml_test_{}.csv'.format(dataset_name,interval, dataset_name))

    edge_raw_features = np.zeros((len(graph_df['ts'])+1, 1))
    node_raw_features = np.zeros((len(np.unique(np.concatenate((graph_df['u'], graph_df['i']), axis=0)))+1, 1))
    
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)        
    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'    

    
    unique_nodes = np.unique(np.concatenate((graph_df.u.values, graph_df.i.values)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    src_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.u.values], dtype=np.longlong)
    dst_node_ids = np.array([node_id_mapping[node_id] for node_id in graph_df.i.values], dtype=np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    sampleInterval = graph_df.sampleInterval.values
    is_sorted = np.all(node_interact_times[:-1] <= node_interact_times[1:])
    if not is_sorted:
        node_interact_times = np.sort(node_interact_times)
        sorted_indices = np.argsort(node_interact_times)
        src_node_ids = src_node_ids[sorted_indices]
        dst_node_ids = dst_node_ids[sorted_indices]
        edge_ids = graph_df.idx.values.astype(np.longlong)[sorted_indices]
        labels = graph_df.label.values[sorted_indices]
        sampleInterval = graph_df.sampleInterval.values[sorted_indices]

    test_data = Data_withSampleInterval(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,sampleInterval=sampleInterval, edge_ids=edge_ids, labels=labels)

    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, test_data



def adjust_times(before_node,node_interact_times, n):
    adjusted_times = np.zeros_like(node_interact_times)

    adjusted_times[0] = before_node

    for i in range(1, len(node_interact_times)):
        if node_interact_times[i] == node_interact_times[i - 1]:
            adjusted_times[i] = adjusted_times[i - 1]
        else:
            adjusted_times[i] = adjusted_times[i - 1] + n

    return adjusted_times

