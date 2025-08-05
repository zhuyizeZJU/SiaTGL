import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.DyGIL_faster import DyGIL_faster as DyGIL_mul
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from transformers import GPT2LMHeadModel, GPT2Config
import random
import torch.utils.data as data_
from torch.utils.data import DataLoader


class MyData(data_.Dataset):
    def __init__(self, data):
        self.dataset_num = 1
        self.src_node_ids = data.train_src_node_ids
        self.dst_node_ids = data.train_dst_node_ids
        self.node_interact_times = data.train_node_interact_times
        self.sampleInterval = data.train_sampleInterval
        self.edge_ids = data.train_edge_ids

    def __len__(self):
        return len(self.src_node_ids)

    def __getitem__(self, idx):
        return (self.src_node_ids[idx], 
                self.dst_node_ids[idx],
                self.node_interact_times[idx],
                self.sampleInterval[idx],
                self.edge_ids[idx])

class my_collate_fn:
    def __init__(self, batch_num, patch_size, max_input_sequence_length,
                 train_neg_edge_sampler, train_neighbor_sampler, is_train=True):
        self.batch_num = batch_num
        self.data_num = 1
        self.is_train = is_train
        self.patch_size = patch_size
        self.max_input_sequence_length = max_input_sequence_length
        self.train_neg_edge_sampler = train_neg_edge_sampler
        self.train_neighbor_sampler = train_neighbor_sampler

    def fix_ids(self, src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids, node_map):
        shape0_l = src_padded_nodes_neighbor_ids.shape[0]
        shape0_r = src_padded_nodes_neighbor_ids.shape[1]
        
        shape1_l = dst_padded_nodes_neighbor_ids.shape[0]
        shape1_r = dst_padded_nodes_neighbor_ids.shape[1]
        
        src_padded_nodes_neighbor_ids = src_padded_nodes_neighbor_ids.reshape(-1, self.seq_len, shape0_r)
        dst_padded_nodes_neighbor_ids = dst_padded_nodes_neighbor_ids.reshape(-1, self.seq_len, shape1_r)
        
        return src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids, ids_pos_map

    def Pretrain_collate_fn(self, batch_data):
        batch_src_node_ids = np.array([sq[0] for sq in batch_data])
        batch_dst_node_ids = np.array([sq[1] for sq in batch_data])
        batch_node_interact_times = np.array([sq[2] for sq in batch_data])
        batch_sampleInterval = np.array([sq[3] for sq in batch_data])
        batch_edge_ids = np.array([sq[4] for sq in batch_data])

        train_data_indices = np.argsort(batch_node_interact_times)
        batch_src_node_ids = batch_src_node_ids[train_data_indices]
        batch_dst_node_ids = batch_dst_node_ids[train_data_indices]
        batch_node_interact_times = batch_node_interact_times[train_data_indices]
        batch_sampleInterval = batch_sampleInterval[train_data_indices]
        batch_edge_ids = batch_edge_ids[train_data_indices]

        output = self.get_batch_of_output(
            batch_src_node_ids.reshape(-1, 1),
            batch_dst_node_ids.reshape(-1, 1),
            batch_node_interact_times.reshape(-1, 1),
            batch_sampleInterval.reshape(-1, 1),
            0
        )

        return (output[0].reshape(1, -1), 
                output[1].reshape(1, -1),
                output[2].reshape(1, -1), 
                output[3].reshape(1, -1),
                output[4].reshape(1, -1), 
                output[5].reshape(1, -1),
                batch_edge_ids.reshape(1, -1))

    def get_batch_of_output(self, batch_src_node_ids, batch_dst_node_ids, 
                          batch_node_interact_times, batch_sampleInterval, data_idx):
        _, batch_neg_dst_node_ids = self.train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
        batch_neg_src_node_ids = batch_src_node_ids.copy().reshape(-1, 1)
        batch_neg_dst_node_ids = batch_neg_dst_node_ids.reshape(-1, 1)
        batch_node_interact_times = batch_node_interact_times.reshape(-1, 1)
        batch_sampleInterval = batch_sampleInterval.reshape(-1, 1)

        return (batch_src_node_ids, batch_dst_node_ids,
                batch_neg_src_node_ids, batch_neg_dst_node_ids,
                batch_node_interact_times, batch_sampleInterval)