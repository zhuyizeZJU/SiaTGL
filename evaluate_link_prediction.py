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
import pickle 
from models.SiaTGL import SiaTGL
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler,get_neighbor_sampler_sampleInterval
from utils.incontext_evaluate_models_utils import evaluate_model_link_prediction,evaluate_model_link_prediction_sampleIntervalAware
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_test_data,get_pretrain_test_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from transformers import GPT2LMHeadModel, GPT2Config
from datetime import datetime
from utils.utils_forInterval import MyData,my_collate_fn

from torch.utils.data import DataLoader


if __name__ == "__main__":
    remove_ratio = 0.8
    warnings.filterwarnings('ignore')
    num_workers=32
    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    testGap = 'original'
    # get data for training, validation and testing

    with open('./generate_dataset/dataset_mapping.pkl', 'rb') as f:
        dataset_dict_read = pickle.load(f)
    node_raw_features, edge_raw_features, test_data= \
        get_pretrain_test_data(dataset_name=args.pretrainTestDataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio,interval=args.testInterval)
    
    
    test_neighbor_sampler = get_neighbor_sampler_sampleInterval(data=test_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)


    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=test_data.src_node_ids, dst_node_ids=test_data.dst_node_ids, seed=2)

    patch_size = args.patch_size
    max_input_sequence_length = args.max_input_sequence_length

        

        
    indices_3 = np.argsort(test_data.node_interact_times)
    test_data.train_src_node_ids = test_data.src_node_ids[indices_3]
    test_data.train_dst_node_ids = test_data.dst_node_ids[indices_3]
    test_data.train_node_interact_times = test_data.node_interact_times[indices_3]
    test_data.train_sampleInterval = test_data.sampleInterval[indices_3]
    test_data.train_edge_ids = test_data.edge_ids[indices_3]
    test_data.train_labels = test_data.labels[indices_3]
    
    

    test_data = MyData(test_data)
    test_fn = my_collate_fn(args.batch_size,patch_size,max_input_sequence_length,test_neg_edge_sampler,test_neighbor_sampler,False)
    test_idx_data_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=32,shuffle=False, collate_fn=test_fn.Pretrain_collate_fn,drop_last=True,pin_memory=True)


    val_metric_all_runs, new_node_val_metric_all_runs,double_new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs,double_new_node_test_metric_all_runs = [], [], [], [], [], []
    average_precision_mean_list = []
    new_node_average_precision_mean_list = []
    double_new_node_average_precision_mean_list = []
    
    epoch = 49
    for run in range(args.num_runs):
        set_random_seed(seed=2)
        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.pretrainTestDataset}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.pretrainTestDataset}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')
        my_config = GPT2Config(vocab_size=2,n_positions=100,n_embd=172,n_layer=4,n_head=4)
        mymodel = GPT2LMHeadModel(my_config)
        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'NeighborSeq':

            dynamic_backbone = NeighborSeq(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)

        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'SiaTGL':
            dynamic_backbone = SiaTGL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device,ssl_split_n = args.ssl_split_n)
        elif args.model_name == 'FreeDyG':
            dynamic_backbone = FreeDyG(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=test_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device,ssl_split_n = args.ssl_split_n)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        link_predictor = MergeLayer(input_dim1=args.time_feat_dim, input_dim2=args.time_feat_dim,
                                    hidden_dim=args.time_feat_dim, output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        
 
        model.load_state_dict(torch.load(f'./test_saved/finetune/{args.model_name}/{dataset_dict_read[args.pretrainTestDataset]}/seed{run}_ssl{args.ssl_split_n}_f{args.factor}_i{args.testInterval}_d{args.time_feat_dim}/save_model_{epoch}.pkl',map_location=args.device))

        model = convert_to_gpu(model, device=args.device)
        
        model.eval()
        
        model[0].set_neighbor_sampler(test_neighbor_sampler)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        loss_func = nn.BCELoss()
    
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
            
            
        val_losses, val_metrics = evaluate_model_link_prediction_sampleIntervalAware(model_name=args.model_name,
                                                                 model=model,
                                                                 neighbor_sampler=test_neighbor_sampler,
                                                                 evaluate_idx_data_loader=test_idx_data_loader,
                                                                 evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                 evaluate_data=test_data,
                                                                 loss_func=loss_func,
                                                                 num_neighbors=args.num_neighbors,
                                                                 time_gap=args.time_gap,
                                                                 args = args)



        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
        
        avg_precisions = [val_metric["average_precision"] for val_metric in val_metrics]
        average_precision_mean = np.mean(avg_precisions)
        average_precision_mean_list.append(average_precision_mean)
        
        
        val_metric_indicator = []
        for metric_name in val_metrics[0].keys():
            val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
    # load the best model
    result_average = np.mean(average_precision_mean_list)
    result_std_dev = np.std(average_precision_mean_list)


    os.makedirs(f"./logs_eval/{epoch}/{args.model_name}/{args.pretrainTestDataset}/seed{args.seed}", exist_ok=True)
    
    result_logger = logging.getLogger('result_logger')
    result_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(f'./logs_eval/{epoch}/{args.model_name}/{args.pretrainTestDataset}/seed{args.seed}/{args.pretrainTestDataset}.log')
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    result_logger.addHandler(fh)
    result_logger.addHandler(ch)

    result_logger.info('val_metrics: %s',round(result_average,4))
    result_logger.info('std: %s',round(result_std_dev,4))

        
    sys.exit()
