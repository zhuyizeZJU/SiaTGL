# eval修改事项
# get_neighbor_sampler -> get_neighbor_sampler_sampleInterval
import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import pickle 
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler


from models.SiaTGL import SiaTGL

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler,get_neighbor_sampler_sampleInterval
from utils.metrics import get_link_prediction_metrics,get_link_prediction_metrics_original
from utils.DataLoader import get_pretrain_finetune_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from transformers import GPT2LMHeadModel, GPT2Config
import random
import torch.utils.data as data_
from torch.utils.data import DataLoader

from utils.utils_forInterval import MyData,my_collate_fn
from torch.utils.data import DataLoader
from utils.experiment_config import save_experiment_config

def get_dataset(args):
    node_raw_features, edge_raw_features, train_data= \
        get_pretrain_finetune_data(dataset_name=args.pretrainTestDataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio, interval=str(args.interval))

    train_neighbor_sampler = get_neighbor_sampler_sampleInterval(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0)
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)

    return node_raw_features, edge_raw_features, train_data,train_neighbor_sampler, train_neg_edge_sampler

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    args = get_link_prediction_args(is_evaluation=False)
    patch_size = args.patch_size
    max_input_sequence_length = args.max_input_sequence_length
    
    node_raw_features, edge_raw_features,  train_data,  train_neighbor_sampler, train_neg_edge_sampler= get_dataset(args)

    train_data.train_src_node_ids = train_data.src_node_ids
    train_data.train_dst_node_ids = train_data.dst_node_ids
    train_data.train_node_interact_times = train_data.node_interact_times
    train_data.train_sampleInterval = train_data.sampleInterval
    train_data.train_edge_ids = train_data.edge_ids
    train_data.train_labels = train_data.labels


    for run in range(args.num_runs):
        
        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/original_Benchmark/{args.model_name}/{args.pretrainTestDataset}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/original_Benchmark/{args.model_name}/{args.pretrainTestDataset}/{args.save_model_name}/{str(time.time())}.log")
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
        
        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'FreeDyG':
            dynamic_backbone = FreeDyG(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)

        elif args.model_name == 'SiaTGL':
            dynamic_backbone = SiaTGL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device,interval=args.interval,
                                         ssl_split_n = args.ssl_split_n)



        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        
        
        link_predictor = MergeLayer(input_dim1=args.time_feat_dim, input_dim2=args.time_feat_dim,
                                    hidden_dim=args.time_feat_dim, output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        finetune_epoch = 49
        model.load_state_dict(torch.load(f'./test_saved/pretrain/{args.model_name}/{args.pretrainTestDataset}/seed{run}_ssl{args.ssl_split_n}_f{args.factor}_i0_d{args.time_feat_dim}/save_model_{finetune_epoch}.pkl',map_location=args.device))

        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./test_saved/finetune/{args.model_name}/{args.pretrainTestDataset}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        num_batches = len(train_data.src_node_ids) // args.batch_size
        
        loss_func = nn.BCELoss()
        num_training_steps = args.num_epochs * num_batches
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=num_batches*2, num_training_steps=num_training_steps
        )
        save_path = f'./test_saved/finetune/{args.model_name}/{args.pretrainTestDataset}/seed{args.seed}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("already created:", save_path)
        else:
            print("already exists:", save_path)
        
        save_experiment_config(save_path, args, "finetune")

        for epoch in range(args.num_epochs):
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            all_train_data = MyData(train_data)
            train_fn = my_collate_fn(batch_num=args.batch_size, patch_size=patch_size, max_input_sequence_length=max_input_sequence_length, train_neg_edge_sampler=train_neg_edge_sampler, train_neighbor_sampler=train_neighbor_sampler, is_train=True)
            pretrain_data_loader = DataLoader(all_train_data, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=train_fn.Pretrain_collate_fn, drop_last=True, pin_memory=True)   
            
            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','FreeDyG','SiaTGL']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(pretrain_data_loader, ncols=120)
            for batch_idx, batch_data in enumerate(train_idx_data_loader_tqdm):

                batch_src_node_ids_ls = batch_data[0].astype(int)
                batch_dst_node_ids_ls = batch_data[1].astype(int)
                batch_neg_src_node_ids_ls = batch_data[2].astype(int)
                batch_neg_dst_node_ids_ls = batch_data[3].astype(int)
                batch_node_interact_times_ls = batch_data[4].astype(float)
                batch_sampleInterval_ls = batch_data[5].astype(float)
                # print()
                
                batch_edge_ids_ls = batch_data[6].astype(float)

                
                batch_neg_ssl_loss_ls = []
                batch_pos_ssl_loss_ls = []
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','FreeDyG','SiaTGL']:
                    batch_src_node_embeddings_ls = []
                    batch_dst_node_embeddings_ls = []
                    batch_neg_src_node_embeddings_ls = []
                    batch_neg_dst_node_embeddings_ls = []
                    batch_neg_ssl_loss_ls = []
                    batch_pos_ssl_loss_ls = []

                    for data_idx in range(batch_src_node_ids_ls.shape[0]):

                        model[0].set_neighbor_sampler(train_neighbor_sampler)
                        batch_src_node_embeddings, batch_dst_node_embeddings,ssl_pos_loss =\
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids_ls[data_idx],
                                                                              dst_node_ids=batch_dst_node_ids_ls[data_idx],
                                                                    node_interact_times=batch_node_interact_times_ls[data_idx],
                                                                    node_sampleInterval=batch_sampleInterval_ls[data_idx])

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings,ssl_neg_loss = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids_ls[data_idx],
                                                                              dst_node_ids=batch_neg_dst_node_ids_ls[data_idx],
                                                                    node_interact_times=batch_node_interact_times_ls[data_idx],
                                                                    node_sampleInterval=batch_sampleInterval_ls[data_idx])
                        batch_neg_ssl_loss_ls.append(ssl_pos_loss)
                        batch_pos_ssl_loss_ls.append(ssl_neg_loss)
                        batch_src_node_embeddings_ls.append(batch_src_node_embeddings)
                        batch_dst_node_embeddings_ls.append(batch_dst_node_embeddings)
                        batch_neg_src_node_embeddings_ls.append(batch_neg_src_node_embeddings)
                        batch_neg_dst_node_embeddings_ls.append(batch_neg_dst_node_embeddings)
                    batch_src_node_embeddings_ls = torch.cat(batch_src_node_embeddings_ls,dim=0)
                    batch_dst_node_embeddings_ls = torch.cat(batch_dst_node_embeddings_ls,dim=0)
                    batch_neg_src_node_embeddings_ls = torch.cat(batch_neg_src_node_embeddings_ls,dim=0)
                    batch_neg_dst_node_embeddings_ls = torch.cat(batch_neg_dst_node_embeddings_ls,dim=0)
                
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                
                #continue
                positive_probabilities = model[1](input_1=batch_src_node_embeddings_ls, input_2=batch_dst_node_embeddings_ls).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings_ls, input_2=batch_neg_dst_node_embeddings_ls).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                assert args.factor >= 0 and args.factor <= 1, "factor should be between 0 and 1"
                loss = args.factor*loss_func(input=predicts, target=labels) + (1-args.factor)*(ssl_pos_loss+ssl_neg_loss)

                train_losses.append(loss.item())
                #print(logits.shape)
                train_metrics.append(get_link_prediction_metrics_original(predicts=predicts.view(-1,1), labels=labels.view(-1,1).float()))
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                   
                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                #break
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                       model[0].memory_bank.detach_memory_bank()
                
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
            torch.save(model.state_dict(), save_path+f'save_model_{epoch}.pkl')
            print(save_path+f'save_model_{epoch}.pkl')
            
            continue
            
    sys.exit()
