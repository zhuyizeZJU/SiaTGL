import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class SiaTGL(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu',interval=0,ssl_split_n: int =2):
        
        super(SiaTGL, self).__init__()
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.interval = int(interval)
        self.ssl_split_n = ssl_split_n

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.sampleInterval_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'sampleInterval': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })
        self.num_channels = 5

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)
        self.embeddingPredictor = EmbeddingPredictor(input_dim=self.node_feat_dim) 
        self.criterion = nn.MSELoss()
        

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,node_sampleInterval:np.ndarray):

        # get neighbors
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list,src_nodes_neighbor_sampleInterval_list= \
            self.neighbor_sampler.get_all_first_hop_neighbors_sampleInterval(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list,dst_nodes_neighbor_sampleInterval_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors_sampleInterval(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # padding
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times ,src_padded_nodes_neighbor_sampleInterval,src_sequence_lengths= \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times,node_sampleInterval = node_sampleInterval, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               nodes_neighbor_sampleInterval_list = src_nodes_neighbor_sampleInterval_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times ,dst_padded_nodes_neighbor_sampleInterval,dst_sequence_lengths= \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, node_sampleInterval = node_sampleInterval,nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               nodes_neighbor_sampleInterval_list = dst_nodes_neighbor_sampleInterval_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # co-occurrence features
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)
        # basic features
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features,src_padded_nodes_neighbor_sampleInterval_feature = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              padded_nodes_neighbor_sampleInterval=src_padded_nodes_neighbor_sampleInterval,
                              time_encoder=self.time_encoder,sampleInterval_encoder=self.sampleInterval_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features,dst_padded_nodes_neighbor_sampleInterval_feature = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, 
                              padded_nodes_neighbor_sampleInterval=dst_padded_nodes_neighbor_sampleInterval,
                              time_encoder=self.time_encoder,sampleInterval_encoder=self.sampleInterval_encoder)
        # patching
        src_patches_main = self.get_patches(
            padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
            padded_nodes_neighbor_sampleInterval_feature=src_padded_nodes_neighbor_sampleInterval_feature,
            padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
            patch_size=self.patch_size)

        dst_patches_main = self.get_patches(
            padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
            padded_nodes_neighbor_sampleInterval_feature=dst_padded_nodes_neighbor_sampleInterval_feature,
            padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
            patch_size=self.patch_size)
        
        # projection
        src_projected_patches = [self.projection_layer[name](patches) for name, patches in zip(['node', 'edge', 'time', 'sampleInterval', 'neighbor_co_occurrence'], src_patches_main)]
        dst_projected_patches = [self.projection_layer[name](patches) for name, patches in zip(['node', 'edge', 'time', 'sampleInterval', 'neighbor_co_occurrence'], dst_patches_main)]

        # encoding
        src_node_embeddings, dst_node_embeddings = self.encode_patches(src_projected_patches, dst_projected_patches)


        # SSL
        src_node_features_list, src_edge_features_list, src_time_features_list, src_interval_features_list = \
            self.get_features_list_optimized(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                             padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                                             padded_nodes_neighbor_sampleInterval=src_padded_nodes_neighbor_sampleInterval,
                                             sequence_lengths=src_sequence_lengths, ssl_split_n=self.ssl_split_n)

        dst_node_features_list, dst_edge_features_list, dst_time_features_list, dst_interval_features_list = \
            self.get_features_list_optimized(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                                             padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                                             padded_nodes_neighbor_sampleInterval=dst_padded_nodes_neighbor_sampleInterval,
                                             sequence_lengths=dst_sequence_lengths, ssl_split_n=self.ssl_split_n)

        # patching
        src_patches_lists = self.get_patches_list(
            [src_node_features_list, src_edge_features_list, src_time_features_list, src_interval_features_list], self.patch_size)
        
        dst_patches_lists = self.get_patches_list(
            [dst_node_features_list, dst_edge_features_list, dst_time_features_list, dst_interval_features_list], self.patch_size)

        src_node_embeddings_list, dst_node_embeddings_list = self.process_patches_list_optimized(src_patches_lists, dst_patches_lists)

        # ssl_loss
        ssl_loss_src = self.embedding_prediction_loss(src_node_embeddings_list)
        ssl_loss_dst = self.embedding_prediction_loss(dst_node_embeddings_list)
        ssl_loss = ssl_loss_src + ssl_loss_dst
        
        return src_node_embeddings, dst_node_embeddings, ssl_loss

    def encode_patches(self, src_projected_patches: list, dst_projected_patches: list):

        batch_size = src_projected_patches[0].shape[0]
        src_num_patches = src_projected_patches[0].shape[1]
        dst_num_patches = dst_projected_patches[0].shape[1]

        # concat patch
        patches_data = [torch.cat([src_patch, dst_patch], dim=1) for src_patch, dst_patch in zip(src_projected_patches, dst_projected_patches)]
        
        patches_data = torch.stack(patches_data, dim=2)
        
        # Reshape for Transformer
        patches_data = patches_data.view(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        src_patches_data = patches_data[:, :src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches:, :]

        # pooling
        src_agg_patches = torch.mean(src_patches_data, dim=1)
        dst_agg_patches = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_agg_patches)
        dst_node_embeddings = self.output_layer(dst_agg_patches)

        return src_node_embeddings, dst_node_embeddings

    def embedding_prediction_loss(self, embedding_list: list):

        if not embedding_list or not embedding_list[0]:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0
        num_predictions = 0
        
        for single_granularity_embeddings in embedding_list:
            for i in range(len(single_granularity_embeddings) - 1):
                # detach target to prevent gradients from flowing back through the target
                input_emb = single_granularity_embeddings[i]
                target_emb = single_granularity_embeddings[i+1].detach() 
                
                pred_emb = self.embeddingPredictor(input_emb)
                total_loss += self.criterion(pred_emb, target_emb)
                num_predictions += 1
        
        return total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0, device=self.device)

    def process_patches_list_optimized(self, src_patches_lists: list, dst_patches_lists: list):

        src_node_embeddings_list, dst_node_embeddings_list = [], []

        num_granularities = len(src_patches_lists[0])
        if num_granularities == 0:
            return [], []
        num_timesteps = len(src_patches_lists[0][0])
        if num_timesteps == 0:
            return [[] for _ in range(num_granularities)], [[] for _ in range(num_granularities)]
        
        # [node_list, edge_list, time_list, interval_list]
        all_src_patches_cat = [torch.cat([p for sublist in feature_type for p in sublist], dim=0) for feature_type in src_patches_lists]
        all_dst_patches_cat = [torch.cat([p for sublist in feature_type for p in sublist], dim=0) for feature_type in dst_patches_lists]

        src_dims = [(p.shape[0], p.shape[1]) for sublist in src_patches_lists[0] for p in sublist]
        dst_dims = [(p.shape[0], p.shape[1]) for sublist in dst_patches_lists[0] for p in sublist]

        src_projected = [self.projection_layer[name](patches) for name, patches in zip(['node', 'edge', 'time', 'sampleInterval'], all_src_patches_cat)]
        dst_projected = [self.projection_layer[name](patches) for name, patches in zip(['node', 'edge', 'time', 'sampleInterval'], all_dst_patches_cat)]
        
        total_src_b_x_p = sum(b * p for b, p in src_dims)
        total_dst_b_x_p = sum(b * p for b, p in dst_dims)
        zero_co_occurrence_src = torch.zeros(total_src_b_x_p, self.channel_embedding_dim, device=self.device)
        zero_co_occurrence_dst = torch.zeros(total_dst_b_x_p, self.channel_embedding_dim, device=self.device)
        
        src_projected_reshaped = [p.view(-1, self.channel_embedding_dim) for p in src_projected]
        dst_projected_reshaped = [p.view(-1, self.channel_embedding_dim) for p in dst_projected]
        
        # concate all features
        src_combined = torch.cat(src_projected_reshaped + [zero_co_occurrence_src], dim=1)
        dst_combined = torch.cat(dst_projected_reshaped + [zero_co_occurrence_dst], dim=1)
        
        src_combined = src_combined.view(-1, self.num_channels, self.channel_embedding_dim)
        dst_combined = dst_combined.view(-1, self.num_channels, self.channel_embedding_dim)
        
        combined_data = []
        src_start, dst_start = 0, 0
        for (b_s, p_s), (b_d, p_d) in zip(src_dims, dst_dims):
            combined_data.append(src_combined[src_start : src_start + b_s * p_s].view(b_s, p_s, self.num_channels, self.channel_embedding_dim))
            combined_data.append(dst_combined[dst_start : dst_start + b_d * p_d].view(b_d, p_d, self.num_channels, self.channel_embedding_dim))
            src_start += b_s * p_s
            dst_start += b_d * p_d
            
        final_patches = torch.cat([torch.cat(t, dim=1) for t in zip(combined_data[0::2], combined_data[1::2])], dim=0)
        final_patches = final_patches.view(final_patches.shape[0], final_patches.shape[1], -1)

        encoded_patches = final_patches
        for transformer in self.transformers:
            encoded_patches = transformer(encoded_patches)
            
        results = []
        current_idx = 0
        for (b, p_s), (_, p_d) in zip(src_dims, dst_dims):
            chunk = encoded_patches[current_idx:current_idx+b, :, :]
            src_p = chunk[:, :p_s, :].mean(dim=1)
            dst_p = chunk[:, p_s:p_s+p_d, :].mean(dim=1)
            results.append((self.output_layer(src_p), self.output_layer(dst_p)))
            current_idx += b

        result_idx = 0
        for i in range(num_granularities):
            src_inner_list, dst_inner_list = [], []
            for j in range(num_timesteps):
                src_emb, dst_emb = results[result_idx]
                src_inner_list.append(src_emb)
                dst_inner_list.append(dst_emb)
                result_idx += 1
            src_node_embeddings_list.append(src_inner_list)
            dst_node_embeddings_list.append(dst_inner_list)
            
        return src_node_embeddings_list, dst_node_embeddings_list

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, padded_nodes_neighbor_sampleInterval:np.ndarray,
                     time_encoder: TimeEncoder, sampleInterval_encoder:TimeEncoder):
        padded_nodes_neighbor_ids_ts = torch.from_numpy(padded_nodes_neighbor_ids).long().to(self.device)
        padded_nodes_edge_ids_ts = torch.from_numpy(padded_nodes_edge_ids).long().to(self.device)
        node_interact_times_ts = torch.from_numpy(node_interact_times).float().to(self.device)
        padded_nodes_neighbor_times_ts = torch.from_numpy(padded_nodes_neighbor_times).float().to(self.device)
        padded_nodes_neighbor_sampleInterval_ts = torch.from_numpy(padded_nodes_neighbor_sampleInterval).float().to(self.device)

        node_features = self.node_raw_features[padded_nodes_neighbor_ids_ts]
        edge_features = self.edge_raw_features[padded_nodes_edge_ids_ts]
        
        # time features
        time_diff = node_interact_times_ts.unsqueeze(1) - padded_nodes_neighbor_times_ts
        time_features = time_encoder(time_diff)
        
        # interval features
        interval_features = sampleInterval_encoder(padded_nodes_neighbor_sampleInterval_ts)
        
        # padding
        padding_mask = (padded_nodes_neighbor_ids_ts == 0)
        time_features[padding_mask] = 0.0
        interval_features[padding_mask] = 0.0

        return node_features, edge_features, time_features, interval_features

    def get_features_list_optimized(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, padded_nodes_neighbor_sampleInterval:np.ndarray, sequence_lengths: np.ndarray, ssl_split_n:int = 2):
        device = self.device
        padded_ids_ts = torch.from_numpy(padded_nodes_neighbor_ids).long().to(device)
        padded_edge_ids_ts = torch.from_numpy(padded_nodes_edge_ids).long().to(device)
        padded_interval_ts = torch.from_numpy(padded_nodes_neighbor_sampleInterval).float().to(device)
        interact_times_ts = torch.from_numpy(node_interact_times).float().to(device).unsqueeze(1)
        neighbor_times_ts = torch.from_numpy(padded_nodes_neighbor_times).float().to(device)
        seq_lengths_ts = torch.from_numpy(sequence_lengths).float().to(device)

        # mask
        batch_size, max_seq_len = padded_ids_ts.shape
        masks = []
        split_indices = torch.arange(1, ssl_split_n + 1, device=device).float() / ssl_split_n
        split_points = torch.ceil(seq_lengths_ts.view(-1, 1) * split_indices.view(1, -1)).long()
        
        for i in range(ssl_split_n):
            mask = torch.arange(max_seq_len, device=device).expand(batch_size, -1) < split_points[:, i].unsqueeze(1)
            masks.append(mask)

        max_interval = 3
        node_features_list, edge_features_list, time_features_list, interval_features_list = [], [], [], []
        padding_mask = (padded_ids_ts == 0)

        for i in range(max_interval + 1 - self.interval):
            inner_node, inner_edge = [], []
            for mask in masks:
                inner_node.append(self.node_raw_features[padded_ids_ts * mask])
                inner_edge.append(self.edge_raw_features[padded_edge_ids_ts * mask])
            node_features_list.append(inner_node)
            edge_features_list.append(inner_edge)

        time_diff_base = interact_times_ts - neighbor_times_ts
        
        inner_time = []
        for mask in masks:
            time_feat = self.time_encoder(time_diff_base * mask.float())
            time_feat[padding_mask] = 0.0
            inner_time.append(time_feat)
        time_features_list.append(inner_time)

        for i in range(self.interval + 1, max_interval + 1):
            inner_time = []
            downsampled_time = torch.floor(time_diff_base / (10**i))
            for mask in masks:
                time_feat = self.time_encoder(downsampled_time * mask.float())
                time_feat[padding_mask] = 0.0
                inner_time.append(time_feat)
            time_features_list.append(inner_time)
            
        for i in range(0, max_interval + 1 - self.interval):
            inner_interval = []
            scaled_interval = padded_interval_ts * (10**i)
            for mask in masks:
                interval_feat = self.sampleInterval_encoder(scaled_interval * mask.float())
                interval_feat[padding_mask] = 0.0 
                inner_interval.append(interval_feat)
            interval_features_list.append(inner_interval)

        return node_features_list, edge_features_list, time_features_list, interval_features_list

    def get_patches_list(self, features_lists: list, patch_size: int = 1):
        all_patches_lists = []
        
        feat_dims = [self.node_feat_dim, self.edge_feat_dim, self.time_feat_dim, self.time_feat_dim]
        
        for feat_list, feat_dim in zip(features_lists, feat_dims):
            patches_list_for_feature_type = []
            for inner_list in feat_list:
                inner_patches = []
                for features_tensor in inner_list:
                    batch_size, max_seq_length, _ = features_tensor.shape
                    assert max_seq_length % patch_size == 0, f"Sequence length {max_seq_length} is not divisible by patch size {patch_size}"
                    num_patches = max_seq_length // patch_size
                    
                    # Reshape and transpose to create patches efficiently
                    patches = features_tensor.view(batch_size, num_patches, patch_size, feat_dim)
                    patches = patches.view(batch_size, num_patches, patch_size * feat_dim)
                    inner_patches.append(patches)
                patches_list_for_feature_type.append(inner_patches)
            all_patches_lists.append(patches_list_for_feature_type)
            
        return all_patches_lists

    def get_patches(self, *, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_sampleInterval_feature:torch.Tensor,
                    padded_nodes_neighbor_co_occurrence_features: torch.Tensor, patch_size: int = 1):
        features = [
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
            padded_nodes_neighbor_sampleInterval_feature,
            padded_nodes_neighbor_co_occurrence_features
        ]
        
        feat_dims = [
            self.node_feat_dim, self.edge_feat_dim, self.time_feat_dim,
            self.time_feat_dim, self.neighbor_co_occurrence_feat_dim
        ]

        output_patches = []
        for tensor, feat_dim in zip(features, feat_dims):
            batch_size, max_seq_length, _ = tensor.shape
            assert max_seq_length % patch_size == 0
            num_patches = max_seq_length // patch_size
            
            # Efficiently reshape to get patches
            patches = tensor.view(batch_size, num_patches, patch_size * feat_dim)
            output_patches.append(patches)
            
        return tuple(output_patches)

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, 
                    node_sampleInterval: np.ndarray, nodes_neighbor_ids_list: list, 
                    nodes_edge_ids_list: list, nodes_neighbor_times_list: list,
                    nodes_neighbor_sampleInterval_list: list, patch_size: int = 1, 
                    max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors should be greater than 1!'
        
        for idx in range(len(nodes_neighbor_ids_list)):
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                start_idx = len(nodes_neighbor_ids_list[idx]) - (max_input_sequence_length - 1)
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][start_idx:]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][start_idx:]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][start_idx:]
                nodes_neighbor_sampleInterval_list[idx] = nodes_neighbor_sampleInterval_list[idx][start_idx:]
        
        sequence_lengths = np.array([len(neighbors) + 1 for neighbors in nodes_neighbor_ids_list])
        max_seq_length = sequence_lengths.max()

        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        
        assert max_seq_length % patch_size == 0

        batch_size = len(node_ids)
        padded_nodes_neighbor_ids = np.zeros((batch_size, max_seq_length), dtype=np.longlong)
        padded_nodes_edge_ids = np.zeros((batch_size, max_seq_length), dtype=np.longlong)
        padded_nodes_neighbor_times = np.zeros((batch_size, max_seq_length), dtype=np.float32)
        padded_nodes_neighbor_sampleInterval = np.zeros((batch_size, max_seq_length), dtype=np.float32)
        
        for idx, length in enumerate(sequence_lengths):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            padded_nodes_neighbor_sampleInterval[idx, 0] = node_sampleInterval[idx]

            if length > 1:
                neighbors_len = length - 1
                padded_nodes_neighbor_ids[idx, 1:length] = nodes_neighbor_ids_list[idx][:neighbors_len]
                padded_nodes_edge_ids[idx, 1:length] = nodes_edge_ids_list[idx][:neighbors_len]
                padded_nodes_neighbor_times[idx, 1:length] = nodes_neighbor_times_list[idx][:neighbors_len]
                padded_nodes_neighbor_sampleInterval[idx, 1:length] = nodes_neighbor_sampleInterval_list[idx][:neighbors_len]

        return (padded_nodes_neighbor_ids, padded_nodes_edge_ids, 
                padded_nodes_neighbor_times, padded_nodes_neighbor_sampleInterval, 
                sequence_lengths)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):
    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            src_padded_nodes_appearances.append(torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            dst_padded_nodes_appearances.append(torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))
        
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0
        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        transposed_inputs = inputs.transpose(0, 1)
        norm_inputs = self.norm_layers[0](transposed_inputs)
        hidden_states, _ = self.multi_head_attention(query=norm_inputs, key=norm_inputs, value=norm_inputs)
        hidden_states = hidden_states.transpose(0, 1)
        outputs = inputs + self.dropout(hidden_states)
        norm_outputs = self.norm_layers[1](outputs)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](norm_outputs))))
        outputs = outputs + self.dropout(hidden_states)
        return outputs
    

class EmbeddingPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.model(x)