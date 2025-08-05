import argparse
import sys
import torch

def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--trainDataset', type=str, help='dataset train on ', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts','review','nearby','dgraphfin','mathoverflow','bitcoin','college','ubuntu','email','wiki-talk','superuser','dppi'])
    parser.add_argument('--pretrainTestDataset', type=str, help='dataset train on ', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts','review','nearby','dgraphfin','mathoverflow','bitcoin','college','ubuntu','email','wiki-talk','superuser','dppi'])
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts','review','nearby','dgraphfin','mathoverflow','bitcoin','college','ubuntu','email','wiki-talk','superuser','dppi'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size baseline')
    parser.add_argument('--ssl_split_n', type=int, default=2, help='ssl_split_n')
    parser.add_argument('--factor', type=float, default=1, help='factor')       
    parser.add_argument('--interval', type=str, default='day', help='interval')
    parser.add_argument('--testInterval', type=str, default='day', help='testInterval')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL','FreeDyG', 'GraphMixer', 'DyGFormer','FreeDyG','SiaTGL'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=256, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=128, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=256, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=2, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--test', default='single_loss')
    parser.add_argument('--alpha', type=float,default=0.0)
    parser.add_argument('--time_shuffle', type=int,default=1)
    parser.add_argument('--use_gpt', type=int,default=1)
    parser.add_argument('--is_normalize', type=int,default=1)
    parser.add_argument('--seq_num', type=int,default=120)
    parser.add_argument('--data_ratio', type=float,default=1.0)
    parser.add_argument('--domain_num', type=int,default=6)
    parser.add_argument('--multi_task', type=int,default=1)

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()


    return args
