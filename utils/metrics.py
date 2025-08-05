import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import average_precision_score, roc_auc_score

def get_regression_metrics(pos_predicts: torch.Tensor, neg_predicts: torch.Tensor,
                           pos_labels: torch.Tensor, neg_labels: torch.Tensor):
    """
    Get metrics for the regression task
    :param pos_predicts: Tensor, shape (num_pos_samples,)
    :param neg_predicts: Tensor, shape (num_neg_samples,)
    :param pos_labels: Tensor, shape (num_pos_samples,)
    :param neg_labels: Tensor, shape (num_neg_samples,)
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    pos_predicts = pos_predicts.cpu().detach().numpy()
    neg_predicts = neg_predicts.cpu().detach().numpy()
    pos_labels = pos_labels.cpu().numpy()
    neg_labels = neg_labels.cpu().numpy()

    all_predicts = np.concatenate([pos_predicts, neg_predicts])
    all_labels = np.concatenate([pos_labels, neg_labels])

    rmse = np.sqrt(mean_squared_error(all_labels, all_predicts))
    mae = mean_absolute_error(all_labels, all_predicts)
    r2 = r2_score(all_labels, all_predicts)

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def get_link_prediction_metrics_original(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_link_prediction_metrics(predicts_time: torch.Tensor, time_diffs: torch.Tensor):
    intervals = torch.arange(0.5, 100.5, 0.5) / 100 
    metrics = {}

    for interval in intervals:
        delta = interval * time_diffs
        corrects = (torch.abs(predicts_time - time_diffs) < delta).sum()
        eval_metric = corrects.float() / len(predicts_time)
        metrics[f'diff_{int(interval * 1000)}'] = eval_metric 

    return metrics


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
