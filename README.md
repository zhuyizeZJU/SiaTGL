# SiaTGL
SiaTGL (Sampling Interval-Aware Temporal Graph Learning) is a novel framework that learns robust representations from dynamic graphs with varying time scales. It is the first model to explicitly incorporate the sampling interval as a core feature. Through its unique Sampling Interval Alignment (SIA) module, SiaTGL uses a self-supervised task on multi-granularity data to capture true evolutionary dynamics. This enables state-of-the-art generalization to different, even unseen, temporal intervals without requiring fine-tuning.

## Usage

### Training
To train the model, use `train_link_prediction.py`:

```bash
python3 train_link_prediction.py \
    --pretrainTestDataset {dataset} \
    --model_name SiaTGL \
    --patch_size 2 \
    --num_runs 3 \
    --gpu 0 \
    --negative_sample_strategy inductive \
    --time_feat_dim 172 \
    --factor 0.5 \
    --interval 0
```

### Fine-tuning
For fine-tuning, use `finetune_link_prediction.py`:

```bash
python3 finetune_link_prediction.py \
    --pretrainTestDataset {dataset} \
    --model_name SiaTGL \
    --patch_size 2 \
    --num_runs 3 \
    --gpu 0 \
    --negative_sample_strategy inductive \
    --time_feat_dim 172 \
    --factor 0.5 \
    --interval 3
```

### Evaluation
To evaluate the model, use `evaluate_link_prediction.py`:

```bash
python3 evaluate_link_prediction.py \
    --pretrainTestDataset {dataset} \
    --model_name SiaTGL \
    --patch_size 2 \
    --num_runs 3 \
    --gpu 0 \
    --negative_sample_strategy inductive \
    --time_feat_dim 172 \
    --factor 0.5 \
    --interval 3 \
    --testInterval 3
```
