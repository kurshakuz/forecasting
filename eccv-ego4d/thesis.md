# ECCV Solution workspace

## Update json and model paths

Open `modeling_finetune_uniformer_ego4d.py` and modify the line `497` and change the `model_folder` variable.

Open `datasets.py` and modify lines `22-23` and change path variables.

```
mkdir -p /workspace/thesis-ego4d/eccv-models/
```

Download pre-trained model for training:
```
wget https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_verb_uniformer_base_16x320_k600_ep9.pt -P /workspace/thesis-ego4d/eccv-models
```

Download trained model for evaluation:
```
wget https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_fhp_uniformer8x320.pth -P /workspace/thesis-ego4d/eccv-models
```

test -e thesis-ws || git clone https://github.com/kurshakuz/thesis-ws.git
cd /workspace/thesis-ws/
git pull

pip install -r /workspace/thesis-ws/eccv-ego4d/requirements.txt


gdown --no-cookies 1ozcEiH9g_utlwl60Gu3iHYnxnmf7slxP

## Run evaluation script
Local:
```bash
python3 run_ego4d_hands.py \
--model uniformer_base_256_ego4d_finetune \
--nb_verb_classes 20 \
--nb_noun_classes 0 \
--data_set ego4d_hands \
--data_path ./dummy \
--log_dir ./workdir/ego4d_hands_uniformer_base \
--output_dir ./workdir/ego4d_hands_uniformer_base \
--batch_size 16 \
--num_sample 1 \
--num_segments 8 \
--finetune /home/dev/workspace/thesis-ego4d/eccv-models/ego4d_fhp_uniformer8x320.pth \
--warmup_epochs  1 \
--input_size 320 \
--short_side_size 320 \
--save_ckpt_freq 1 \
--num_frames 16 \
--layer_decay 1. \
--opt adamw \
--no_auto_resume \
--lr 1e-3 \
--opt_betas 0.9 0.999 \
--weight_decay 0.05 \
--epochs 10 \
--eval \
--test_num_segment 1 \
--test_num_crop 1 \
--test_subset val \
```

Remote:
```bash
!python3 /workspace/thesis-ws/eccv-ego4d/run_ego4d_hands.py \
    --model uniformer_base_256_ego4d_finetune \
    --nb_verb_classes 20 \
    --nb_noun_classes 0 \
    --data_set ego4d_hands \
    --data_path ./dummy \
    --log_dir ./workdir/ego4d_hands_uniformer_base \
    --output_dir ./workdir/ego4d_hands_uniformer_base \
    --batch_size 64 \
    --num_sample 1 \
    --num_segments 8 \
    --finetune /workspace/thesis-ego4d/eccv-models/ego4d_fhp_uniformer8x320.pth \
    --warmup_epochs  1 \
    --input_size 320 \
    --short_side_size 320 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --layer_decay 1. \
    --opt adamw \
    --no_auto_resume \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 10 \
    --eval \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --test_subset val
```

Training
```bash
python3  /workspace/thesis-ws/eccv-ego4d/run_ego4d_hands.py \
    --model uniformer_base_256_ego4d_finetune \
    --nb_verb_classes 20 \
    --nb_noun_classes 0 \
    --data_set ego4d_hands \
    --data_path ./dummy \
    --log_dir ./workdir/ego4d_hands_uniformer_base \
    --output_dir ./workdir/ego4d_hands_uniformer_base \
    --batch_size 64 \
    --num_segments 8 \
    --num_sample 1 \
    --warmup_epochs  5 \
    --input_size 320 \
    --short_side_size 320 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --layer_decay 1. \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 30 \
    --test_num_segment 2 \
    --test_num_crop 3 \
```