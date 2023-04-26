## Ego4D FHP Thesis 

## Data preparation

### Prepare filtered list of clips to download
Run `python3 extract_all_uids.py` and change the `data_split` to `train`, `val`, `test_unannotated` accordingly to extract required clip uids.

### Prepare frame numbers for each clip to be extracted.
Run `python3 get_frame_numbers_and_store.py` for each split.

### Download according clips
Run `sh ./download_ego4d.sh` to download the required clips.

### Extracted interaction clips
Run `python3 frame_extractor.py` according to the required data split.

## Trainig

### Configure workspace

```shell
conda create --name slowfast4
```

```shell
conda activate slowfast4
```

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install fvcore
pip install simplejson
pip install av
pip install iopath
pip install psutil
pip install opencv-python
pip install tensorboard
pip install moviepy
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
```

```shell
git clone https://github.com/facebookresearch/slowfast
cd slowfast
python3 setup.py build develop
export PYTHONPATH=$WRK/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/slowfast:$PYTHONPATH
```

### Start training

Navigate to the FHP folder:
```shell
cd forecasting/Ego4D-Future-Hand-Prediction/
```

Run the following command to start training.
```shell
python3 tools/run_net.py --cfg $WRK/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50.yaml OUTPUT_DIR $WRK/thesis-ego4d/results/output/
```

Change values in the `I3D_8x8_R50.yaml` depending on the data location:
```
Data path: /media/dev/HIKVISION/ego4d_data
```

### Evaluation and Visualization:
```
python3 tools/evaluation.py --cfg $WRK/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50_vast_3090_eval.yaml
```

### Generate test result and submission
```
python3 /workspace/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/tools/run_net.py --cfg /workspace/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50_vast_3090_eval.yaml
```
```
python3 /workspace/thesis-ws/ego4d/forecasting/Ego4D-Future-Hand-Prediction/tools/generate_submission.py --num_clips 30 --output_file ./output.pkl
```