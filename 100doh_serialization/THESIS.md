# 100DoH trajectory annotation pipeline

## Setup workspace
```shell
conda activate ek100hands
python setup.py develop
```

## Test serialization setup
```shell
python3 tests/test_serialisation.py 
```

## Configure Hand Object detector requirements
1. Follow installation on https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes#library
2. Follow installation on https://github.com/ddshan/hand_object_detector#prerequisites

## Capture frames from the video sequence
```shell
python3 capture_frames.py
```

## Annotate hand trajectory and serializer the results
```shell
python3 detector_serialization.py --cuda --checkpoint=132028
```
