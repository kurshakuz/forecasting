conda activate ek100hands
python setup.py develop
python3 tests/test_serialisation.py 



1. follow installation on https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes#library
2. follow installation on https://github.com/ddshan/hand_object_detector#prerequisites

conda activate thesis-annot


python3 capture_frames.py
python3 detector_serialization.py --cuda --checkpoint=132028