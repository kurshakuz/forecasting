# Ego4D FHP Thesis Preparation

## Prepare filtered list of clips to download
Run `python3 extract_all_uids.py` and change the `data_split` to `train`, `val`, `test_unannotated` accordingly to extract required clip uids.

## Prepare frame numbers for each clip to be extracted.
Run `python3 get_frame_numbers_and_store.py` for each split.

## Download according clips
Run `sh ./download_ego4d.sh` to download the required clips.

## Extracted interaction clips
Run `python3 frame_extractor.py` according to the required data split.
