import json
import os

split = "train" # train, val, test_unannotated

# load the JSON data from a file or API response
data = json.load(open(f"./extracted_files/fho_hands_{split}_all.json"))
path_to_store = f"/home/dev/workspace/thesis-ego4d/{split}_data/extracted_frame_nums"

for clip in data["clips"]:
    clip_uid = clip["clip_uid"]
    clip_id = clip["clip_id"]
    for frame in clip["frames"]:
        frames = []
        pre_45_frame = frame["pre_45"]["frame"]
        # get all 60 values after last_observable frame or before pre_frame
        for i in range(pre_45_frame - 60, pre_45_frame):
            frames.append(i)

        # extract key frames
        try:
            frames.append(pre_45_frame)
        except:
            # print(f"missing annotations on {clip_uid}_{clip_id}")
            pass
        try:
            frames.append(frame["pre_30"]["frame"])
        except:
            pass
        try:
            frames.append(frame["pre_15"]["frame"])
        except:
            pass
        try:
            frames.append(frame["pre_frame"]["frame"])
        except:
            pass
        try:
            frames.append(frame["contact_frame"]["frame"])
        except:
            pass

        file_name = f"{clip_id}_{pre_45_frame-1}.txt"
        # print(file_name)

        if not os.path.exists(f"{path_to_store}/{clip_uid}"):
            os.makedirs(f"{path_to_store}/{clip_uid}")

        with open(f"{path_to_store}/{clip_uid}/{file_name}", "w") as f:
            f.write('\n'.join(map(str, frames)))
