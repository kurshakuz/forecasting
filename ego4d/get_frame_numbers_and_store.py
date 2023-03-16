import json
import os

# load the JSON data from a file or API response
data = json.load(open("./fho_hands_train_contact.json"))

for clip in data["clips"]:
    clip_uid = clip["clip_uid"]
    clip_id = clip["clip_id"]
    for frame in clip["frames"]:
        frames = []
        pre_45_frame = frame["pre_45"]["clip_frame"]
        # get all 60 values after last_observable frame or before pre_frame
        for i in range(pre_45_frame - 60, pre_45_frame):
            frames.append(i)
        try:
            frames.append(pre_45_frame)
            frames.append(frame["pre_30"]["clip_frame"])
            frames.append(frame["pre_15"]["clip_frame"])
            frames.append(frame["pre_frame"]["clip_frame"])
        except:
            print(f"missing annotations on {clip_uid}_{clip_id}")
            continue
        frames.append(frame["contact_frame"]["clip_frame"])


        file_name = f"{clip_id}_{pre_45_frame-1}.txt"
        # print(file_name)

        if not os.path.exists(f"../thesis-ego4d/extracted_frames/{clip_uid}"):
            os.makedirs(f"../thesis-ego4d/extracted_frames/{clip_uid}")

        with open(f"../thesis-ego4d/extracted_frames/{clip_uid}/{file_name}", "w") as f:
            f.write('\n'.join(map(str, frames)))
