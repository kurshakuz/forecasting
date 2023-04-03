import json

# set data split here
data_split = "test_unannotated" # "train" or "val"

# load the JSON data from a file or API response
data = json.load(open(f"/home/dev/workspace/thesis-ego4d/train_data/content/ego4d_data/v1/annotations/fho_hands_{data_split}.json"))

# extract objects that contain "contact_frame" instance
video_uids = set()
clip_uids = set()
fho_data = {
    "version": data["version"],
    "date": data["date"],
    "description": data["description"],
    "manifest": data["manifest"],
    "split": data["split"],
    "clips": [],
}
for clip in data["clips"]:
    contact_frames = []
    for frame in clip["frames"]:
        contact_frames.append(frame)
        video_uids.add(clip["video_uid"])
        clip_uids.add(clip["clip_uid"])
    if contact_frames != []:
        new_clip = {
            "clip_id": clip["clip_id"],
            "clip_uid": clip["clip_uid"],
            "video_uid": clip["video_uid"],
            "frames": contact_frames,
        }
        fho_data["clips"].append(new_clip)

# save the extracted frames
with open(f"./fho_hands_{data_split}_all.json", "w") as f:
    json.dump(fho_data, f, indent=2)

# store uids string
with open(f"./fho_hands_{data_split}_all_video_uids.txt", "w") as f:
    video_uids = list(video_uids)
    video_uids = sorted(video_uids)
    f.write(' '.join(video_uids))

with open(f"./fho_hands_{data_split}_all_clip_uids.txt", "w") as f:
    clip_uids = list(clip_uids)
    clip_uids = sorted(clip_uids)
    f.write(' '.join(clip_uids))
