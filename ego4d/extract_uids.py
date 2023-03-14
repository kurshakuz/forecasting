import json

# load the JSON data from a file or API response
data = json.load(open("../content/ego4d_data/v1/annotations/fho_hands_train.json"))

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
    for frame in clip["frames"]:
        if "contact_frame" in frame:
            new_clip = {
                "clip_id": clip["clip_id"],
                "clip_uid": clip["clip_uid"],
                "video_uid": clip["video_uid"],
                "frames": [],
            }
            new_clip["frames"].append(frame)

            video_uids.add(clip["video_uid"])
            clip_uids.add(clip["clip_uid"])
    fho_data["clips"].append(new_clip)
    # break

# save the extracted frames
with open("./fho_hands_train_contact.json", "w") as f:
    json.dump(fho_data, f, indent=2)

# store uids string
with open("./fho_hands_train_contact_video_uids.txt", "w") as f:
    f.write(' '.join(video_uids))

with open("./fho_hands_train_contact_clip_uids.txt", "w") as f:
    f.write(' '.join(clip_uids))

# print(uids)
# json.dumps(contact_clips, indent=2)
