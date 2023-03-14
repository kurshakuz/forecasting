import json

# load the JSON data from a file or API response
data = json.load(open("../content/ego4d_data/v1/annotations/fho_hands_train.json"))

# extract objects that contain "contact_frame" instance
fho_data = {
    "version": data["version"],
    "date": data["date"],
    "description": data["description"],
    "manifest": data["manifest"],
    "split": data["split"],
    "clips": [],
}
for clip in data["clips"]:
    new_clip = {
        "clip_id": clip["clip_id"],
        "clip_uid": clip["clip_uid"],
        "video_uid": clip["video_uid"],
        "frames": [],
    }
    for frame in clip["frames"]:
        if "contact_frame" in frame:
            new_clip["frames"].append(frame)
    fho_data["clips"].append(new_clip)
    # break

# save the extracted frames
with open("./fho_hands_train_contact.json", "w") as f:
    json.dump(fho_data, f, indent=2)
# json.dumps(contact_clips, indent=2)
