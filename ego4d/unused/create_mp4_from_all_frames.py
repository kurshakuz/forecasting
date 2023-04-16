import os
import cv2


split = "val" # train, val, test_unannotated
folder_path = f"/home/dev/workspace/thesis-ego4d/{split}_data/extracted_frame_nums"
video_path_prefix = f"/media/dev/HIKVISION/val_data/extracted_videos"
downscale = False

for clip_uid_folder in os.listdir(folder_path):
    clip_uid_folder_path = os.path.join(folder_path, clip_uid_folder)
    if os.path.isdir(clip_uid_folder_path):
        for txt_file in os.listdir(clip_uid_folder_path):
            txt_file_path = os.path.join(clip_uid_folder_path, txt_file)
            if os.path.isfile(txt_file_path):
                # print(f"Content of {txt_file_path}")
                # print(os.path.basename(os.path.dirname(clip_uid_folder_path)))
                clip_uid = os.path.basename(clip_uid_folder_path)
                clip_id = os.path.splitext(txt_file)[0]
                if downscale:
                    frames_dir = f"/media/dev/HIKVISION/val_data/extracted_all_dwn_frames/{clip_uid}"
                else:
                    frames_dir = f"/media/dev/HIKVISION/val_data/extracted_all_frames/{clip_uid}"
                with open(txt_file_path, "r") as f:
                    frames = [int(line.strip()) for line in f.readlines()]
                # print(frames)
                # iterate over the frame numbers and extract each frame from the video
                video_name = f"{video_path_prefix}/{clip_id}.mp4"
                # os.makedirs(video_path_prefix, exist_ok=True)

                if os.path.exists(video_name):
                    continue
                print(f"Creating video {video_name}...")

                # print(video_name)

                # print(os.listdir(frames_dir))

                images = [img for img in os.listdir(frames_dir)
                        if img.endswith(".jpg")]
                # print(images)

                image_name = f"frame_{0}.jpg"
                img = cv2.imread(os.path.join(frames_dir, image_name))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_name, fourcc, 30, (img.shape[1], img.shape[0]))

                for frame in frames:
                    image_name = f"frame_{frame}.jpg"
                    img = cv2.imread(os.path.join(frames_dir, image_name))
                    out.write(img)

                cv2.destroyAllWindows()
                out.release()
