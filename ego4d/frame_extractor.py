import os
import cv2

folder_path = "../thesis-ego4d/extracted_frame_nums"

for clip_uid_folder in os.listdir(folder_path):
    clip_uid_folder_path = os.path.join(folder_path, clip_uid_folder)
    if os.path.isdir(clip_uid_folder_path):
        for txt_file in os.listdir(clip_uid_folder_path):
            txt_file_path = os.path.join(clip_uid_folder_path, txt_file)
            if os.path.isfile(txt_file_path):
                print(f"Content of {txt_file_path}")
                print(os.path.basename(os.path.dirname(clip_uid_folder_path)))
                clip_uid = os.path.basename(clip_uid_folder_path)
                clip_id = os.path.splitext(txt_file)[0]
                # frames_dir = f"../thesis-ego4d/extracted_frames/{clip_uid}/{clip_id}"
                frames_dir = f"/media/dev/HIKVISION/extracted_frames/{clip_uid}/{clip_id}"
                if os.path.exists(frames_dir):
                    print("Frames already extracted")
                    continue
                os.makedirs(frames_dir, exist_ok=True)
                with open(txt_file_path, "r") as f:
                    frames = [int(line.strip()) for line in f.readlines()]
                print(frames)
                # iterate over the frame numbers and extract each frame from the video
                video_path = f"../thesis-ego4d/content/ego4d_data/v1/clips/{clip_uid}.mp4"
                cap = cv2.VideoCapture(video_path)
                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_num in frames:
                        frame_path = os.path.join(frames_dir, f"frame_{frame_num}.jpg")
                        cv2.imwrite(frame_path, frame)
                    frame_num += 1
                cap.release()
