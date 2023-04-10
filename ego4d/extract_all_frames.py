import os
import cv2


video_path_prefix = f"/home/dev/workspace/thesis-ego4d/val_data/content/ego4d_data/v1/clips"
downscale = True

for video_path_prefix_inst in os.listdir(video_path_prefix):
    video_path = os.path.join(video_path_prefix, video_path_prefix_inst)
    if os.path.isfile(video_path) and video_path.endswith(".mp4"):
        clip_uid = os.path.splitext(video_path_prefix_inst)[0]
        print(clip_uid)
        if downscale:
            frames_dir = f"/media/dev/HIKVISION/val_data/extracted_all_frames/{clip_uid}"
        else:
            frames_dir = f"/media/dev/HIKVISION/val_data/extracted_all_frames/{clip_uid}"
        # frames_dir = f"/media/dev/HIKVISION/extracted_frames/{clip_uid}/{clip_id}"
        if os.path.exists(frames_dir):
            print("Frames already extracted")
            continue
        os.makedirs(frames_dir, exist_ok=True)
        # iterate over the frame numbers and extract each frame from the video
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f"frame_{frame_num}.jpg")
            if downscale:
                new_height = 256
                new_width = int(frame.shape[1] * (new_height / frame.shape[0]))
                dim = (new_width, new_height)
                frame_res = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(frame_path, frame_res)
            else:
                cv2.imwrite(frame_path, frame)
            frame_num += 1
        cap.release()
