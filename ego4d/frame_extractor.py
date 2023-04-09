import os
import cv2


split = "train" # train, val, test_unannotated
folder_path = f"/home/dev/workspace/thesis-ego4d/{split}_data/extracted_frame_nums"
video_path_prefix = f"/home/dev/workspace/thesis-ego4d/{split}_data/content/ego4d_data/v1/clips"
video_store_path = f"/media/dev/HIKVISION/{split}_data/extracted_videos"
downscale = True

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
                # if downscale:
                #     frames_dir = f"/media/dev/HIKVISION/{split}_data/extracted_downscaled_frames/{clip_uid}/{clip_id}"
                # else:
                #     frames_dir = f"/media/dev/HIKVISION/{split}_data/extracted_frames/{clip_uid}/{clip_id}"
                # frames_dir = f"/media/dev/HIKVISION/extracted_frames/{clip_uid}/{clip_id}"
                # if os.path.exists(frames_dir):
                #     print("Frames already extracted")
                #     continue
                with open(txt_file_path, "r") as f:
                    frames = [int(line.strip()) for line in f.readlines()]
                print(frames)

                # iterate over the frame numbers and extract each frame from the video
                video_path = f"{video_path_prefix}/{clip_uid}.mp4"
                print(video_path)
                cap = cv2.VideoCapture(video_path)

                # extract only first 60 frames from frames
                frames = frames[:60]

                # create video writer
                width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                new_height = 256
                new_width = int(width * (new_height / height))
                dim = (new_width, new_height)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                txt_name = os.path.splitext(txt_file)[0]
                os.makedirs(video_store_path, exist_ok=True)
                video_name_store = f"{video_store_path}/{txt_name}.mp4"
                print(video_name_store)
                out = cv2.VideoWriter(video_name_store, fourcc, 30, dim)

                for frame_num in frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    # print(ret)
                    # print(frame_num)
                    if downscale:
                        resized_image = cv2.resize(frame, dim)
                        out.write(resized_image)
                        # cv2.imwrite(frame_path, frame_res)
                    else:
                        # cv2.imwrite(frame_path, frame)
                        pass
                out.release()
                cap.release()
            # break
    # break