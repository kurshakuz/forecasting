# importing libraries
import os
import shutil

video_path = "/media/dev/HIKVISION/extracted_videos"
store_path = "/media/dev/HIKVISION/ego4d_data/cropped_videos_ant"

def gather_video():
    for clip_uid in os.listdir(video_path):
        video_folder = f"{video_path}/{clip_uid}"
        for video in os.listdir(video_folder):
            video_full_path = f"{video_folder}/{video}"
            video_new_path = f"{store_path}/{video}"
            print(video_new_path)
            print(video_full_path)
            shutil.copyfile(video_full_path, video_new_path)

gather_video()
