# importing libraries
import os
import cv2
from PIL import Image

path = "/media/dev/HIKVISION/extracted_frames"
video_path = "/media/dev/HIKVISION/extracted_videos"

# Video Generating function
def generate_video():
    for video_uid in os.listdir(path):
        for clip_uid in os.listdir(f"{path}/{video_uid}"):
            # print(f"{path}/{video_uid}/{clip_uid}")
            image_folder = f"{path}/{video_uid}/{clip_uid}"
            video_folder = f"{video_path}/{video_uid}/"
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)

            video_name = f"{video_folder}/{clip_uid}.mp4"
            images = [img for img in os.listdir(image_folder)
                    if img.endswith(".jpg")]
            
            img = cv2.imread(os.path.join(image_folder, images[0]))
            new_height = 256
            new_width = int(img.shape[1] * (new_height / img.shape[0]))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_name, fourcc, 30, (new_width, new_height))
            for i in range(len(images)):
                img = cv2.imread(os.path.join(image_folder, images[i]))
                resized_image = cv2.resize(img, (new_width, new_height))
                out.write(resized_image)

            cv2.destroyAllWindows()
            out.release()

# Calling the generate_video function
generate_video()
