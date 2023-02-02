import cv2
import os 

video_link = 'xqRfcJX2f_o'
# sample frame every 0.5 seconds
every_s = 0.5

vidcap = cv2.VideoCapture('video.mp4')
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
print(fps)

frame_num = 0
success = True
while success:
    success, image = vidcap.read()
    print('read a new frame:', success)
    if frame_num % int(every_s*fps) == 0:
        frame_name = str(frame_num).zfill(10)
        folder_name = f'./images/{video_link}'
        os.makedirs(folder_name, exist_ok=True)
        cv2.imwrite(f'{folder_name}/frame_{frame_name}.jpg', image)
        print('successfully written 10th frame')
    frame_num += 1
