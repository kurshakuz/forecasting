#!/bin/bash

# get the name of the video file without the extension
video_name=$(basename "$1" | cut -f 1 -d '.')

# create a directory with the same name as the video file
mkdir "$video_name"

# read the frame numbers from the txt file and store them in an array
mapfile -t frames < "$video_name.txt"

# iterate over the frame numbers and extract each frame from the video
for frame in "${frames[@]}"
do
  # use ffmpeg to extract the frame and save it in the video directory
  ffmpeg -i "$1" -vf "select=gte(n\,$frame)" -vframes 1 "$video_name/frame_$frame.jpg"
done
