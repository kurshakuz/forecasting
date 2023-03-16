#!/bin/bash

# get the clip_uid and clip_id from the given path
clip_uid=$(dirname "$1")
clip_id=$(basename "$1" | cut -f 1 -d '.')

# create a directory for the clip_id inside the clip_uid directory
mkdir -p "$clip_uid/$clip_id"

# read the frame numbers from the txt file and store them in an array
mapfile -t frames < "$clip_uid/$clip_id.txt"

# iterate over the frame numbers and extract each frame from the video
for frame in "${frames[@]}"
do
  # use ffmpeg to extract the frame and save it in the clip_id directory with the corresponding frame number as the name
  ffmpeg -i "$clip_uid/$clip_id.mp4" -vf "select=gte(n\,$frame)" -vframes 1 "$clip_uid/$clip_id/frame_$frame.jpg"
done
