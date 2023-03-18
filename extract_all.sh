#!/bin/bash

folder_path="../thesis-ego4d/extracted_frame_nums"

for clip_uid_folder in $folder_path/*; do
    if [ -d "$clip_uid_folder" ]; then
        for txt_file in "$clip_uid_folder"/*.txt; do
            if [ -f "$txt_file" ]; then
                echo "content of $txt_file"
                echo "$(basename $(dirname "$clip_uid_folder"))"
                clip_uid="$(basename $clip_uid_folder)"
                clip_id="$(basename "$txt_file" | cut -f 1 -d '.')"
                mkdir -p "../thesis-ego4d/extracted_frames/$clip_uid/$clip_id"
                mapfile -t frames < "$txt_file"

                # iterate over the frame numbers and extract each frame from the video
                for frame in "${frames[@]}"
                do
                    # use ffmpeg to extract the frame and save it in the clip_id directory with the corresponding frame number as the name
                    ffmpeg -i "../thesis-ego4d/content/ego4d_data/v1/clips/$clip_uid.mp4" -vf "select=gte(n\,$frame)" -vframes 1 "../thesis-ego4d/extracted_frames/$clip_uid/$clip_id/frame_$frame.jpg"
                done
            fi
        done
    fi
done