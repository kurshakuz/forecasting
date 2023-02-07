import os
import pickle
import argparse
import pandas as pd
import cv2

from hoi_forecast.preprocess.traj_util import compute_hand_traj
from hoi_forecast.preprocess.dataset_util import FrameDetections, sample_action_anticipation_frames, fetch_data, save_video_info
from hoi_forecast.preprocess.obj_util import compute_obj_traj
from hoi_forecast.preprocess.affordance_util import compute_obj_affordance
from hoi_forecast.preprocess.vis_util import vis_affordance, vis_hand_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="thesis-data", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./hoi_forecast/figs", type=str, help="generated results save path")
    parser.add_argument('--fps', default=10, type=int, help="sample frames per second")
    parser.add_argument('--hand_threshold', default=0.5, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.5, type=float, help="object detection threshold")
    parser.add_argument('--contact_ratio', default=0.4, type=float, help="active obj contact frames ratio")
    parser.add_argument('--num_sampling', default=20, type=int, help="sampling points for affordance")
    parser.add_argument('--num_points', default=5, type=int, help="selected points for affordance")
    parser.add_argument('--video_id', default="empty", type=str, help="video_id")
    parser.add_argument('--start_act_frame', default=0, type=int, help="start_act_frame")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    video_id = args.video_id
    frames_path = os.path.join(args.dataset_path, video_id, "rgb_frames")
    ho_path = os.path.join(args.dataset_path, video_id, "hand-objects", "{}.pkl".format(video_id))
    start_act_frame = args.start_act_frame

    frames_idxs = sample_action_anticipation_frames(start_act_frame, fps=args.fps)
    print(frames_idxs)

    uid = video_id + "_" + str(start_act_frame)

    with open(ho_path, "rb") as f:
        video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]

    results = fetch_data(frames_path, video_detections, frames_idxs)
    if results is None:
        print("data fetch failed")
    else:
        frames_idxs, frames, annots, hand_sides = results
        # print(f"frames: {len(frames)}")
        # print(f"annots: {annots}")
        # print(f"hand_sides: {hand_sides}")

        results_hand = compute_hand_traj(frames, annots, hand_sides, hand_threshold=args.hand_threshold,
                                         obj_threshold=args.obj_threshold)
        if results_hand is None:
            print("compute traj failed")  # homography fails or not enough points
        else:
            homography_stack, hand_trajs = results_hand
            results_obj = compute_obj_traj(frames, annots, hand_sides, homography_stack,
                                           hand_threshold=args.hand_threshold,
                                           obj_threshold=args.obj_threshold,
                                           contact_ratio=args.contact_ratio)
            if results_obj is None:
                print("compute obj traj failed")
            else:
                contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj = results_obj
                frame, homography = frames[-1], homography_stack[-1]
                affordance_info = compute_obj_affordance(frame, annots[-1], active_obj, active_object_idx, homography,
                                                         active_obj_traj=obj_trajs['traj'], obj_bboxs_traj=obj_bboxs_traj,
                                                         num_points=args.num_points, num_sampling=args.num_sampling)
                if affordance_info is not None:
                    img_vis = vis_hand_traj(frames, hand_trajs)
                    img_vis = vis_affordance(img_vis, affordance_info)
                    img = cv2.hconcat([img_vis, frames[-1]])
                    cv2.imwrite(os.path.join(save_path, "demo_{}.jpg".format(uid)), img)
                    save_video_info(save_path, uid, frames_idxs, homography_stack, contacts, hand_trajs, obj_trajs, affordance_info)
    print(f"result stored at {save_path}")
