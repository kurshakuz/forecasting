import os
import sys
import cv2

import tempfile
import numpy as np
import torch
from decord import VideoReader, cpu
from timm.models import create_model

# from video_sampler import store_snippet_in_memory
from video_sampler import create_tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), 'uniformer'))
import volume_transforms as volume_transforms
import video_transforms as video_transforms
import modeling_finetune_uniformer_ego4d

sys.path.append(os.path.join(os.path.dirname(__file__), 'tsptw'))
from main_hands import hands_tsptw_solver

# data_path = '/workspace/data'
data_path = '/home/dev/workspace/sample_videos'
# model_path = '/workspaces/thesis-ws/ego4d_fhp_uniformer8x320.pth'
# model_path = '/workspace/train-models/checkpoint-13-multitask.pth'
model_path = '/home/shyngys/workspace/train-models/checkpoint-13-multitask.pth'


def loadvideo_decord(sample, new_width=340, new_height=256, num_segment=1, test_num_segment=10, keep_aspect_ratio=True):
    """Load video content using Decord"""
    fname = sample

    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        if keep_aspect_ratio:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        else:
            vr = VideoReader(fname, width=new_width, height=new_height,
                             num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []

    num_frames = len(vr)

    tick = num_frames / float(num_segment)
    all_index = []
    for t_seg in range(test_num_segment):
        tmp_index = [
            int(t_seg * tick / test_num_segment + tick * x)
            for x in range(num_segment)
        ]
        all_index.extend(tmp_index)
    all_index = list(np.array(all_index))
    # all_index = list(np.sort(np.array(all_index)))
    # print("all_index: ", all_index)
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Inferencer:
    def __init__(self):
        args = AttrDict({
            'model': 'uniformer_base_320_ego4d_finetune',
            'nb_verb_classes': 42,
            'nb_noun_classes': 0,
            'num_segments': 8,
            'finetune': model_path,
            'input_size': 320,
            'short_side_size': 320,
            'test_num_segment': 1,
            'device': 'cuda',
        })

        assert args.nb_verb_classes > 0 and args.nb_noun_classes == 0
        model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_verb_classes,
        )

        ckpt = model.get_pretrained_checkpoint_file(args.finetune)
        a, b = model.load_state_dict(ckpt, strict=True)
        print("Finetune model loading: ", a, b)

        device = torch.device(args.device)
        model.to(device)
        model.eval()

        data_transform = video_transforms.Compose([
            video_transforms.Resize(
                args.short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(
                size=(args.input_size, args.input_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        ])

        self.args = args
        self.model = model
        self.device = device
        self.data_transform = data_transform

    def predict_video(self, clip_name):
        clip_path = os.path.join(data_path, clip_name + '.mp4')
        if not os.path.exists(clip_path):
            print("Video not found: ", clip_path)
            return []

        buffer = loadvideo_decord(clip_path, new_width=320, new_height=256, num_segment=self.args.num_segments,
                                  test_num_segment=self.args.test_num_segment, keep_aspect_ratio=True)
        buffer = self.data_transform(buffer)
        videos = buffer
        videos = videos.to(self.device, non_blocking=True)
        videos = torch.unsqueeze(videos, 0)
        # print(f'videos: {videos.shape}')

        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            test_output_ls = []
            for i in range(self.args.test_num_segment):
                inputs = videos[:, :, i *
                                self.args.num_segments:(i + 1) * self.args.num_segments]
                # print(inputs)
                # print(f'inputs: {inputs.shape}')
                outputs = self.model(inputs)
                # print(outputs)
                test_output_ls.append(outputs)
            outputs = torch.stack(test_output_ls).mean(dim=0)
            # print(outputs)
            reg_out = outputs[:, :21]
            masks_out = outputs[:, 21:]
            masks = (masks_out > 0.7).float()

            # Coordinate Loss
            reg_out = reg_out * masks
            # print(reg_out)

            if self.args.device != 'cpu':
                return reg_out.cpu().numpy()
            else:
                return reg_out.cpu().numpy()

    def predict_in_memory(self, temp_file_path):
        buffer = loadvideo_decord(temp_file_path, new_width=320, new_height=256, num_segment=self.args.num_segments,
                                  test_num_segment=self.args.test_num_segment, keep_aspect_ratio=True)
        buffer = self.data_transform(buffer)
        videos = buffer
        videos = videos.to(self.device, non_blocking=True)
        videos = torch.unsqueeze(videos, 0)
        # print(f'videos: {videos.shape}')

        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            test_output_ls = []
            for i in range(self.args.test_num_segment):
                inputs = videos[:, :, i *
                                self.args.num_segments:(i + 1) * self.args.num_segments]
                # print(inputs)
                # print(f'inputs: {inputs.shape}')
                outputs = self.model(inputs)
                # print(outputs)
                test_output_ls.append(outputs)
            outputs = torch.stack(test_output_ls).mean(dim=0)
            # print(outputs)
            reg_out = outputs[:, :21]
            masks_out = outputs[:, 21:]
            masks = (masks_out > 0.7).float()

            # Coordinate Loss
            reg_out = reg_out * masks
            # print(reg_out)

            if self.args.device != 'cpu':
                return reg_out.cpu().numpy()
            else:
                return reg_out.cpu().numpy()


def inference_snippets_video(video_path, snippet_length, overlap_length):
    inferencer = Inferencer()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    snippet_frames = snippet_length * fps
    overlap_frames = overlap_length * fps
    snippet_count = (total_frames - snippet_frames) // overlap_frames + 1

    buffer_frames = []
    snippet_num = 1
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        buffer_frames.append(frame)

        if len(buffer_frames) > snippet_frames:
            buffer_frames.pop(0)

        if len(buffer_frames) == snippet_frames:
            if frame_num % overlap_frames == 0 or frame_num == total_frames:
                snippet = buffer_frames.copy()
                temp_file_path = create_tempfile(snippet)
                result = inferencer.predict_in_memory(temp_file_path)
                print(result)
                hands_tsptw_solver(30, 8, 'rdy', result)
                snippet_num += 1

    cap.release()


if __name__ == '__main__':
    # inferencer = Inferencer()
    # for i in range(10):
    #     clip_name = f"snippet_{i}"
    #     result = inferencer.predict(clip_name)
    #     print(result)
    video_path = '/home/dev/workspace/sample_videos/30da536e-4848-4d54-8f2b-6fe1ad54be11.mp4'
    snippet_length = 2  # in seconds
    overlap_length = 1  # in seconds

    inference_snippets_video(video_path, snippet_length, overlap_length)
