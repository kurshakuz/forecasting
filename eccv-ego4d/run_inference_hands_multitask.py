import os

import numpy as np
import torch
from decord import VideoReader, cpu
from timm.models import create_model

import modeling_finetune_uniformer_ego4d
import video_transforms as video_transforms
import volume_transforms as volume_transforms

data_path = '/workspace/data'
# model_path = '/workspaces/thesis-ws/ego4d_fhp_uniformer8x320.pth'
model_path = '/workspace/train_models/checkpoint-13.pth'


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
    print("all_index: ", all_index)
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == '__main__':
    args = AttrDict({
        'model': 'uniformer_base_320_ego4d_finetune',
        'nb_verb_classes': 42,
        'nb_noun_classes': 0,
        'num_segments': 8,
        'finetune': model_path,
        'input_size': 320,
        'short_side_size': 320,
        'test_num_segment': 8,
        'device': 'cuda',
    })

    # clip_name = "258_12542"
    clip_name = "8_8890"
    clip_path = os.path.join(data_path, 'cropped_videos_ant', clip_name + '.mp4')

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

    data_transform = video_transforms.Compose([
        video_transforms.Resize(args.short_side_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    
    print(clip_path)

    buffer = loadvideo_decord(clip_path, new_width=320, new_height=256, num_segment=args.num_segments, test_num_segment=args.test_num_segment, keep_aspect_ratio=True)
    buffer = data_transform(buffer)
    videos = buffer
    videos = videos.to(device, non_blocking=True)
    videos = torch.unsqueeze(videos, 0)
    print(f'videos: {videos.shape}')

    model.eval()

    with torch.no_grad():
        # with torch.cuda.amp.autocast():
        test_output_ls = []
        for i in range(args.test_num_segment):
            inputs = videos[:, :, i * args.num_segments:(i + 1) * args.num_segments]
            # print(f'inputs: {inputs.shape}')
            outputs = model(inputs)
            # print(outputs)
            test_output_ls.append(outputs)
        outputs = torch.stack(test_output_ls).mean(dim=0)
        print(outputs)
        reg_out = outputs[:, :21]
        masks_out = outputs[:, 21:]
        masks = (masks_out > 0.4).float()

        # Coordinate Loss
        reg_out = reg_out * masks
        print(reg_out)
