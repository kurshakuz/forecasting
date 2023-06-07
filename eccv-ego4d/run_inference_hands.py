from ego4d_hands_inference import Ego4dHandsDataset
import torch
from timm.models import create_model
import video_transforms as video_transforms
import volume_transforms as volume_transforms
import os


def build_dataset(is_train, test_mode, args):
    dataset = Ego4dHandsDataset(
        anno_path='./',
        data_path='/content/drive/MyDrive/data_path',
        mode='mode',
        clip_len=args.num_frames,
        num_segment=args.num_segments,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args)
    verb_classes = 20
    noun_classes = 0

    print(noun_classes, args.nb_noun_classes)
    print(verb_classes, args.nb_verb_classes)
    assert verb_classes == args.nb_verb_classes
    assert noun_classes == args.nb_noun_classes
    # print("Number of the class = %d" % args.nb_classes)
    # print("Now this datasets only support EGO4D!")
    return dataset, verb_classes, noun_classes

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == '__main__':
    args = AttrDict({
        'model': 'uniformer_base_320_ego4d_finetune_trained',
        'nb_verb_classes': 20,
        'nb_noun_classes': 0,
        'data_set': 'ego4d_hands',
        'data_path': './dummy',
        'log_dir': './workdir1/ego4d_hands_uniformer_base',
        'output_dir': './workdir1/ego4d_hands_uniformer_base',
        'batch_size': 1,
        'num_sample': 1,
        'num_segments': 8,
        'finetune': '/workspace/thesis-ego4d/ego4d_hands_uniformer_base/checkpoint-13.pth',
        'warmup_epochs': 1,
        'input_size': 320,
        'short_side_size': 320,
        'save_ckpt_freq': 1,
        'num_frames': 16,
        'layer_decay': 1.,
        'opt': 'adamw',
        'test_num_segment': 30,
        'test_num_crop': 1,
        'num_workers': 1,
        'pin_mem': True,
        'device': 'cpu',
    })

    dataset, _, _ = build_dataset(is_train=False, test_mode=False, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    assert args.nb_verb_classes > 0 and args.nb_noun_classes == 0
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_verb_classes,
    )

    device = torch.device(args.device)
    model.to(device)
    pass
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    clip_name = "258_12542"
    data_path = '/content/drive/MyDrive/data_path'
    x = os.path.join(data_path, 'cropped_videos_ant', clip_name + '.mp4')
    print(x)
    assert os.path.exists(x)

    data_transform = video_transforms.Compose([
        video_transforms.Resize(args.short_side_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    buffer = dataset.loadvideo_decord(x)
    print(buffer.shape)
    buffer = dataset.data_transform(buffer)
    videos = buffer
    videos = videos.to(device, non_blocking=True)
    print(videos.shape)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            test_output_ls = []
            print(videos.shape)
            for i in range(args.test_num_segment):
                # print(i * args.num_segments, (i + 1) * args.num_segments)
                # inputs = videos[:, :, i * args.num_segments:(i + 1) * args.num_segments]
                inputs = videos[:, i * args.num_segments:(i + 1) * args.num_segments, :, :]
                print(inputs.shape)
                inputs = inputs[None, :, :, :, :]
                print(inputs.shape)
                print(model(inputs))
                outputs = model(inputs)
                test_output_ls.append(outputs)
            outputs = torch.stack(test_output_ls).mean(dim=0)
    print(outputs)