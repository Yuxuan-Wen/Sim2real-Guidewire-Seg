# coding:utf-8
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import glob
import os
import re

parser = argparse.ArgumentParser("Gen Pseudo-labels")
parser.add_argument("--image", type=str, default="./data/real/image", required=False,
                    help="path to the image that used to inference")
parser.add_argument("--checkpoint", default="$YOUR_TRAINED_WEIGHTS$", type=str, required=False,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="vit_b", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./output_x_ray300",
                    help="save the image of the model")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_workers", "-j", type=int, default=1,
                    help="divided value")
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--model_type", default="sam", choices=["dino", "sam"], type=str,
                    help="backbone type")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from learnerable_seg import PromptSAM, PromptDiNo
from scheduler import PolyLRScheduler
from metrics.metric import Metric


class SegDataset_inference:
    def __init__(self, img_paths,
                 pixel_mean=[0.5] * 3, pixel_std=[0.5] * 3,
                 img_size=1024) -> None:
        self.img_paths = img_paths
        self.length = len(img_paths)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        # mask = Image.open(mask_path).convert("L")
        # mask = np.asarray(mask)

        # if self.mask_divide:
        #     mask = mask // self.divide_value
        transform = Compose(
            [
                # ColorJitter(),
                # VerticalFlip(),
                # HorizontalFlip(),
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.pixel_mean, std=self.pixel_std)
            ]
        )
        aug_data = transform(image=img)
        x = aug_data["image"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), self.img_paths[index]


def main(args):
    img_path = args.image
    # mask_path = args.mask_path
    # epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    # optimizer = args.optimizer
    # weight_decay = args.weight_decay
    # lr = args.lr
    # momentum = args.momentum
    bs = args.batch_size
    # divide = args.divide
    # divide_value = args.divide_value
    num_workers = args.num_workers
    model_type = args.model_type
    # pixel_mean=[123.675, 116.28, 103.53],
    # pixel_std=[58.395, 57.12, 57.375],
    # pixel_mean = np.array(pixel_mean) / 255
    # pixel_std = np.array(pixel_std) / 255
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    basename = os.path.basename(img_path)
    _, ext = os.path.splitext(basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        img_paths = [file for file in glob.glob(os.path.join(img_path, "*.*")) if regex.match(file)]
        print("Inference with {} imgs".format(len(img_paths)))
        # mask_paths = [os.path.join(mask_path, os.path.basename(file.replace("jpg", "png"))) for file in img_paths]
    else:
        bs = 1
        img_paths = [img_path]
        # mask_paths = [mask_path]
        num_workers = 1
    
    img_size = 1024
    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=None, num_classes=num_classes, reduction=4, upsample_times=2,
                          groups=4)
    elif model_type == "dino":
        model = PromptDiNo(name=model_name, checkpoint=None, num_classes=num_classes)
        img_size = 518
    
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    
    dataset = SegDataset_inference(img_paths, pixel_mean=pixel_mean, pixel_std=pixel_std, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    best_iou = 0.
    for i, (x, name) in enumerate(dataloader):
        x = x.to(device)

        # if device_type == "cuda" and args.mix_precision:
        #     x = x.to(dtype=torch.float16)
        #     with torch.autocast(device_type=device_type, dtype=torch.float16):
        #         pred = model(x)

        # else:
        x = x.to(dtype=torch.float32)
        pred = model(x)

        pred = pred.argmax(dim=1)  # 如果是多类别分割，取预测的类别索引
        pred = pred.cpu().numpy()  # 转换为 numpy 数组
        
        pred[pred>0.9] = 1

        # 保存每个样本的分割结果
        for j in range(pred.shape[0]):
            pred_img = pred[j]
            pred_img = (pred_img * (255 / pred_img.max())).astype(np.uint8)
            pred_img = Image.fromarray(pred_img)
            
            save_path_ = os.path.join(name[0].replace('jpg', 'png').replace(img_path, save_path))
            
            print("saving", save_path_)
            
            pred_img.save(save_path_)


if __name__ == "__main__":
    main(args)
