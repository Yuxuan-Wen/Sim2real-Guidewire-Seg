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
import random
import torch.nn.functional as F


parser = argparse.ArgumentParser("Coarse Stage")
parser.add_argument("--img_path_synthesized", type=str, default="./data/synthesized/image", required=False,
                    help="path to the image that used to train the model")
parser.add_argument("--mask_path_synthesized", type=str, default="./data/synthesized/mask", required=False,
                    help="path to the mask file for training")
# parser.add_argument("--img_path_real", type=str, default="./data/real/image", required=False,
#                     help="path to the image that used to inference the model for pseudo-labels")
# parser.add_argument("--mask_path_real", type=str, default="./data/real/mask", required=False,
#                     help="path to the mask file to inference the model for pseudo-labels (for evaluation only)")
parser.add_argument("--epoch", type=int, default=3,
                    help="training epochs")
parser.add_argument("--checkpoint", default="./sam_vit_b_01ec64.pth", type=str, required=False,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="vit_b", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./ckpt_prompt",
                    help="save the weights of the model")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precison training")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--optimizer", default="adamw", type=str,
                    help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float,
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum for the sgd")
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
from transform import FullAugmentor

class SegDataset:
    def __init__(self, img_paths, mask_paths,
                 pixel_mean=[0.5] * 3, pixel_std=[0.5] * 3,
                 img_size=1024) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.length = len(img_paths)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
        self.augmentor = FullAugmentor()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        mask = Image.open(mask_path).convert("L")
        mask = np.asarray(mask)

        img = np.copy(img)
        mask = np.copy(mask)

        mask[mask > 0] = 1

        transform = Compose(
            [
                ColorJitter(),
                VerticalFlip(),
                HorizontalFlip(),
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.pixel_mean, std=self.pixel_std)
            ]
        )
        aug_data = transform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        
        x = torch.from_numpy(x)
        target = torch.from_numpy(target)

        img_t1 = self.augmentor.forward_image(x)

        return x, target, img_t1

def sample_lists(img_path, mask_path):
    basename = os.path.basename(img_path)
    _, ext = os.path.splitext(basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        img_paths = [file for file in glob.glob(os.path.join(img_path, "*.*")) if regex.match(file)]
        mask_paths = [os.path.join(mask_path, os.path.basename(file.replace("jpg", "png"))) for file in
                      img_paths]

    return img_paths, mask_paths

def split_train_test(img_paths, mask_paths, test_ratio=0.2):
    img_paths = sorted(img_paths)
    mask_paths = sorted(mask_paths)

    # Combine the image and mask paths into a list of tuples
    data_pairs = list(zip(img_paths, mask_paths))

    # Shuffle the combined list to randomize the data
    random.shuffle(data_pairs)

    # Split the data into training and testing sets
    split_index = int(len(data_pairs) * (1 - test_ratio))
    train_pairs = data_pairs[:split_index]
    test_pairs = data_pairs[split_index:]

    # Unzip the pairs back into separate lists
    train_img_paths, train_mask_paths = zip(*train_pairs)
    test_img_paths, test_mask_paths = zip(*test_pairs)

    return list(train_img_paths), list(train_mask_paths), list(test_img_paths), list(test_mask_paths)

def consist_loss(inputs, targets, key_list=None):
    """
    Consistency regularization between two augmented views
    """
    loss = 0.0
    keys = key_list if key_list is not None else list(inputs.keys())
    for key in keys:
        loss += (1.0 - F.cosine_similarity(inputs[key], targets[key], dim=1)).mean()
    return loss

def uncertainty_loss(inputs, targets):
    """
    Uncertainty rectified pseudo supervised loss
    """
    # detach from the computational graph
    pseudo_label = F.softmax(targets, dim=1).detach()
    vanilla_loss = F.cross_entropy(inputs, pseudo_label, reduction='none')
    # uncertainty rectification
    kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'), dim=1)
    uncertainty_loss = (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
    return uncertainty_loss

def main(args):
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    momentum = args.momentum
    bs = args.batch_size
    divide = args.divide
    num_workers = args.num_workers
    model_type = args.model_type
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes

    img_size = 1024
    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes, reduction=4, upsample_times=2,
                          groups=4)
    elif model_type == "dino":
        model = PromptDiNo(name=model_name, checkpoint=checkpoint, num_classes=num_classes)
        img_size = 518

    img_paths_synthesized, mask_paths_synthesized = sample_lists(args.img_path_synthesized, args.mask_path_img_path_synthesized)
    print("train with {} Synthesized images".format(len(img_paths_synthesized)))

    db_train_synthesized = SegDataset(img_paths=img_paths_synthesized, mask_paths=mask_paths_synthesized, mask_divide=divide, divide_value=divide_value,
                          pixel_mean=pixel_mean, pixel_std=pixel_std, img_size=img_size)

    dataloader_synthesized = DataLoader(db_train_synthesized, batch_size=bs, shuffle=True, num_workers=num_workers)

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if optimizer == "adamw":
        optim = opt.AdamW([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params": model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=True)
    loss_func = nn.CrossEntropyLoss()
    scheduler = PolyLRScheduler(optim, num_images=len(img_paths_synthesized), batch_size=bs, epochs=epochs)
    metric = Metric(num_classes=num_classes)
    
    for epoch in range(epochs):
        for i, synthesized_data in enumerate(dataloader_synthesized):

            _, target_synthesized, x_synthesized  = synthesized_data
            x_synthesized = x_synthesized.to(device)
            target_synthesized = target_synthesized.to(device, dtype=torch.long)
            
            optim.zero_grad()
            x_synthesized = x_synthesized.to(dtype=torch.float32)

            pred_synthesized = model(x_synthesized)
            loss_seg = loss_func(pred_synthesized, target_synthesized)
            loss = 10 * loss_seg

            loss.backward()
            optim.step()

            print("epoch:{}-{}: loss_seg:{}".format(epoch + 1, i + 1, loss_seg.item()))
            scheduler.step()

        torch.save(
            model.state_dict(), os.path.join(save_path, "Coarse_Stage_{}_{}_Epoch{}.pth".format(model_type, model_name, epoch))
        )


if __name__ == "__main__":
    main(args)
