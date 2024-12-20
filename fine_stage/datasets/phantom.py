import os
import cv2
import random
import glob
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage.draw import polygon2mask

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_, decode_mask


class PhantomDataset(Dataset):
    def __init__(self, cfg, root_dir, transform=None, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform

        valid_extensions = ('.jpg', '.png')
        images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(valid_extensions)]
        images = sorted(images)

        self.images = images
        self.gts = [image_path.replace("image", "mask").replace("jpg", "png") for image_path in self.images]

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gt_path = self.gts[idx]
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        gt_mask = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (gt_mask.shape[1], gt_mask.shape[0]))

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = image_path
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image


        masks = []
        bboxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)

        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            masks.append(mask)
            x, y, w, h = cv2.boundingRect(mask)
            bboxes.append([x, y, x + w, y + h])
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)

            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            image_name =  os.path.splitext(os.path.basename(self.images[idx]))[0]
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_name, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)

            return image, torch.tensor(bboxes), torch.tensor(masks).float()


class PhantomDatasetwithCoarse(PhantomDataset):

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_path = self.gts[idx]
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        masks = []
        bboxes = []
        approxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)

            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype)            
            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes.append([x_min, y_min, x_max, y_max])

            masks.append(mask)
            categories.append("0")
            approxes.append(approx)

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            image_name =  os.path.splitext(os.path.basename(self.images[idx]))[0]
            origin_image = image
            origin_approxes = approxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), self.cfg.visual)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_name, padding, origin_image, origin_approxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = PhantomDataset(
        cfg,
        root_dir=cfg.datasets.phantom.test_real,
        transform=transform,
    )
    train = PhantomDataset(
        cfg,
        root_dir=cfg.datasets.phantom.train_real,
        transform=transform,
        if_self_training=cfg.augment,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = PhantomDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.phantom.test,
        transform=transform,
    )
    train = PhantomDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.phantom.train,
        transform=transform,
        if_self_training=cfg.augment,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = PhantomDataset(
        cfg,
        root_dir=cfg.datasets.phantom.test,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader


def load_datasets_visual_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = PhantomDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.phantom.test,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader


def load_datasets_prompt(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = PhantomDataset(
        cfg,
        root_dir=cfg.datasets.phantom.train,
        transform=transform,
        if_self_training=cfg.augment,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return train_dataloader