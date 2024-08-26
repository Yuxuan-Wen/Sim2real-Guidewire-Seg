import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
import numpy as np
from PIL import Image


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


# def get_prompts(cfg: Box, bboxes, gt_masks):
    
#     # print(cfg.prompt)
    
#     if cfg.prompt == "box" or cfg.prompt == "coarse":
#         prompts = bboxes
#     elif cfg.prompt == "point":
#         prompts = get_point_prompts(gt_masks, cfg.num_points)
#     else:
#         raise ValueError("Prompt Type Error!")
#     return prompts

def get_prompts(cfg: Box, bboxes, gt_masks):
    
    prompts_boxes = bboxes
    prompts_points = get_point_prompts(gt_masks, cfg.num_points)
    
    prompts = [prompts_boxes, prompts_points]

    return prompts


def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    
    print("Validate End-to-End")
    
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    accuracys = AverageMeter()
    sensitivitys = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, None)
            
            gt_masks = torch.stack(gt_masks, dim=0)
            
#             for i in range(pred_masks.shape[0]):
#                 # print(pred_masks.shape)
#                 image = images[i,0,:,:].cpu().numpy()
#                 pred_mask = pred_masks[i,:,:,:].cpu().numpy()  # 转为numpy数组
                
#                 def sigmoid(x):
#                     return 1 / (1 + np.exp(-x))
#                 pred_mask = sigmoid(pred_mask)
#                 pred_mask = (pred_mask > 0.5).astype(np.uint8)
                
#                 pred_mask = (pred_mask * 255).astype(np.uint8)  # 归一化到[0, 255]范围
#                 image = (image * 255).astype(np.uint8)  # 归一化到[0, 255]范围
                
#                 if pred_mask.shape[0] == 1:
#                     pred_mask = np.squeeze(pred_mask, axis=0)  # 移除单通道维度
#                 if image.shape[0] == 1:
#                     image = np.squeeze(image, axis=0)  # 移除单通道维度
                
#                 pred_image = Image.fromarray(pred_mask)
#                 image = Image.fromarray(image)
                
#                 pred_image.save(os.path.join("HUG_save_ours", f"pred_mask_{iter}_{i}.png"))
#                 image.save(os.path.join("HUG_save_ours", f"pred_mask_{iter}_{i}_image.png"))
            
            # for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            batch_stats = smp.metrics.get_stats(
                pred_masks,
                gt_masks.int(),
                mode='binary',
                threshold=0.5,
            )
            '''
            fbeta_score.__doc__ += _doc
            f1_score.__doc__ += _doc
            iou_score.__doc__ += _doc
            accuracy.__doc__ += _doc
            sensitivity.__doc__ += _doc
            specificity.__doc__ += _doc
            balanced_accuracy.__doc__ += _doc
            '''
            
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            batch_accuracy = smp.metrics.accuracy(*batch_stats, reduction="micro-imagewise")
            batch_sensitivity = smp.metrics.sensitivity(*batch_stats, reduction="micro-imagewise")
            
            ious.update(batch_iou, num_images)
            f1_scores.update(batch_f1, num_images)
            accuracys.update(batch_accuracy, num_images)
            sensitivitys.update(batch_sensitivity, num_images)
            
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]-- Mean Accuracy: [{accuracys.avg:.4f}]-- Mean Sensitivity: [{sensitivitys.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": "end-to-end", "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "Mean Accuracy":  f"{accuracys.avg:.4f}", "Mean Sensitivity": f"{sensitivitys.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def validate_with_prompt(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    
    print("Validate with prompt type:", cfg.prompt)
    
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    accuracys = AverageMeter()
    sensitivitys = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts[0])
            
            gt_masks = torch.stack(gt_masks, dim=0)
            
            
#             for i in range(pred_masks.shape[0]):
#                 # print(pred_masks.shape)
#                 image = images[i,0,:,:].cpu().numpy()
#                 pred_mask = pred_masks[i,:,:,:].cpu().numpy()  # 转为numpy数组
                
#                 def sigmoid(x):
#                     return 1 / (1 + np.exp(-x))
#                 pred_mask = sigmoid(pred_mask)
#                 pred_mask = (pred_mask > 0.5).astype(np.uint8)
                
#                 pred_mask = (pred_mask * 255).astype(np.uint8)  # 归一化到[0, 255]范围
#                 image = (image * 255).astype(np.uint8)  # 归一化到[0, 255]范围
                
#                 if pred_mask.shape[0] == 1:
#                     pred_mask = np.squeeze(pred_mask, axis=0)  # 移除单通道维度
#                 if image.shape[0] == 1:
#                     image = np.squeeze(image, axis=0)  # 移除单通道维度
                
#                 pred_image = Image.fromarray(pred_mask)
#                 image = Image.fromarray(image)
                
#                 pred_image.save(os.path.join("HUG_save_teacher_box", f"pred_mask_{iter}_{i}.png"))
#                 image.save(os.path.join("HUG_save_teacher_box", f"pred_mask_{iter}_{i}_image.png"))
            

            # for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            batch_stats = smp.metrics.get_stats(
                pred_masks,
                gt_masks.int(),
                mode='binary',
                threshold=0.5,
            )
            '''
            fbeta_score.__doc__ += _doc
            f1_score.__doc__ += _doc
            iou_score.__doc__ += _doc
            accuracy.__doc__ += _doc
            sensitivity.__doc__ += _doc
            specificity.__doc__ += _doc
            balanced_accuracy.__doc__ += _doc
            '''
            
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            batch_accuracy = smp.metrics.accuracy(*batch_stats, reduction="micro-imagewise")
            batch_sensitivity = smp.metrics.sensitivity(*batch_stats, reduction="micro-imagewise")
            
            ious.update(batch_iou, num_images)
            f1_scores.update(batch_f1, num_images)
            accuracys.update(batch_accuracy, num_images)
            sensitivitys.update(batch_sensitivity, num_images)
            
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]-- Mean Accuracy: [{accuracys.avg:.4f}]-- Mean Sensitivity: [{sensitivitys.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "Mean Accuracy":  f"{accuracys.avg:.4f}", "Mean Sensitivity": f"{sensitivitys.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def validate_with_prompt_point(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    
    print("Validate with prompt type:", cfg.prompt)
    
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    accuracys = AverageMeter()
    sensitivitys = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts[1])
            
            gt_masks = torch.stack(gt_masks, dim=0)
            
            # for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            batch_stats = smp.metrics.get_stats(
                pred_masks,
                gt_masks.int(),
                mode='binary',
                threshold=0.5,
            )
            '''
            fbeta_score.__doc__ += _doc
            f1_score.__doc__ += _doc
            iou_score.__doc__ += _doc
            accuracy.__doc__ += _doc
            sensitivity.__doc__ += _doc
            specificity.__doc__ += _doc
            balanced_accuracy.__doc__ += _doc
            '''
            
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            batch_accuracy = smp.metrics.accuracy(*batch_stats, reduction="micro-imagewise")
            batch_sensitivity = smp.metrics.sensitivity(*batch_stats, reduction="micro-imagewise")
            
            ious.update(batch_iou, num_images)
            f1_scores.update(batch_f1, num_images)
            accuracys.update(batch_accuracy, num_images)
            sensitivitys.update(batch_sensitivity, num_images)
            
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]-- Mean Accuracy: [{accuracys.avg:.4f}]-- Mean Sensitivity: [{sensitivitys.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "Mean Accuracy":  f"{accuracys.avg:.4f}", "Mean Sensitivity": f"{sensitivitys.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def validate_with_prompt_all(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    
    print("Validate with prompt type:", cfg.prompt)
    
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    accuracys = AverageMeter()
    sensitivitys = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            
            gt_masks = torch.stack(gt_masks, dim=0)
            
            # for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            batch_stats = smp.metrics.get_stats(
                pred_masks,
                gt_masks.int(),
                mode='binary',
                threshold=0.5,
            )
            '''
            fbeta_score.__doc__ += _doc
            f1_score.__doc__ += _doc
            iou_score.__doc__ += _doc
            accuracy.__doc__ += _doc
            sensitivity.__doc__ += _doc
            specificity.__doc__ += _doc
            balanced_accuracy.__doc__ += _doc
            '''
            
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            batch_accuracy = smp.metrics.accuracy(*batch_stats, reduction="micro-imagewise")
            batch_sensitivity = smp.metrics.sensitivity(*batch_stats, reduction="micro-imagewise")
            
            ious.update(batch_iou, num_images)
            f1_scores.update(batch_f1, num_images)
            accuracys.update(batch_accuracy, num_images)
            sensitivitys.update(batch_sensitivity, num_images)
            
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]-- Mean Accuracy: [{accuracys.avg:.4f}]-- Mean Sensitivity: [{sensitivitys.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "Mean Accuracy":  f"{accuracys.avg:.4f}", "Mean Sensitivity": f"{sensitivitys.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def unspervised_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    init_prompt = cfg.prompt
    cfg.prompt = "box"
    iou_box, f1_box = validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = "point"
    iou_point, f1_point = validate(fabric, cfg, model, val_dataloader, name, iters)
    # cfg.prompt = "coarse"
    # validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = init_prompt
    return iou_box, f1_box, iou_point, f1_point


def contrast_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0, loss: float = 0.):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters, "loss": loss}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"metrics-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg
