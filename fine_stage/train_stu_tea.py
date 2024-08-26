import os
import glob
import time
import torch
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts, validate_with_prompt_box, validate_with_prompt_point, validate_with_prompt_all
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    teacher_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    optimizer_teacher: _FabricOptimizer,
    scheduler_teacher: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    focal_losses = AverageMeter()
    dice_losses = AverageMeter()
    iou_losses = AverageMeter()
    ws_losses = AverageMeter()
    total_losses = AverageMeter()
    c_emb_losses = AverageMeter()

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    c_emb_loss = ContraLoss()

    end = time.time()
    current_iter = 0
    num_iters = cfg.num_iters
    max_iou = 0.


    print(f"Training {num_iters} Epoches.")

    # _, _ = validate_with_prompt_box(fabric, cfg, teacher_model, val_dataloader, cfg.name, current_iter)
    # _, _ = validate_with_prompt_point(fabric, cfg, teacher_model, val_dataloader, cfg.name, current_iter)
    # _, _ = validate_with_prompt_all(fabric, cfg, teacher_model, val_dataloader, cfg.name, current_iter)
    
    for epoch in range(num_iters):
        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, pseudo_label_masks = data
            batch_size = images_weak.size(0)
            
            prompts = get_prompts(cfg, bboxes, pseudo_label_masks)

            soft_image_embeds, soft_masks, _, soft_res_masks = model(images_weak, None)    # student
            pred_image_embeds, pred_masks, _, pred_res_masks = model(images_strong, None)   # student
            
            
            if epoch <= cfg.warm_up_epoch:
                soft_image_embeds_tea, soft_masks_tea, _, soft_res_masks_tea = teacher_model(images_weak, prompts[0])    # teacher
                pred_image_embeds_tea, pred_masks_tea, _, pred_res_masks_tea = teacher_model(images_strong, prompts[0])   # teacher
            
            else:
                with torch.no_grad(): 
                    soft_image_embeds_tea, soft_masks_tea, _, soft_res_masks_tea = teacher_model(images_weak, prompts[0])    # teacher
                    pred_image_embeds_tea, pred_masks_tea, _, pred_res_masks_tea = teacher_model(images_strong, prompts[0])   # teacher


            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            loss_ws = torch.tensor(0., device=fabric.device)
            loss_c_emb = torch.tensor(0., device=fabric.device)

            pseudo_label_mask = torch.stack(pseudo_label_masks, dim=0)
            pred_mask, soft_mask = pred_masks, soft_masks
            
            if epoch <= cfg.warm_up_epoch:
                pseudo_label_mask = (pseudo_label_mask > 0.).float()
                loss_ws += (0.5 * dice_loss(pred_mask, pseudo_label_mask) + 0.5 * dice_loss(soft_mask, pseudo_label_mask))
                loss_ws +=  (0.5 * dice_loss(pred_masks_tea, pseudo_label_mask) + 0.5 * dice_loss(soft_masks_tea, pseudo_label_mask))
            
                soft_masks_p0 = (soft_masks_p0 > 0.5).float()
                loss_focal += 0.5 * focal_loss(pred_masks_tea, soft_masks_p0, num_masks)
                loss_dice += 0.5 * dice_loss(pred_masks_tea, soft_masks_p0, num_masks)
                
                loss_c_emb += 0.5 * c_emb_loss(soft_image_embeds, soft_image_embeds_tea, soft_res_masks.clone().detach(), soft_res_masks_tea.clone().detach())
                loss_c_emb += 0.5 * c_emb_loss(pred_image_embeds, soft_image_embeds_tea, pred_res_masks.clone().detach(), soft_res_masks_tea.clone().detach())
        
            else:

                loss_c_emb += c_emb_loss(soft_image_embeds, soft_image_embeds_tea, soft_res_masks.clone().detach(), soft_res_masks_tea.clone().detach())
                loss_c_emb += c_emb_loss(pred_image_embeds, soft_image_embeds_tea, pred_res_masks.clone().detach(), soft_res_masks_tea.clone().detach())

                soft_masks_p0 = (soft_masks_p0 > 0.5).float()
                loss_ws += 5.0 * (0.5 * dice_loss(pred_mask, soft_masks_p0.detach()) + 0.5 * dice_loss(soft_mask, soft_masks_p0.detach()))

            loss_total =  loss_ws + loss_c_emb + loss_focal + loss_dice
            
            fabric.backward(loss_total)

            optimizer.step()
            scheduler.step()
            optimizer_teacher.step()
            scheduler_teacher.step()
            
            optimizer.zero_grad()
            optimizer_teacher.zero_grad()
            torch.cuda.empty_cache()
            
            
            current_iter += 1
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            ws_losses.update(loss_ws.item(), batch_size)
            c_emb_losses.update(loss_c_emb.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Iter: [{current_iter + 1}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | WS Loss [{ws_losses.val:.4f} ({ws_losses.avg:.4f})]'
                         f' | C_emb Loss [{c_emb_losses.val:.4f} ({c_emb_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            loss_logger = {"Focal Loss": focal_losses.avg, "Dice Loss": dice_losses.avg,
                 "ws Loss": ws_losses.avg, "Total Loss": total_losses.avg}

            fabric.log_dict(loss_logger, current_iter)
            torch.cuda.empty_cache()


        print("Evaluate e2e student model")
        iou, f1_score = validate(fabric, cfg, model, val_dataloader, cfg.name, current_iter)
        
        # if iou > max_iou:
        #     state = {"model": model, "optimizer": optimizer}
        #     fabric.save(os.path.join(cfg.out_dir, "save", f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth"), state)
        #     max_iou = iou

        state = {"model": model, "optimizer": optimizer}
        fabric.save(os.path.join(cfg.out_dir, "save", f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth"), state)
        
        if epoch <= cfg.warm_up_epoch:
            print("Evaluate teacher model with box prompt")
            iou_tea, f1_score_tea = validate_with_prompt_box(fabric, cfg, teacher_model, val_dataloader, cfg.name, current_iter)


def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def multi_main(cfg):
    prompts = ["box"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)

        
def multi_main_ckpt(cfg, ckpt):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        ckpt = ckpt.replace("box", prompt)
        ckpt = ckpt.replace("point", prompt)
        torch.cuda.empty_cache()
        main(cfg, ckpt)


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        LoRA_Sam(model.model, 4)
        
        teacher_model = Model(cfg)
        teacher_model.setup()
        LoRA_Sam(teacher_model.model, 4)

    load_datasets = call_load_dataset(cfg)
    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])

    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    model, optimizer = fabric.setup(model, optimizer)

    optimizer_teacher, scheduler_teacher = configure_opt(cfg, teacher_model.model)
    teacher_model, optimizer_teacher = fabric.setup(teacher_model, optimizer_teacher)

    train_sam(cfg, fabric, model, teacher_model, optimizer, scheduler, optimizer_teacher, scheduler_teacher, train_data, val_data)

    del model, train_data, val_data


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    multi_main(cfg)
    torch.cuda.empty_cache()
