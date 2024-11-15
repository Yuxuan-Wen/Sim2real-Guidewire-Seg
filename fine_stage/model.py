import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam_lora import LoRA_Sam


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_embeddings = None

    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            # checkpoint = "./sam_vit_b_01ec64.pth"
            checkpoint = "/root/autodl-tmp/LearnablePromptSAM-sim-to-real/sam_vit_b_01ec64.pth"
        elif model_type == "vit_l":
            checkpoint = "./sam_vit_l_0b3195.pth"
        elif model_type == "vit_h":
            # checkpoint = "./sam_vit_h_4b8939.pth"
            checkpoint = "/root/autodl-tmp/LearnablePromptSAM-sim-to-real/sam_vit_h_4b8939.pth"
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def setup(self):
        checkpoint = self.get_checkpoint(self.cfg.model.type)
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)

        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        # self.finetune()

    def finetune(self):
        LoRA_Sam(self.model, 4)
        # self.set_norm_layer()
        # self.set_evp_adaptor_layer()
        # self.set_prompt_layer()

    def set_norm_layer(self):
        for name, param in self.model.image_encoder.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def set_evp_adaptor_layer(self):
        for param in self.model.image_encoder.prompt_generator.parameters():
            param.requires_grad = True

    def set_prompt_layer(self):
        self.model.image_encoder.Prompt_Tokens.requires_grad = True

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts):
        image_embeddings = self.encode(images)
        pred_masks, ious, res_masks = self.decode(prompts)
        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        self.image_embeddings = self.model.image_encoder(images)
        return self.image_embeddings 

    def decode(self, prompts, image_embeddings=None):
        if image_embeddings is None:
            image_embeddings = self.image_embeddings

        if prompts is not None:
            
            masks = []
            iou_predictions = []
            low_res_masks = []
            
            if len(prompts) == 2 and isinstance(prorpts, list):
                for prompt_box, prompt_point, embedding in zip(prompts[0], prompts[1], image_embeddings):

                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt_point,
                    boxes=prompt_box,
                    masks=None,
                    )

                    low_res_mask, iou_prediction = self.model.mask_decoder(
                        image_embeddings = embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    mask = F.interpolate(
                        low_res_mask,
                        self.image_shape,
                        mode="bilinear",
                        align_corners=False,
                    )

                    masks.append(mask.squeeze(1))
                    iou_predictions.append(iou_prediction)
                    low_res_masks.append(low_res_mask.squeeze(1))

                masks = torch.stack(masks, dim=0)
                iou_predictions = torch.stack(iou_predictions, dim=0)
                # print(iou_predictions.shape)
                low_res_masks = torch.stack(low_res_masks, dim=0)

                return masks, iou_predictions, low_res_masks

                    
            
            else:
                for prompt, embedding in zip(prompts, image_embeddings):

                    if isinstance(prompt, torch.Tensor):
                        prompt = prompt.to(device=embedding.device)
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=prompt,
                        masks=None,
                    )
                    elif isinstance(prompt, tuple):
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=prompt,
                        boxes=None,
                        masks=None,
                    )


                    low_res_mask, iou_prediction = self.model.mask_decoder(
                        image_embeddings = embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    mask = F.interpolate(
                        low_res_mask,
                        self.image_shape,
                        mode="bilinear",
                        align_corners=False,
                    )

                    # print(masks)
                    # print(masks.shape)

                    masks.append(mask.squeeze(1))
                    iou_predictions.append(iou_prediction)
                    low_res_masks.append(low_res_mask.squeeze(1))

                masks = torch.stack(masks, dim=0)
                iou_predictions = torch.stack(iou_predictions, dim=0)
                # print(iou_predictions.shape)
                low_res_masks = torch.stack(low_res_masks, dim=0)

                return masks, iou_predictions, low_res_masks

        else:
            # for embedding in image_embeddings:

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                self.image_shape,
                mode="bilinear",
                align_corners=False,
            )
            # pred_masks.append(masks.squeeze(1))
            # ious.append(iou_predictions)
            # res_masks.append(low_res_masks)
            
            return masks, iou_predictions, low_res_masks

        # return pred_masks, ious, res_masks

    def get_predictor(self):
        return SamPredictor(self.model)

    def get_generator(self, output_mode):
        return SamAutomaticMaskGenerator(self.model, output_mode=output_mode)
