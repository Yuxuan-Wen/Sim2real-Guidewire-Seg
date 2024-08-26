# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/13 11:09
    @filename: metric.py
    @software: PyCharm
"""

import torch

from .confusion_matrix import confusion_matrix

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        self.matrix = torch.zeros((num_classes, num_classes))

    def update(self, output, target):
        if (output.dim() == 4 or target.dim() == 2) and self.num_classes != 1:
            output = torch.max(output, dim=1)[1]
        if self.num_classes == 1:
            output = torch.where(output >= 0.5, 1, 0)
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        matrix = confusion_matrix(output.detach().cpu(), target.detach().cpu(), num_classes)
        # if self.matrix.device != matrix.device:
        #     self.matrix = self.matrix.to(matrix.device)
        self.matrix += matrix.detach().cpu()

    def evaluate(self):
        result = dict()
        FP = self.matrix.sum(0) - torch.diag(self.matrix)
        FN = self.matrix.sum(1) - torch.diag(self.matrix)
        TP = torch.diag(self.matrix)
        TN = self.matrix.sum() - (FP + FN + TP)
        precision = TP / (TP + FP)
        acc = (TP + TN) / (TP+FP+FN+TN)
        recall = TP / (TP + FN)
        npv = TN/(TN+FN)
        fnr = FN / (TP+FN)
        fpr = FP / (FP+TN)
        mcc = (TP*TN-FP*FN) / torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # f1 =  2 * (precision * recall) / (precision + recall)
        specficity = TN / (TN + FP)
        iou = TP / (TP + FN +FP)
        dice = (2*TP) / (2*TP + FN + FP)
        
        sen = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        
        '''
        sensitivity = TP / (TP + FN)
        sensitivity 表示灵敏度，表示对正例的预测能力（越高越好），数值上等于召回率。

        specificity = TN / (TN + FP)
        specificity 表示特异度，表示对负例的预测能力（越高越好）。

        可以看到sensitivity 、specificity 就是归一化混淆矩阵对角线上的值。

        F1 = 2 * precision * recall / (precision + recall)
        '''
        result["FP"] = FP
        result["FN"] = FN
        result["TP"] = TP
        result["TN"] = TN
        result["precision"] = precision
        result["acc"] = acc
        
        result["f1"] = F1
        result["sen"] = sen
        
        result["dice"] = dice
        result["specifity"] = specficity
        result["iou"] = iou
        result["recall"] = recall
        result["mk"] = precision + npv - 1
        result["npv"] = npv
        result["mcc"] = mcc
        result["bm"] = (recall+specficity - 1)
        result["fnr"] = fnr
        result["fpr"] = fpr
        result["tpr"] = recall
        result["tnr"] = specficity
        return result

class SeparateMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.ious = []
        self.dices = []

    def update(self, output, target):
        if (output.dim() == 4 or target.dim() == 2) and self.num_classes != 1:
            output = torch.max(output, dim=1)[1]
        if self.num_classes == 1:
            output = torch.where(output >= 0.5, 1, 0)
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        matrix = confusion_matrix(output.detach().cpu(), target.detach().cpu(), num_classes)
        FP = matrix.sum(0) - torch.diag(matrix)
        FN = matrix.sum(1) - torch.diag(matrix)
        TP = torch.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
        iou = TP / (TP + FN + FP)
        dice = (2 * TP) / (2 * TP + FN + FP)
        self.ious.append(iou)
        self.dices.append(dice)

    def evaluate(self):
        return self.dices, self.ious