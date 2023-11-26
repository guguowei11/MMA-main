import torch
import numpy as np
# 计算miou
# SR: Segmentation Result
# GT: Ground Truth
# TP（True Positive）：真正例，模型预测为正例，实际是正例（模型预测为类别1，实际是类别1）
# FP（False Positive）：假正例，模型预测为正例，实际是反例 （模型预测为类别1，实际是类别2）
# FN（False Negative）：假反例，模型预测为反例，实际是正例 （模型预测为类别2，实际是类别1）
# TN（True Negative）：真反例，模型预测为反例，实际是反例 （模型预测为类别2，实际是类别2）

# 准确率（Accuracy），对应：语义分割的像素准确率 PA
# 公式：Accuracy = (TP + TN) / (TP + TN + FP + FN)
# 意义：对角线计算。预测结果中正确的占总预测值的比例（对角线元素值的和 / 总元素值的和


# 计算混淆矩阵
def fast_hist(label_true, label_pred, n_class):
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # np.bincount()会给出索引对应的元素个数
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.epsilon = np.finfo(np.float32).eps  # 防止÷0变成nan

    def pixel_accuracy(self, hist):
        pa=np.diag(hist).sum() / hist.sum()
        return pa

    def mean_pixel_accuracy(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        mpa = np.nanmean(cpa)
        # return cpa[0], cpa[1], cpa[2], cpa[3], cpa[4], mpa
        return cpa[0], cpa[1], cpa[2], mpa
        # return cpa[0], cpa[1], mpa

    def precision(self, hist):
        precision = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        precision = np.nanmean(precision)
        return precision

    def recall(self, hist):
        recall1 = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + self.epsilon)
        recall = np.nanmean(recall1)
        # return recall1[0], recall1[1], recall1[2], recall1[3], recall1[4], recall
        return recall1[0], recall1[1], recall1[2], recall
        # return recall1[0], recall1[1], recall

    def f1_score(self, hist):
        f11 = (np.diag(hist) + self.epsilon) * 2 / (hist.sum(axis=1) * 2 + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        f1 = np.nanmean(f11)
        # return f11[0], f11[1], f11[2], f11[3], f11[4], f1
        return f11[0], f11[1], f11[2], f1
        # return f11[0], f11[1], f1

    def mean_intersection_over_union(self, hist):
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self, hist):
        freq = hist.sum(axis=1) / hist.sum()
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

