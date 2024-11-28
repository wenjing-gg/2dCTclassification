import os
import sys
import json
import pickle
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torch.nn.functional as F


import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        # 获取类别数
        num_classes = pred.size(1)
        
        # 计算平滑标签
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)  # one-hot 编码
        target = (1 - self.epsilon) * one_hot + self.epsilon / num_classes  # 标签平滑
        
        # 计算交叉熵损失
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(target * log_pred, dim=1)
        
        # 进行reduction（如求均值或求和）
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = LabelSmoothingCrossEntropy(epsilon=0.1)  # 设置标签平滑的ε
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 存储所有真实标签和预测标签
    all_labels = []
    all_preds = []
    all_probs = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 获取模型的概率输出
        pred = model(images.to(device))
        probabilities = F.softmax(pred, dim=1)  # 将 logits 转换为概率
        pred_classes = torch.max(probabilities, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                              accu_loss.item() / (step + 1),
                                                                              accu_num.item() / sample_num)

        # 收集所有真实标签、预测类别和预测概率
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_classes.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

    # 计算混淆矩阵
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # 特异性和敏感性
    if cm.shape == (2, 2):  # 仅对二分类问题计算
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"Specificity: {specificity:.3f}")
        print(f"Sensitivity: {sensitivity:.3f}")

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # AUC 和 Youden Index
    if all_probs.shape[1] == 2:
        # 计算类别 1 的 AUC
        auc = roc_auc_score(all_labels, all_probs[:, 1])

        # 计算 Youden Index
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        youden_index = tpr - fpr
        best_threshold_index = np.argmax(youden_index)
        best_threshold = thresholds[best_threshold_index]
        best_youden = youden_index[best_threshold_index]

        print(f"AUC: {auc:.3f}")
        print(f"Best Youden index: {best_youden:.3f} at threshold: {best_threshold:.3f}")

        # 使用 Youden cutoff 重新计算 test_acc
        pred_classes_cutoff = (all_probs[:, 1] >= best_threshold).astype(int)
        test_acc_cutoff = np.mean(pred_classes_cutoff == all_labels)
        print(f"Test Accuracy with Youden cutoff: {test_acc_cutoff:.3f}")
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"Multiclass AUC: {auc:.3f}")
        best_threshold = None
        best_youden = None
        test_acc_cutoff = None

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, best_youden, best_threshold, test_acc_cutoff
