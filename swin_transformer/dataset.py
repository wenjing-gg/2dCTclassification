from PIL import Image
import os
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """
    自定义数据集类，用于加载图片路径和对应的标签。

    Attributes:
        images_path (list): 图像文件的路径列表。
        images_class (list): 图像对应的类别标签列表。
        transform (callable, optional): 图像的预处理转换。
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        Args:
            images_path (list): 图像文件路径列表。
            images_class (list): 图像对应的类别标签列表。
            transform (callable, optional): 图像的预处理转换。
        """
        if len(images_path) != len(images_class):
            raise ValueError("images_path 和 images_class 的长度不一致！")

        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        """返回数据集的样本数量。"""
        return len(self.images_path)

    def __getitem__(self, item):
        """
        获取指定索引处的图像和标签。

        Args:
            item (int): 索引值。

        Returns:
            img (Tensor): 预处理后的图像张量（三通道）。
            label (int): 图像的类别标签。
        """
        # 当前图像路径和文件名
        current_path = self.images_path[item]
        folder, current_name = os.path.split(current_path)

        # 提取当前图像的编号
        try:
            base_name, ext = os.path.splitext(current_name)
            base, slice_id = base_name.rsplit('_slice', 1)
            slice_id = int(slice_id)
        except ValueError:
            raise ValueError(f"文件名格式错误: {current_name}, 期望格式为 <prefix>_slice<number>.png")

        # 构造相邻图片的文件名
        prev_name = f"{base}_slice{slice_id - 1}{ext}"
        next_name = f"{base}_slice{slice_id + 1}{ext}"

        prev_path = os.path.join(folder, prev_name)
        next_path = os.path.join(folder, next_name)

        # 打开当前图片
        try:
            img = Image.open(current_path).convert("L")
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到图像文件: {current_path}")

        # 打开相邻图片，如果存在
        img_prev = Image.open(prev_path).convert("L") if os.path.exists(prev_path) else None
        img_next = Image.open(next_path).convert("L") if os.path.exists(next_path) else None

        # 构造三通道图像
        if img_prev is not None and img_next is not None:
            img_combined = Image.merge("RGB", (img_prev, img, img_next))
        else:
            # 如果相邻图片不存在，将当前图片重复三次
            img_combined = Image.merge("RGB", (img, img, img))

        # 获取图像对应的标签
        label = self.images_class[item]

        # 应用图像预处理
        if self.transform is not None:
            img_combined = self.transform(img_combined)

        return img_combined, label

    @staticmethod
    def collate_fn(batch):
        """
        自定义的批处理函数，用于 DataLoader。

        Args:
            batch (list): 一个批次的数据样本，每个样本是 (img, label) 的元组。

        Returns:
            images (Tensor): 一个批次的图像张量，形状为 [batch_size, 3, H, W]。
            labels (Tensor): 一个批次的标签张量，形状为 [batch_size]。
        """
        # 解压批次中的图像和标签
        images, labels = tuple(zip(*batch))

        # 将图像堆叠为一个张量
        images = torch.stack(images, dim=0)
        # 将标签转换为张量
        labels = torch.as_tensor(labels)

        return images, labels
