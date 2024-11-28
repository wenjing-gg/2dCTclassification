from PIL import Image
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
            img (Tensor): 预处理后的灰度图像张量。
            label (int): 图像的类别标签。
        """
        try:
            # 打开图像文件
            img = Image.open(self.images_path[item])
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到图像文件: {self.images_path[item]}")
        except Exception as e:
            raise RuntimeError(f"加载图像文件 {self.images_path[item]} 时发生错误: {e}")

        # 强制将图像转换为灰度模式
        if img.mode != 'RGB':
            # print(f"图像 {self.images_path[item]} 的模式为 {img.mode}，正在转换为灰度模式（L）。")
            img = img.convert('RGB')

        # 获取图像对应的标签
        label = self.images_class[item]

        # 应用图像预处理
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        """
        自定义的批处理函数，用于 DataLoader。

        Args:
            batch (list): 一个批次的数据样本，每个样本是 (img, label) 的元组。

        Returns:
            images (Tensor): 一个批次的图像张量，形状为 [batch_size, 1, H, W]。
            labels (Tensor): 一个批次的标签张量，形状为 [batch_size]。
        """
        # 解压批次中的图像和标签
        images, labels = tuple(zip(*batch))

        # 将图像堆叠为一个张量
        images = torch.stack(images, dim=0)
        # 将标签转换为张量
        labels = torch.as_tensor(labels)

        return images, labels
