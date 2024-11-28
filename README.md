# 数据集说明

## 数据集路径格式

数据集的路径格式应类似如下所示：
/home/yuwenjing/data/sm_6/test/0/1_2_slice133.png
其中，每张 `.png` 文件表示一个样本的 ROI 切片。你可以使用自己的方法处理原始数据，得到相应的 ROI 切片。

## 数据集结构

假设数据集存放在路径 `/home/yuwenjing/data/sm_6/` 下，数据集的目录结构大致如下：
/home/yuwenjing/data/sm_6/
/home/yuwenjing/data/sm_6/
├── test/                # 测试集数据
│   ├── 0/              # 类别 0 的测试样本
│   │   ├── 1_2_slice133.png
│   │   ├── 1_2_slice134.png
│   │   ├── 1_2_slice135.png
│   │   └── ...         # 更多的样本切片
│   ├── 1/              # 类别 1 的测试样本
│   │   ├── 1_3_slice133.png
│   │   ├── 1_3_slice134.png
│   │   ├── 1_3_slice135.png
│   │   └── ...         # 更多的样本切片
│   └── ...             # 其他类别的测试数据
├── train/               # 训练集数据
│   ├── 0/              # 类别 0 的训练样本
│   │   ├── 1_4_slice133.png
│   │   ├── 1_4_slice134.png
│   │   ├── 1_4_slice135.png
│   │   └── ...         # 更多的样本切片
│   ├── 1/              # 类别 1 的训练样本
│   │   ├── 1_5_slice133.png
│   │   ├── 1_5_slice134.png
│   │   ├── 1_5_slice135.png
│   │   └── ...         # 更多的样本切片
│   └── ...             # 其他类别的训练数据
├── VIT/                 # VIT 模型相关文件
│   ├── weights/         # 模型权重
│   │   ├── best_model.pth
│   │   ├── best_model_auc.pth
│   │   └── ...
│   └── vit_base_patch16_224_in21k.pth
└── README.md            # 项目的说明文件



## 处理数据

你可以根据自己的需求处理原始数据，将其转化为适合模型训练和验证的 ROI 切片。例如，可以使用医学图像处理工具（如 SimpleITK、pydicom、OpenCV 等）来提取目标区域并保存为 `.png` 文件。

## 注意事项

- 请确保每个样本的 `.png` 文件命名规则一致，便于后续加载和处理。
- 数据集中应包含 `train` 和 `test` 两个子目录，用于分别存放训练集和测试集数据。



