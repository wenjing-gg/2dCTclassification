import torch

def load_weights_info(weights_path):
    """
    加载并打印 PyTorch 权重文件的信息，包括键名称和权重形状。

    Args:
        weights_path (str): 权重文件路径。
    """
    # 检查文件是否存在
    try:
        assert weights_path.endswith(".pth"), "Provided file is not a .pth file"
        state_dict = torch.load(weights_path, map_location='cpu')  # 加载权重文件

        # 检查文件内容
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]  # 提取模型权重字典
        else:
            raise ValueError("Unexpected weight file format!")

        print(f"加载的权重文件包含 {len(state_dict)} 个参数:\n")

        for key, value in state_dict.items():
            print(f"{key}: shape {tuple(value.shape)}")

    except FileNotFoundError:
        print(f"权重文件 {weights_path} 不存在。")
    except Exception as e:
        print(f"读取权重文件时出错: {e}")

# 权重文件路径
weights_path = "/home/yuwenjing/DeepLearning/swin_transformer/swin_base_patch4_window7_224_22k.pth"

# 读取并打印权重信息
load_weights_info(weights_path)
