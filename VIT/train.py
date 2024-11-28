import os
import math
import argparse
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate


def wait_for_available_gpu():
    """依次检查显卡是否空闲，一旦找到空闲的显卡就返回其ID"""
    while True:
        for device_id in range(3):  # 检查cuda:0到cuda:2
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            if free_mem >= 0.4 * total_mem:  # 如果空闲显存接近总显存，表示显卡未被占用
                print(f"Device cuda:{device_id} is now fully available. Starting training...")
                return device_id  # 返回找到的空闲设备ID
            else:
                print(f"Device cuda:{device_id} is currently in use. Free memory: {free_mem} bytes, "
                      f"Total memory: {total_mem} bytes.")
        # 如果没有找到空闲显卡，等待 30 秒后重新检查
        print("No available GPU found. Waiting...")
        time.sleep(30)


def read_data(data_dir):
    """
    从指定目录中读取数据路径和对应的标签。
    Args:
        data_dir (str): 数据集目录路径。
    Returns:
        images_path (list): 图像路径列表。
        images_label (list): 标签列表。
    """
    images_path = []
    images_label = []

    # 遍历文件夹，获取图像路径和标签
    for label, class_dir in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if img_file.endswith((".jpg", ".png", ".jpeg")):
                images_path.append(os.path.join(class_path, img_file))
                images_label.append(label)

    print(f"从 {data_dir} 加载了 {len(images_path)} 张图片，包含 {len(set(images_label))} 个类别。")
    return images_path, images_label


def main(args):
    # 等待并选择一个空闲的显卡
    available_device_id = wait_for_available_gpu()
    # 设置设备
    device = torch.device(f"cuda:{available_device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} for training.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 加载训练集和测试集
    train_images_path, train_images_label = read_data(args.train_path)
    test_images_path, test_images_label = read_data(args.test_path)

    print(f"训练集: {len(train_images_path)} 张图片")
    print(f"测试集: {len(test_images_path)} 张图片")

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # 实例化数据集
    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label,
                              transform=data_transform["train"])
    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label,
                             transform=data_transform["test"])

    # 实例化数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers every process')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              pin_memory=True, num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 用于跟踪最佳 AUC 和权重
    best_auc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch)
        scheduler.step()

        # validate
        test_loss, test_acc, test_auc, test_youden, test_threshold, test_acc_cutoff = evaluate(
            model=model, data_loader=test_loader, device=device, epoch=epoch)

        # 保存最佳 AUC 对应的权重
        if test_auc > best_auc:
            best_auc = test_auc
            best_acc4auc = test_acc_cutoff
            best_epoch = epoch
            torch.save(model.state_dict(), "./weights/best_model_auc.pth")
            print(f"保存最佳 AUC 权重，当前 AUC: {test_auc:.4f}, 发生在 epoch: {epoch}")

        # 显示所有指标
        print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
              f"AUC: {test_auc:.4f}, Best AUC: {best_auc:.4f}, "
              f"Youden Index: {test_youden:.4f}, Best Threshold: {test_threshold:.4f}, "
              f"Test Acc with Youden: {test_acc_cutoff:.4f}")

        # 记录到 TensorBoard
        tags = ["train_loss", "train_acc", "test_loss", "test_acc", "test_auc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], test_loss, epoch)
        tb_writer.add_scalar(tags[3], test_acc, epoch)
        tb_writer.add_scalar(tags[4], test_auc, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

    print(f"训练完成，最佳 AUC: {best_auc:.4f}, 此时的ACC: {best_acc4auc:.4f} , 发生在 epoch: {best_epoch}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lrf', type=float, default=1e-7)

    # 数据集路径
    parser.add_argument('--train-path', type=str, default="/home/yuwenjing/data/sm_6/train",
                        help='root directory of the training set')
    parser.add_argument('--test-path', type=str, default="/home/yuwenjing/data/sm_6/test",
                        help='root directory of the testing set')

    # 模型权重路径
    parser.add_argument('--weights', type=str, default='/home/yuwenjing/DeepLearning/VIT/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()
    main(opt)
