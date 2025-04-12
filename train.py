import os
import argparse
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
# 确保从正确的 utils 导入 Dataset 类 (假设路径是 tools.utils)
from tools.utils import SegmentationDataset
torch.backends.cudnn.enabled = False
from sklearn.model_selection import train_test_split
# 确保从正确的 engine 导入所需函数 (假设路径是 tools.engine)
from tools.engine import create_model, create_optimizer, train, create_lr_scheduler

"""
可供选择的模型：fcn_resnet50 vgg16unet deeplabv3_resnet50 (以及 engine.py 中定义的其他模型)
"""
def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)

    parser.add_argument('--batch-size', default=64, type=int) # 默认值可能对分割任务过大
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Model parameters
    parser.add_argument('--model', default='unet', type=str, metavar='MODEL', # 将默认改为 unet 示例
                        help='Name of model to train')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum') # 主要用于 SGD
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 注意：bool类型的参数处理方式，直接给 True/False 可能不工作，通常用 store_true/store_false
    # parser.add_argument("--aux", default=True, type=bool, help="auxilier loss") # 这种方式可能不正确
    parser.add_argument('--aux', action='store_true', help='use auxiliary loss (if model supports it)') # 正确方式1: 默认False
    parser.add_argument('--no-aux', action='store_false', dest='aux', help='do not use auxiliary loss') # 正确方式2: 设置默认True
    parser.set_defaults(aux=False) # 设定 aux 的默认值

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--in_channels', default=3, type=int,
                        help='channels')
    parser.add_argument('--num_classes', default=4, type=int, # 将默认改为 4 (根据你的数据)
                        help='the number of classes')
    # 修改了默认路径以反映 Kaggle 环境
    parser.add_argument('--data-path', type=str, default="/kaggle/working/converted_images/",
                        help='data-path (parent directory containing train/test images)')
    parser.add_argument('--label-path', type=str,
                        default="/kaggle/input/datastes2/dataset/labels/",
                        help='label-path (parent directory containing train/test labels)')

    parser.add_argument('--weight-path', type=str,
                        default="/kaggle/working/best_segmentation_model.pth", # 修改了默认保存路径
                        help='weight-path')
    # parser.add_argument("--amp", default=False, type=bool, # 这种方式可能不正确
    #                     help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--amp', action='store_true', help='Use torch.cuda.amp for mixed precision training')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Do not use torch.cuda.amp')
    parser.set_defaults(amp=False) # 设定 amp 的默认值

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 用来保存训练以及验证过程中信息 (这行似乎未被使用，注释掉)
    # results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    batch_size = args.batch_size
    # 自动计算 num_workers
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"Using {num_workers} dataloader workers")

    # --- 插入：正确的递归文件查找逻辑 ---
    image_folder = args.data_path
    label_folder = args.label_path
    print(f"开始在图像目录中查找文件: {image_folder}")
    all_image_paths = []
    # 确保包含你转换后的图像文件扩展名 (通常是 .png)
    valid_image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    for root, _, files in os.walk(image_folder): # 递归遍历目录
        for file in files:
            if file.lower().endswith(valid_image_extensions): # 检查扩展名
                all_image_paths.append(os.path.join(root, file)) # 添加文件的完整路径
    all_image_paths.sort() # 排序
    print(f"找到 {len(all_image_paths)} 个潜在的图像文件。")

    print(f"开始在标签目录中查找文件: {label_folder}")
    all_label_paths = []
    valid_label_extensions = ('.png',) # 标签文件必须是 .png
    for root, _, files in os.walk(label_folder): # 递归遍历目录
        for file in files:
             if file.lower().endswith(valid_label_extensions): # 检查扩展名
                all_label_paths.append(os.path.join(root, file)) # 添加文件的完整路径
    all_label_paths.sort() # 排序
    print(f"找到 {len(all_label_paths)} 个潜在的标签文件。")

    # 在分割前进行基本的健全性检查
    if len(all_image_paths) == 0:
         raise ValueError(f"在指定的图像路径 '{image_folder}' 或其子目录中未找到有效的图像文件！")
    if len(all_label_paths) == 0:
         raise ValueError(f"在指定的标签路径 '{label_folder}' 或其子目录中未找到有效的标签文件！")
    if len(all_image_paths) != len(all_label_paths):
        # 如果图像和标签数量不匹配，可能需要更复杂的基于文件名的匹配逻辑，但首先是报错
        # 尝试基于文件名进行配对 (如果数量不匹配) - 这是一个简单的示例，可能需要调整
        print(f"警告：找到的图像文件数量 ({len(all_image_paths)}) 与标签文件数量 ({len(all_label_paths)}) 不匹配。将尝试基于文件名进行匹配...")
        img_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in all_image_paths}
        lbl_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in all_label_paths}
        matched_img_paths = []
        matched_lbl_paths = []
        for basename, img_path in img_basenames.items():
            if basename in lbl_basenames:
                matched_img_paths.append(img_path)
                matched_lbl_paths.append(lbl_basenames[basename])
        all_image_paths = sorted(matched_img_paths)
        all_label_paths = sorted(matched_lbl_paths)
        if not all_image_paths:
             raise ValueError("图像和标签文件名无法匹配，请检查文件名是否一致（除了扩展名）！")
        print(f"文件名匹配后，找到 {len(all_image_paths)} 对有效的图像/标签。")


    # 将找到的完整文件列表赋值给后续 train_test_split 使用的变量
    image_paths = all_image_paths
    label_paths = all_label_paths
    # --- 插入的代码结束 ---


    # 使用修正后的文件列表进行分割
    # 注意：如果总样本数很少（比如只有2），test_size=0.2 仍然可能导致训练或测试集过小
    # 考虑是否需要这个分割，或者调整 test_size
    if len(image_paths) > 1: # 只有多于1个样本时才分割
        train_image_paths, test_image_paths, train_label_paths, test_label_paths = train_test_split(
            image_paths, label_paths, test_size=0.2, random_state=42 # 考虑调整 test_size 或完全使用训练集
        )
        print(f"分割后 - 训练集大小: {len(train_image_paths)}")
        print(f"分割后 - 测试集大小: {len(test_image_paths)}")
    elif len(image_paths) == 1: # 如果只有一个样本，无法分割，全部用于训练
        print("警告：只找到一个样本，将全部用于训练，测试集为空。")
        train_image_paths, train_label_paths = image_paths, label_paths
        test_image_paths, test_label_paths = [], []
    else: # 不应该发生，因为前面有检查
         raise ValueError("未能加载任何图像/标签对。")


    # --- 数据预处理和转换 ---
    # 注意：分割任务的 transform 通常比较复杂，可能需要自定义，这里仅作示例
    # 并且通常 transform 应用在 Dataset 的 __getitem__ 中
    # 这里的 transform 主要是针对输入模型的 Tensor 格式
    # ToTensor 会将 PIL Image [0, 255] 转为 [0.0, 1.0] Tensor (C x H x W)
    # Normalize 使用 ImageNet 均值和标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 对于分割任务，标签通常不需要 Normalize，只需要转成 Tensor
    # 这些转换最好在 Dataset 的 __getitem__ 中根据需要应用

    # --- 创建 Dataset 和 DataLoader ---
    # 注意：transform 应该传递给 Dataset 类，让它在 __getitem__ 中应用
    train_dataset = SegmentationDataset(train_image_paths, train_label_paths, transform=None) # 暂时不传 transform
    val_dataset = SegmentationDataset(test_image_paths, test_label_paths, transform=None)    # 暂时不传 transform

    # 确保 DataLoader 使用正确的 collate_fn (默认的可能不适用于分割)
    # 如果 SegmentationDataset 返回的是 (PIL Image, PIL Image)，需要 collate_fn 转 Tensor
    # 如果 SegmentationDataset 内部已经转了 Tensor，collate_fn=None 可能OK
    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=None) # 确认是否需要自定义 collate_fn
    # 测试集通常不需要 shuffle
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size, # 验证时 batch size 可以适当增大
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=None) # 确认是否需要自定义 collate_fn


    # --- 创建模型 ---
    # 注意：engine.py 中的 create_model 可能还需要修改以接收和传递 in_channels
    print(f"正在创建模型: {args.model}")
    model = create_model(aux=args.aux, num_classes=args.num_classes, model_name=args.model) # 可能缺少 in_channels
    model.to(device)

    # --- 打印第一个批次的信息 (调试用) ---
    try:
        print("尝试加载第一个训练批次...")
        for images, labels in train_loader:
            print(f"加载成功 - Images shape: {images.shape}") # 确认形状是否符合预期 e.g., [B, C, H, W]
            print(f"加载成功 - Labels shape: {labels.shape}")  # 确认形状是否符合预期 e.g., [B, H, W]
            # 检查标签值的范围
            print(f"加载成功 - Labels min: {labels.min()}, max: {labels.max()}, dtype: {labels.dtype}")
            break # 只检查第一个批次
    except Exception as e:
        print(f"加载第一个批次时出错: {e}")
        print("请检查 SegmentationDataset 的 __getitem__ 实现以及 DataLoader 的 collate_fn 设置。")
        return # 无法继续，退出

    # --- 打印模型参数量 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_million = total_params / 1e6
    print(f"模型参数量 (Parameters M): {params_in_million:.2f}")

    # --- 创建优化器和学习率调度器 ---
    optimizer = create_optimizer(args.opt, args.lr, model)
    # 确保 lr_scheduler 的步数正确 (len(train_loader) 是每个 epoch 的步数)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # --- 开始训练 ---
    print("开始训练...")
    train(args.epochs, train_loader, val_loader, optimizer, model, args.aux, args.weight_path, device, args.batch_size, args.num_classes, lr_scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # 可以在这里覆盖或检查 args 的值
    print("解析得到的参数:")
    print(args)
    main(args)
