import argparse


def get_argparse():
    # 获取解析器
    parser = argparse.ArgumentParser()

    # 数据集
    parser.add_argument('--train_dir', default='data/train', type=str, help='训练集所在的文件夹名')
    parser.add_argument('--val_dir', default='data/val', type=str, help='验证集所在的文件夹名')
    parser.add_argument('--img_size', default=32, type=int, help='图片resize后的长和宽')

    # 加载模型继续训练
    parser.add_argument('--weight_path', default='output', type=str, help='继续训练的模型的路径')
    parser.add_argument('--weight_name', default='model200_0.9455472773638682.pt', type=str, help='继续训练的模型的名字')

    # 训练过程
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'], help='训练使用的设备')
    parser.add_argument('--epochs', default=20, type=int, help='训练的轮次')
    parser.add_argument('--batch_size', default=512, type=int, help='训练的批次大小')
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'], help='训练用的优化器')
    parser.add_argument('--lr', default=0.001, type=float, help='训练时的学习率')

    # 保存模型
    parser.add_argument('--save_model_path', default='output', type=str, help='模型保存的文件夹')
    parser.add_argument('--save_model_name', default='model', type=str, help='模型保存的名字')
    parser.add_argument('--save_model_epoch', default=10, type=int, help='模型每隔n个epoch保存一次')
    parser.add_argument('--best_acc', default=0.5, type=float, help='保存最好的模型准确率必须大于best_acc')

    # 保存训练过程中的loss和acc的可视化
    parser.add_argument('--save_picture_path', default='output', type=str, help='图片保存的文件夹')
    parser.add_argument('--save_picture_name', default='Training_History.png', help='图片保存的名字')
    return parser

