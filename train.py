import os

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from args.arg_parse import get_argparse
from dataset.dataset import MyDataset
from models.LeNet import LeNet

from collections import defaultdict


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def train(train_loader, model, device, loss_fn, optimizer):
    # 设置为训练模式
    model.train()

    train_loss = 0.0
    train_acc = 0.0
    train_num = len(train_loader.dataset)

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        # 前向传播
        output = model(data)

        # 计算损失
        loss = loss_fn(output, label)

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # argmax 找出最大值的索引
        pred = output.argmax(dim=1)
        # 先保存每一个batch图片预测正确的个数
        train_acc += pred.eq(label).sum().cpu().item()
        # 先保存每一个batch的loss
        train_loss += loss.cpu().item()

    train_loss /= train_num
    train_acc /= train_num
    return train_loss, train_acc


def val(val_loader, model, device, loss_fn):
    # 设置为验证模式
    model.eval()

    val_loss = 0.0
    val_acc = 0.0
    val_num = len(val_loader.dataset)

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            pred = output.argmax(dim=1)
            loss = loss_fn(output, label)

            val_loss += loss.cpu().item()
            val_acc += pred.eq(label).sum().cpu().item()

        val_loss /= val_num
        val_acc /= val_num
    return val_loss, val_acc


def save_model(args, epoch, model, val_acc):
    # 每隔n个epoch保存一次模型
    if epoch % args.save_model_epoch == 0:
        torch.save(model.state_dict(),
                   os.path.join(args.save_model_path, args.save_model_name + str(epoch) + '_' + str(val_acc) + '.pt'))

    # 保存最后一个模型
    if epoch == args.epochs:
        torch.save(model.state_dict(), os.path.join(args.save_model_path, 'last_' + str(val_acc) + '.pt'))
        # 最好的模型改名 best.pt -> best_<acc>.pt
        os.renames(os.path.join(args.save_model_path, 'best.pt'),
                   os.path.join(args.save_model_path, 'best_' + str(args.best_acc) + '.pt'))

    # 保存最好的模型
    if val_acc > args.best_acc:
        args.best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(args.save_model_path, 'best.pt'))


def plot_training_history(args, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    # 画训练和验证时的损失
    ax1.plot(history['train_loss'], label='train loss')
    ax1.plot(history['val_loss'], label='val loss')

    # ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # 画训练和验证时的准确率
    ax2.plot(history['train_acc'], label='train acc')
    ax2.plot(history['val_acc'], label='val acc')

    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    fig.suptitle('Training History')

    plt.savefig(os.path.join(args.save_picture_path, args.save_picture_name))
    plt.show()


def main():
    args = get_argparse().parse_args()

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 读取训练集
    train_data = MyDataset(args, is_train=True)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    # 读取验证集
    val_data = MyDataset(args, is_train=False)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # train_loader, val_loader = load_data_fashion_mnist(args.batch_size, 32)

    # 设置使用的设备
    device = torch.device(args.device)

    # 实例化模型
    # model = vgg11()
    model = LeNet()
    model = model.to(device)

    # 加载模型继续训练
    if os.path.exists(os.path.join(args.weight_path, args.weight_name)):
        model.load_state_dict(torch.load(os.path.join(args.weight_path, args.weight_name)))
        print('---------------加载模型继续训练---------------')

    # 设置优化器
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)

    # 设置损失函数
    loss_fn = CrossEntropyLoss()

    history = defaultdict(list)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(train_loader, model, device, loss_fn, optimizer)
        print('Train Epoch_{}:\ttrain loss:{:.6f}\ttrain acc:{:.2%}'.format(epoch, train_loss, train_acc))

        val_loss, val_acc = val(val_loader, model, device, loss_fn)
        print('Val Epoch_{}:\tval loss:{:.6f}\tval acc:{:.2%}\n'.format(epoch, val_loss, val_acc))

        save_model(args, epoch, model, val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    plot_training_history(args, history)


if __name__ == '__main__':
    main()
