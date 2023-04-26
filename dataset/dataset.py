from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image
import os


# 自定义数据
class MyDataset(Dataset):
    # 初始化一下必要的东西 像图片和标签的位置
    def __init__(self, args, is_train=True):
        super(MyDataset, self).__init__()

        if is_train:
            self.img_path = args.train_dir
        else:
            self.img_path = args.val_dir

        # 保存图片名的列表
        self.img_list = os.listdir(self.img_path)

        # 标准化
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 数据增强
        if is_train:
            self.transform = T.Compose([
                # 变换大小
                T.Resize((args.img_size, args.img_size)),
                # 其他操作
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),

                T.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),

                T.ToTensor(),
                self.normalize
            ])

    # 返回图片和标签 可以对图片做一些变换
    def __getitem__(self, index):
        # index是下标 从0到n-1 (n为图片的数量)
        # 根据下标获取图片
        img_name = self.img_list[index]
        # 获取图片
        img = Image.open(os.path.join(self.img_path, img_name))
        # 变换
        img = self.transform(img)

        # 获取标签
        label = 0 if 'cat' in img_name else 1

        return img, label

    # 返回图片的数量
    def __len__(self):
        return len(self.img_list)


