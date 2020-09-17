import os
import os.path as osp
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class ClassBalancedSampler(Sampler):
    def __init__(self, data_source, num_samples):
        super(ClassBalancedSampler, self).__init__(data_source)
        self.num_samples = num_samples

        label_count = [0] * len(np.unique(data_source.targets))
        for idx in range(len(data_source)):
            label = data_source.targets[idx]
            label_count[label] += 1

        weights_per_cls = 1.0 / np.array(label_count)
        weights = [weights_per_cls[data_source.targets[idx]]
                   for idx in range(len(data_source))]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True
        ).tolist())

    def __len__(self):
        return self.num_samples


def get_cifar10(root="/home/data2/limingjie/data", mode="demo", batch_size=128, **kwargs):
    if not osp.exists(root):
        root = "/data2/limingjie/data"
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if mode == "demo":
        rho = kwargs["rho"]
        train_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "train"),
                                         train_transform)
        sampler_train = ClassBalancedSampler(train_set, num_samples=5000)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "test"),
                                        test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "4class":
        n_last_class = kwargs["n_last_class"]
        train_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "train"),
                                         train_transform)
        sampler_train = ClassBalancedSampler(train_set, num_samples=20000)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)
        test_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "test"),
                                        test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "complete":
        n_init_each_class = kwargs["n_init_each_class"]
        train_set = datasets.ImageFolder(osp.join(root, "CompleteCIFAR10/train"), train_transform)
        sampler_train = ClassBalancedSampler(train_set, num_samples=10 * n_init_each_class)
        # initialize the weight (only [n_init_each_class] samples in each class)
        sampler_train.weights *= 0
        for label in range(10):
            sampler_train.weights[label*5000:label*5000+n_init_each_class] = 1 / n_init_each_class
        # end initialize
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)
        test_set = datasets.ImageFolder(osp.join(root, "CompleteCIFAR10/test"), test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    else:
        raise Exception(f"[get_cifar10] Mode {mode} not supported!")
    return train_loader, test_loader


def get_eval_cifar10(root="/home/data2/limingjie/data", mode="demo", batch_size=128, **kwargs):
    if not osp.exists(root):
        root = "/data2/limingjie/data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if mode == "demo":
        rho = kwargs['rho']
        train_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "train"), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "test"), transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "4class":
        n_last_class = kwargs["n_last_class"]
        train_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "train"), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "test"), transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "complete":
        train_set = datasets.ImageFolder(osp.join(root, "CompleteCIFAR10/train"), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_set = datasets.ImageFolder(osp.join(root, "CompleteCIFAR10/test"), transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    else:
        raise Exception(f"[get_cifar10] Mode {mode} not supported!")
    return train_loader, test_loader


def get_raw_cifar10(root="/home/data2/limingjie/data", mode="demo", batch_size=128, **kwargs):
    if not osp.exists(root):
        root = "/data2/limingjie/data"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if mode == "demo":
        rho = kwargs["rho"]
        train_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "train"), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "test"), transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "4class":
        n_last_class = kwargs["n_last_class"]
        train_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "train"), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "test"), transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "complete":
        train_set = datasets.ImageFolder(osp.join(root, "CompleteCIFAR10/train"), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_set = datasets.ImageFolder(osp.join(root, "CompleteCIFAR10/test"), transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    else:
        raise Exception(f"[get_cifar10] Mode {mode} not supported!")
    return train_loader, test_loader


def get_augmented_cifar10(root="/home/data2/limingjie/data", mode="demo", batch_size=128, **kwargs):
    if not osp.exists(root):
        root = "/data2/limingjie/data"
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if mode == "demo":
        rho = kwargs["rho"]
        train_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "train"),
                                         train_transform)
        sampler_train = ClassBalancedSampler(train_set, num_samples=15000)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_set = datasets.ImageFolder(osp.join(root, f"ImbalancedCIFAR10-3class/CIFAR10-{rho}", "test"),
                                        test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "4class":
        n_last_class = kwargs["n_last_class"]
        train_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "train"),
                                         train_transform)
        sampler_train = ClassBalancedSampler(train_set, num_samples=20000)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)
        test_set = datasets.ImageFolder(osp.join(root, f"CIFAR10-4class/CIFAR10-{n_last_class}", "test"),
                                        test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif mode == "complete":
        raise NotImplementedError
    else:
        raise Exception(f"[get_augmented_cifar10] Mode {mode} not supported!")
    return train_loader, test_loader



def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # for ix, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = lr[0]
    return


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_tsne(X, y, save_path):
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    x_min, x_max = X.min(0), X.max(0)
    # X_normalized = (X - x_min) / (x_max - x_min)
    X_normalized = X
    plt.figure(figsize=(32, 8))
    color = ['cornflowerblue', 'orange', 'green']
    plt.subplot(1, 4, 1)
    plt.xlim(x_min[0], x_max[0])
    plt.ylim(x_min[1], x_max[1])
    for i in range(X_normalized.shape[0]):
        plt.scatter(X_normalized[i, 0], X_normalized[i, 1], s=5, c=color[y[i]], zorder=y[i]+10)
        # font_size = 9 * (y[i] + 1)
        # plt.text(X_normalized[i, 0], X_normalized[i, 1], str(y[i]), color=color[y[i]],
        #          fontdict={'weight': 'bold', 'size': font_size}, zorder=y[i]+10)
    plt.xticks([])
    plt.yticks([])
    for cls_idx in range(3):
        plt.subplot(1, 4, cls_idx + 2)
        plt.xlim(x_min[0], x_max[0])
        plt.ylim(x_min[1], x_max[1])
        for i in range(X_normalized.shape[0]):
            if y[i] != cls_idx:
                continue
            plt.scatter(X_normalized[i, 0], X_normalized[i, 1], s=5, c=color[y[i]], zorder=y[i] + 10)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close("all")


def visualize_tsne_2(X, y, save_path):
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    x_min, x_max = X.min(0), X.max(0)
    # X_normalized = (X - x_min) / (x_max - x_min)
    X_normalized = X
    plt.figure(figsize=(24, 8))
    color = ['cornflowerblue', 'orange', 'green']
    plt.subplot(1, 3, 1)
    plt.xlim(x_min[0], x_max[0])
    plt.ylim(x_min[1], x_max[1])
    for i in range(X_normalized.shape[0]):
        plt.scatter(X_normalized[i, 0], X_normalized[i, 1], s=5, c=color[y[i]], zorder=y[i]+10)
        # font_size = 9 * (y[i] + 1)
        # plt.text(X_normalized[i, 0], X_normalized[i, 1], str(y[i]), color=color[y[i]],
        #          fontdict={'weight': 'bold', 'size': font_size}, zorder=y[i]+10)
    plt.xticks([])
    plt.yticks([])
    for cls_idx in range(2):
        plt.subplot(1, 3, cls_idx + 2)
        plt.xlim(x_min[0], x_max[0])
        plt.ylim(x_min[1], x_max[1])
        for i in range(X_normalized.shape[0]):
            if y[i] != cls_idx:
                continue
            plt.scatter(X_normalized[i, 0], X_normalized[i, 1], s=5, c=color[y[i]], zorder=y[i] + 10)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close("all")


def draw_hist(data, title, xlabel, ylabel, path):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.hist(data, bins=10)
    plt.tight_layout()
    plt.savefig(path)


if __name__ == '__main__':
    # train_loader, _ = get_augmented_cifar10(rho=100)
    train_loader, _ = get_cifar10(mode="complete", n_init_each_class=100)
    print(train_loader.sampler.weights[0], train_loader.sampler.weights.sum())
    train_loader.sampler.weights[0] = 0.
    print(train_loader.dataset.imgs[0])
    print(train_loader.sampler.weights[0], train_loader.sampler.weights.sum())
    print(len(train_loader.dataset.imgs))
    for i, (path, label) in enumerate(train_loader.dataset.imgs):
        assert i // 5000 == label, print(f"Image no. {i} has label {label}.")
