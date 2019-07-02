import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data(opt):
    if opt['dataset'] in ['mnist']:
        transform = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        transform = transforms.Compose(transform)
        train_data = datasets.MNIST(opt['data_dir'], download=True, train=True, transform=transform)
        opt['nsize'] = 32
        opt['nc'] = 1
    elif opt['dataset'] in ['cifar10']:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform)
        train_data = datasets.CIFAR10(opt['data_dir'], download=True, train=True, transform=transform)
        opt['nsize'] = 32
        opt['nc'] = 3
    else:
        raise AssertionError()

    train_loader = DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)

    return train_data, train_loader