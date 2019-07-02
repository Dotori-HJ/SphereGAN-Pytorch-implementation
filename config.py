import os
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='SCGN:Self Converging Generative Network.')
    parser.add_argument('--dataset', type=str,
                        choices=['mnist', 'cifar10'],
                        help='name of dataset cnndnnrnngto use [mnist, cifar10]')
    parser.add_argument('--train', dest='train', action='store_true', help='train network')
    parser.add_argument('--epoch', type=int, default=1000, help='input total epoch')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_checkpoint', type=int, default=500)
    parser.add_argument('--print_log', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--b1', type=float, default=0)
    parser.add_argument('--b2', type=float, default=0.99)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ngs', type=int, default=512)

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--experiments_dir', type=str, default='experiments')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--z_data_dir', type=str, default='z_data')
    opt = vars(parser.parse_args())
    init(opt)
    return opt

def init(opt):
    path = os.path.join(opt['experiments_dir'], opt['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)
    opt['experiments_dir'] = path

    path = os.path.join(opt['checkpoint_dir'], opt['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(opt['checkpoint_dir'], opt['dataset'], 'checkpoint.pth')
    opt['checkpoint'] = path