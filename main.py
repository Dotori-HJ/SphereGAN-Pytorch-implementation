import os
import torch
import torch.optim as optim
from tqdm import tqdm

from config import get_config
from data import load_data
from ops import HyperSphereLoss
from model import Generator, Discriminator, weights_init
from torchvision.utils import save_image

if __name__ == '__main__':
    opt = get_config()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load Training Data
    train_data, train_loader = load_data(opt)

    # Define Generator and Discriminator
    G = Generator(opt).to(device)
    G.apply(weights_init)

    D = Discriminator(opt).to(device)
    D.apply(weights_init)

    # Define Optimizer
    G_optim = optim.Adam(G.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
    D_optim = optim.Adam(D.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))

    # Loss Function
    criterion = HyperSphereLoss()

    # Load CheckPoint
    if os.path.exists(opt['checkpoint']):
        state = torch.load(opt['checkpoint'])
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        G_optim.load_state_dict(state['G_optim'])
        D_optim.load_state_dict(state['D_optim'])
    else:
        state = {}
        state['global_step'] = 0

    # Train
    running_G_loss = 0
    running_D_loss = 0
    for epoch in range(opt['epoch']):
        fixed_vec = torch.randn(25, opt['nz']).to(device)
        for idx, (X, _) in tqdm(enumerate(train_loader), maxinterval=len(train_loader)):
            X = X.to(device)
            bs = X.size()[0]

            # Train G
            G.train()
            G_optim.zero_grad()

            Z = torch.randn(bs, opt['nz']).to(device)
            gen_X = G(Z)
            pred_G_Z = D(gen_X)
            G_loss = criterion(pred_G_Z)
            G_loss.backward()
            running_G_loss += G_loss.item()

            G_optim.step()

            # train D
            D.train()
            D_optim.zero_grad()

            Z = torch.randn(bs, opt['nz']).to(device)
            gen_X = G(Z)
            pred_G_Z = D(gen_X.detach())

            pred_X = D(X)

            D_loss = criterion(pred_X) - criterion(pred_G_Z)
            D_loss.backward()
            running_D_loss += D_loss.item()

            D_optim.step()

            state['global_step'] += 1

            if state['global_step'] % opt['save_checkpoint'] == 0:
                state['G'] = G.state_dict()
                state['D'] = D.state_dict()
                state['G_optim'] = G_optim.state_dict()
                state['D_optim'] = D_optim.state_dict()
                torch.save(state, opt['checkpoint'])

            if state['global_step'] % opt['print_log'] == 0:
                tqdm.write('global_step = {}, G_loss = {}, D_loss = {}'.format(
                    state['global_step'], running_G_loss / opt['print_log'], running_D_loss / opt['print_log']
                ))
                running_G_loss = 0
                running_D_loss = 0

                with torch.no_grad():
                    fixed_img = G(fixed_vec)
                    save_image(fixed_img, '{}/{}.png'.format(opt['experiments_dir'], 'fixed'),
                               nrow=5, normalize=True)
                    save_image(fixed_img, '{}/{}_{:5d}.png'.format(opt['experiments_dir'], 'fixed', state['global_step']),
                               nrow=5, normalize=True)

                    random_img = G(torch.randn(25, opt['nz']).to(device))
                    save_image(random_img, '{}/{}.png'.format(opt['experiments_dir'], 'random'),
                               nrow=5, normalize=True)
                    save_image(fixed_img, '{}/{}_{:5d}.png'.format(opt['experiments_dir'], 'random', state['global_step']),
                               nrow=5, normalize=True)