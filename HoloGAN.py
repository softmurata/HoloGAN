import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader

from torch.optim import Adam

from subnetwork import Generator, Discriminator
from dataset import ImageDataset

from rotation_utils import generate_random_rotation_translation



class HoloGAN(nn.Module):

    def __init__(self, args, device, gf_dim=64):
        super().__init__()

        self.epoch = args.epoch
        self.device = device
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.save_freq = args.save_freq
        self.style_type = args.style_type

        self.z_dim = args.z_dim

        self.model_dir = './results/models/{}/'.format(args.exp_name)
        os.makedirs(self.model_dir, exist_ok=True)

        self.weights_dir = self.model_dir + 'weights/'
        os.makedirs(self.weights_dir, exist_ok=True)

        self.gene_image_train_dir = self.model_dir + 'gene_images/train/'
        self.gene_image_test_dir = self.model_dir + 'gene_images/test/'
        os.makedirs(self.gene_image_train_dir, exist_ok=True)
        os.makedirs(self.gene_image_test_dir, exist_ok=True)

        # build generator
        self.generator = Generator(args)
        # build discriminator
        self.discriminator = Discriminator(args)

        if args.resume_g_path:
            generator_weight_path = args.resume_g_path
            root_dir = './'
            start_epoch = generator_weight_path.split('/')[-1].split('.')[0].split('_')[-1]
            
            for g in generator_weight_path.split('/')[1:-1]:
                root_dir += '{}/'.format(g)
            discriminator_weight_path = root_dir + 'checkpoint_discriminator_{}.pth.tar'.format(start_epoch)

            self.generator.load_state_dict(torch.load(generator_weight_path))
            self.discriminator.load_state_dict(torch.load(discriminator_weight_path))

        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

        # constant input tensor
        self.gf_dim = gf_dim
        # const input
        input_size = self.image_size // (2 ** 4)
        input_tensor = torch.empty(self.batch_size, gf_dim * 8, input_size, input_size, input_size)  # if image size = 128, input_size = 4
        self.const_input = nn.init.normal_(input_tensor, mean=0.0, std=0.2)  # (b, 512, 4, 4, 4) => random normal tensor

        self.g_optimizer = Adam(self.generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

        # transform compositor


        # create dataloader
        # data_num = len(os.listdir(dataset_dir + 'images/'))  #'0.png, 1.png, ....
        data_num = 10
        dataset = ImageDataset(args, data_num)
        dataset.create_dummy_data()
        self.train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def sample_z(self, z_dim, type='uniform'):
        if type == 'uniform':
            return np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, z_dim))
        else:
            return np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, z_dim))


    def train(self):
        # input => (batch_size, output_height, output_width, ch) (real_images)
        # camera pose => (batch_size, 6)
        # z => (batch_size, z_dim)

        bce_loss = nn.BCELoss().to(self.device)  # reduce => mean

        self.generator.train()
        self.discriminator.train()
        
        for e in range(self.epoch):

            g_losses = []
            d_losses = []

            print('training start')

            self.generator.train()

            for n, data in enumerate(self.train_dataloader):
                # data converts into cuda format
                # data => real_image
                real_image = data
                # z => create sampler

                # update D
                print('back propagation for D')
                self.d_optimizer.zero_grad()
                # create z vector
                z = torch.from_numpy(self.sample_z(self.z_dim)).type(torch.float32)
                z_clone = torch.clone(z)  # for discriminator update

                # create view_in
                view_in = generate_random_rotation_translation(self.batch_size)
                view_in = torch.from_numpy(view_in).type(torch.float32)
                view_in_clone = torch.clone(view_in)

                # inference and get variable for discriminator
                fake_g = self.generator(self.const_input, z, view_in)  # G(x)
                # real inference
                d_r, d_logit_r, _, (d_h1_logit_r, d_h2_logit_r, d_h3_logit_r, d_h4_logit_r) = self.discriminator(real_image)
                # fake inference
                d_f, d_logit_f, q_c_given, (d_h1_logit_f, d_h2_logit_f, d_h3_logit_f, d_h4_logit_f) = self.discriminator(fake_g)
                real_loss = bce_loss(d_logit_r, torch.ones_like(d_r))
                fake_loss = bce_loss(d_logit_f, torch.zeros_like(d_f))

                # style loss => this loss corresponds to each layer
                style_loss = 0
                if self.style_type != '':
                    # if discriminator's layer is deeper, loss value has to increase
                    d_h1_loss = bce_loss(d_h1_logit_r, torch.ones_like(d_h1_logit_r)) + bce_loss(d_h1_logit_f, torch.zeros_like(d_h1_logit_f))
                    d_h2_loss = bce_loss(d_h2_logit_r, torch.ones_like(d_h2_logit_r)) + bce_loss(d_h2_logit_f, torch.zeros_like(d_h2_logit_f))
                    d_h3_loss = bce_loss(d_h3_logit_r, torch.ones_like(d_h3_logit_r)) + bce_loss(d_h3_logit_f, torch.zeros_like(d_h3_logit_f))
                    d_h4_loss = bce_loss(d_h4_logit_r, torch.ones_like(d_h4_logit_r)) + bce_loss(d_h4_logit_f, torch.zeros_like(d_h4_logit_f))
                    style_loss = d_h1_loss + d_h2_loss + d_h3_loss + d_h4_loss

                identity_loss = torch.mean(torch.square(z - q_c_given))  # z is input latent vector(cont_dim vector)

                d_loss = real_loss + fake_loss + identity_loss + style_loss

                
                # back propagation
                d_loss.backward()
                self.d_optimizer.step()

                # update G
                print('back propagation for G')
                self.g_optimizer.zero_grad()
                # inference and get variable for generator
                fake_g = self.generator(self.const_input, z_clone, view_in_clone)
                d, d_logit, q_c_given, _ = self.discriminator(fake_g)
                g_loss = bce_loss(d_logit, torch.ones_like(d))

                # identity loss
                identity_loss = torch.mean(torch.square(z - q_c_given))  # z is input latent vector(cont_dim vector)

                g_loss = g_loss + identity_loss  # ToDo: maybe modified?


                # back propagation
                g_loss.backward()
                self.g_optimizer.step()

                print('batch: {}  g_loss:{:.5f}  d_loss:{:.5f}'.format(n, g_loss.item(), d_loss.item()))

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            mean_g_loss = np.mean(g_losses)
            mean_d_loss = np.mean(d_losses)
            # print results
            print('epoch: {}  generator loss: {:.5f}  discriminator loss: {:.5f} '.format(e, mean_g_loss, mean_d_loss))
            
            if e % self.save_freq == 0:
                # create weight path
                generator_weight_path = self.weights_dir + 'checkpoint_generator_%05d.pth.tar' % e
                discriminator_weight_path = self.weights_dir + 'checkpoint_discriminator_%05d.pth.tar' % e

                # save generator weight
                self.save(generator_weight_path, self.generator, e)

                # save discriminator weight
                self.save(discriminator_weight_path, self.discriminator, e)

                # inference
                self.generator.eval()
                z = torch.from_numpy(self.sample_z(self.z_dim)).type(torch.float32)
                view_in = generate_random_rotation_translation(self.batch_size)
                view_in = torch.from_numpy(view_in).type(torch.float32)
                gene_images = self.generator(self.const_input, z, view_in)
                gene_images = (gene_images + 1.0) / 2.0
                gene_images = gene_images.permute(0, 2, 3, 1).detach().cpu().numpy()  # (b, c, h, w) => (b, h, w, c)

                image_path = self.gene_image_train_dir + "0.png"

                plt.imsave(image_path, gene_images)


    def predict(self):
        # ToDo => generator(input), input need to be converted into cuda format(input = input.to(device))
        self.generator.eval()
        # inference
        z = self.sample_z(self.z_dim)
        view_in = generate_random_rotation_translation(self.batch_size)
        gene_images = self.generator(self.const_input, z, view_in)
        gene_images = (gene_images + 1.0) / 2.0

        image_path = self.gene_image_test_dir + "0.png"

        plt.imsave(image_path, gene_images)

    def save(self, weight_path, model, epoch):

        torch.save({
            "epoch": epoch,
            "model": model.state_dict()
        }, weight_path)

