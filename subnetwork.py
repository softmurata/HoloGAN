import numpy as np
import torch
import torch.nn as nn

from rotation_utils import transform3d
from utils import transform_voxel_to_match_image

class Generator(nn.Module):

    def __init__(self, args, c_dim=3, gf_dim=64, voxel_size=128):

        super().__init__()
        self.image_size = args.image_size
        self.batch_size = args.batch_size

        self.gf_dim = gf_dim
        self.voxel_size = voxel_size
        self.c_dim = c_dim  # out_channels

        self.in_channels = self.gf_dim * 8  # 512

        self.z_mapping_mlp = ZMappingMLP()
        self.adain = AdaIn()

        # in_channels => const_tensor.shape[1]
        self.deconv3d_1 = nn.ConvTranspose3d(in_channels=self.in_channels, out_channels=self.gf_dim * 2, kernel_size=2, stride=2, padding=0)
        self.deconv3d_2 = nn.ConvTranspose3d(in_channels=self.gf_dim * 2, out_channels=self.gf_dim * 1, kernel_size=2, stride=2, padding=0)

        # convolution 2d
        # for projection
        self.deconv2d_1 = nn.ConvTranspose2d(in_channels=self.gf_dim * self.voxel_size, out_channels=self.gf_dim * 16, kernel_size=1, stride=1, padding=0)

        self.deconv2d_2 = nn.ConvTranspose2d(in_channels=self.gf_dim * 16, out_channels=self.gf_dim * 4, kernel_size=3, stride=1, padding=1)
        self.deconv2d_3 = nn.ConvTranspose2d(in_channels=self.gf_dim * 4, out_channels=self.gf_dim, kernel_size=3, stride=1, padding=1)

        self.deconv2d_4 = nn.ConvTranspose2d(in_channels=self.gf_dim, out_channels=self.c_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, z, view_in):
        """
        input => x (batch_size, channel(512), height(4), width(4), depth(4)), z => (cont_dim vector)
        conv3dblock * 2 => conv3d, adain(args: MLP(z)), leakyrelu

        3d transformer(args: camera pose)

        conv3d block * 2

        projection unit

        conv2d block => conv2d, adain(args: MLP(z)), leakyrelu


        """
        # Input block
        s0, b0 = self.z_mapping_mlp(z, self.in_channels)  # s => scale, b => bias, adain = scale * ((x - mean) / std) + bias
        h0 = self.adain(x, s0, b0)
        h0 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(h0)

        # print('first hidden shape:', h0.shape)  # (batch_size, 512, 4, 4, 4)

        # Block1
        h1 = self.deconv3d_1(h0)
        s1, b1 = self.z_mapping_mlp(z, self.gf_dim * 2)
        h1 = self.adain(h1, s1, b1)
        h1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(h1)  # (batch_size, 128, 8, 8, 8)

        # Block2
        h2 = self.deconv3d_2(h1)
        s2, b2 = self.z_mapping_mlp(z, self.gf_dim * 1)
        h2 = self.adain(h2, s2, b2)
        h2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(h2)  # (batch_size, 64, 16, 16, 16)

        # 3d transformer(args view_in(camera pose))
        # view_in => 6 dimension tensor
        h2_rotated = transform3d(voxel_array=h2, view_params=view_in)  # output => voxel format(b, 1, 128, 128, 128)

        # projection unit(with depth dimension collapsion)
        h2_rotated = transform_voxel_to_match_image(h2_rotated)

        batch_size, channels, depth, height, width = h2_rotated.shape

        # collapsing depth dimension  (batch_size, channels, depth, height, width)
        h2_2d = h2_rotated.reshape(batch_size, channels * depth, height, width)  # (batch_size, channels * depth, height, width)

        # 1 * 1 convolution(occulusion learning for collapsing dimension)
        h3 = self.deconv2d_1(h2_2d)
        h3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(h3)

        # 2d convolution block
        h4 = self.deconv2d_2(h3)
        s4, b4 = self.z_mapping_mlp(z, self.gf_dim * 4)  # ToDo: same function is ok?
        h4 = self.adain(h4, s4, b4)
        h4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(h4)

        h5 = self.deconv2d_3(h4)
        s5, b5 = self.z_mapping_mlp(z, self.gf_dim)
        h5 = self.adain(h5, s5, b5)
        h5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(h5)
        h6 = self.deconv2d_4(h5)
        output = nn.Tanh()(h6)


        return output



class Discriminator(nn.Module):

    def __init__(self, args, in_channels=3, df_dim=64):

        super().__init__()

        self.df_dim = df_dim

        # input size => (batch_size, 3, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.df_dim, kernel_size=3, stride=2, padding=1)  # (batch_size, 64, 64, 64)
        self.conv2 = nn.Conv2d(in_channels=self.df_dim, out_channels=self.df_dim * 2, kernel_size=3, stride=2, padding=1)  # (batch_size, 128, 32, 32)
        self.instance_norm2 = nn.InstanceNorm2d(num_features=self.df_dim * 2)
        self.conv3 = nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 4, kernel_size=3, stride=2, padding=1)  # (batch_size, 256, 16, 16)
        self.instance_norm3 = nn.InstanceNorm2d(num_features=self.df_dim * 4)
        self.conv4 = nn.Conv2d(in_channels=self.df_dim * 4, out_channels=self.df_dim * 8, kernel_size=3, stride=2, padding=1)  # (batch_size, 512, 8, 8)
        self.instance_norm4 = nn.InstanceNorm2d(num_features=self.df_dim * 8)

        # additional layer
        flatten_size = int(self.df_dim * 8 * np.floor(args.image_size // 2 ** 4) ** 2)
        self.linear1 = nn.Linear(in_features=flatten_size, out_features=1)

        self.linear2 = nn.Linear(in_features=flatten_size, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=args.z_dim)


    def forward(self, x):
        """

        h0 => leakyrelu(conv(x))
        h1 => leakyrelu(instance_norm(conv_specnorm(h0)))
        h2 => leakyrelu(instance_norm(conv_specnorm(h1)))
        h3 => leakyrelu(instance_norm(conv_specnorm(h2)))
        h3 => flatten()
        h4 => linear(h3)

        # additional layer
        encoder => leakyrelu(linear(flatten(h3), 128))
        output => linear(encoder, cont_dim)

        """

        # recognition network for latent variables has an additional layer
        # linear layer
        # output => cont_dim vector
        batch_size = x.shape[0]
        h0 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.conv1(x))
        h1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm2(self.conv2(h0)))
        h2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm3(self.conv3(h1)))
        h3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm4(self.conv4(h2)))
        flatten_h3 = h3.view(batch_size, -1)

        # additional layer
        h4 = self.linear1(flatten_h3)

        encoder = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.linear2(flatten_h3))
        latent_output = nn.Tanh()(self.linear3(encoder))

        return h4, nn.Sigmoid()(h4), latent_output, (nn.Sigmoid()(h1), nn.Sigmoid()(h2), nn.Sigmoid()(h3), nn.Sigmoid()(h4))



class ZMappingMLP(nn.Module):

    def __init__(self):

        super().__init__()


    def forward(self, x, out_channels):
        # scale = sigma(y), bias = mu(y)
        # z = (batch_size, z_dim)
        x = nn.Linear(in_features=x.shape[-1], out_features=out_channels * 2)(x)
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(x)
        scale = x[:, :out_channels]
        bias = x[:, out_channels:]

        return scale, bias


class AdaIn(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, s, b):
        
        size = x.size()
        batch_size, in_channels = s.shape

        if len(x.shape) == 5:
            mean = torch.mean(x.view(batch_size, in_channels, -1), 2).view(batch_size, in_channels, 1, 1, 1)
            std = torch.std(x.view(batch_size, in_channels, -1), 2).view(batch_size, in_channels, 1, 1, 1)
            s = s.view(batch_size, in_channels, 1, 1, 1).expand(size)
            b = b.view(batch_size, in_channels, 1, 1, 1).expand(size)
        else:
            mean = torch.mean(x.view(batch_size, in_channels, -1), 2).view(batch_size, in_channels, 1, 1)
            std = torch.std(x.view(batch_size, in_channels, -1), 2).view(batch_size, in_channels, 1, 1)
            s = s.view(batch_size, in_channels, 1, 1).expand(size)
            b = b.view(batch_size, in_channels, 1, 1).expand(size)

        # match dimension (batch_size, channles, depth, height, width)
        mean = mean.expand(size)
        std = std.expand(size)
        

        # print('scale shape:', s.shape, 'bias shape:', b.shape, 'mean:', mean.shape, 'std:', std.shape)

        output = s * (x - mean) / std + b
        return output
