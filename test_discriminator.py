import argparse
from subnetwork import Discriminator
import torch
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--z_dim', type=int, default=128)

args = parser.parse_args()


device = 'cuda:{}'.format(args.gpu_number) if torch.cuda.is_available() else 'cpu'

# build discriminator class
discriminator = Discriminator(args=args, in_channels=3)


real_image = np.random.randn(10, 3, args.image_size, args.image_size)
real_image_tensor = torch.from_numpy(real_image).to(device).float()
print('input image tensor shape:', real_image_tensor.shape)
discriminator(real_image_tensor)


