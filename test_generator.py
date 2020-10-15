import argparse
from subnetwork import Generator
from rotation_utils import generate_random_rotation_translation
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--z_dim', type=int, default=128)
args = parser.parse_args()

device = "cuda:{}".format(args.gpu_number) if torch.cuda.is_available() else 'cpu'

gf_dim = 64
generator = Generator(args)

# inputs
input_size = 4
input_tensor = torch.empty(args.batch_size, gf_dim * 8, input_size, input_size, input_size)  # if image size = 128, input_size = 4
const_input = nn.init.normal_(input_tensor, mean=0.0, std=0.2)  # (b, 512, 4, 4, 4) => random normal tensor

z_input_tensor = torch.empty(args.batch_size, args.z_dim)
z = nn.init.normal_(z_input_tensor, mean=0.0, std=1.0)

# input convert
const_input = const_input.to(device)
z = z.to(device)

view_in = generate_random_rotation_translation(args.batch_size)
view_in = torch.from_numpy(view_in)

print('input tensor shape:', const_input.shape, 'z tensor shape:', z.shape, 'view_in shape:', view_in.shape)

output = generator(const_input, z, view_in)
