import argparse
import torch
from HoloGAN import HoloGAN

# build hologan
# train(), predict()

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--resume_g_path', type=str, default='', help='ex) ./results/models/exp1/checkpoint_generator_050000.pth.tar')
parser.add_argument('--model_dir', type=str, default='./results/models/')
parser.add_argument('--exp_name', type=str, default='exp1')
parser.add_argument('--dataset_dir', type=str, default='./dataset/')
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--style_type', type=str, default='style', help=" 'style' or '' ")

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr_g', type=float, default=0.0001)
parser.add_argument('--lr_d', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)


args = parser.parse_args()

device = "cuda:{}".format(args.gpu_number) if torch.cuda.is_available() else 'cpu'

# build hologan class
phase = 'train'
hologan = HoloGAN(args, device)

if phase == 'train':
    hologan.train()
else:
    hologan.predict()







