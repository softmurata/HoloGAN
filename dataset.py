import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, args, data_num):
        super().__init__()
        self.data_num = data_num
        self.image_size = args.image_size
        self.real_images = []

    def create_dummy_data(self):
        self.real_images = np.random.randn(self.data_num, 3, self.image_size, self.image_size)  # 0 - 1

    def __len__(self):

        return len(self.real_images)

    def __getitem__(self, item):

        real_image = self.real_images[item]
        # transform teal image to pytorch tensor
        real_image = torch.from_numpy(real_image).type(torch.float32)

        return real_image

