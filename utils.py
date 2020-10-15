import torch

def transform_voxel_to_match_image(tensor):
    # tensor => (batch_size, channels, height, width, depth)
    # => (batch_size, channels, depth, height, width[::-1])

    tensor = tensor.permute(0, 1, 4, 2, 3)

    reverse = torch.arange(tensor.shape[-1] - 1, -1, -1)

    return tensor[:, :, :, :, reverse]