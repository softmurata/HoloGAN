import torch
import numpy as np

def transform3d(voxel_array, view_params, size=64, new_size=128):
    # view_params => transformation matrix (3D => 2D)
    # rotation around grid centroid => return transformation matrix, batch_scale
    # rotation resampling => return target, grid_transform


    transformation_matrix, batch_scale = rotation_around_grid_centroid(view_params[:, :3])
    target, grid_transform = rotation_resampling(voxel_array, transformation_matrix, params=view_params, scale_matrix=batch_scale, size=size, new_size=new_size)
    

    return target


def rotation_around_grid_centroid(view_params):
    # ToDo: maybe add device dependency
    # view_params => torch tensor
    batch_size = view_params.shape[0]

    azimuth_angle = view_params[:, 0]  # (batch_size, 1)
    elevation_angle = view_params[:, 1]  # (batch_size, 1)

    azimuth_angle = azimuth_angle.view(batch_size, 1, 1)
    elevation_angle = elevation_angle.view(batch_size, 1, 1)

    azimuth_zeros = torch.zeros_like(azimuth_angle)
    elevation_zeros = torch.zeros_like(elevation_angle)

    azimuth_ones = torch.ones_like(azimuth_angle)
    elevation_ones = torch.ones_like(elevation_angle)

    # Y axis rotation matrix(4 * 4)
    rotY = torch.cat([torch.cat([torch.cos(azimuth_angle), azimuth_zeros, -torch.sin(azimuth_angle), azimuth_angle], axis=2),
                        torch.cat([azimuth_zeros, azimuth_ones, azimuth_zeros, azimuth_zeros], axis=2),
                        torch.cat([torch.sin(azimuth_angle), azimuth_zeros, torch.cos(azimuth_angle), azimuth_zeros], axis=2),
                        torch.cat([azimuth_zeros, azimuth_zeros, azimuth_zeros, azimuth_ones], axis=2)], axis=1)

    # Z axis rotation matrix
    rotZ = torch.cat([torch.cat([torch.cos(elevation_angle), torch.sin(elevation_angle), elevation_zeros, elevation_zeros], axis=2),
                        torch.cat([-torch.sin(elevation_angle), torch.cos(elevation_angle), elevation_zeros, elevation_zeros], axis=2),
                        torch.cat([elevation_zeros, elevation_zeros, elevation_ones, elevation_zeros], axis=2),
                        torch.cat([elevation_zeros, elevation_zeros, elevation_zeros, elevation_ones], axis=2)], axis=1)


    transformation_matrix = torch.matmul(rotY, rotZ)  # (batch_size, 4, 4)


    # sclae matrix
    scale = view_params[:, 2]
    scale = scale.view(batch_size, 1, 1)

    scale_zeros = torch.zeros_like(scale)
    scale_ones = torch.ones_like(scale)

    # (batch_size, 4, 4)
    batch_scale = torch.cat([torch.cat([scale, scale_zeros, scale_zeros, scale_zeros], axis=2),
                                torch.cat([scale_zeros, scale, scale_zeros, scale_zeros], axis=2),
                                torch.cat([scale_zeros, scale_zeros, scale, scale_zeros], axis=2),
                                torch.cat([scale_zeros, scale_zeros, scale_zeros, scale_ones], axis=2)], axis=1)


    return transformation_matrix, batch_scale


def rotation_resampling(voxel_array, transformation_matrix, params, scale_matrix, size, new_size):
    # voxel_array => (batch_size, channel, depth, height, width)  # pytorch format
    batch_size = voxel_array.shape[0]
    n_channels = voxel_array.shape[1]
    T = torch.tensor([[1, 0, 0, -size * 0.5],
                        [0, 1, 0, -size * 0.5],
                        [0, 0, 1, -size * 0.5],
                        [0, 0, 0, 1]])
    T = T.view(1, 4, 4)
    T = T.repeat(batch_size, 1, 1)

    T_new_inv = torch.tensor([[1, 0, 0, new_size * 0.5],
                        [0, 1, 0, new_size * 0.5],
                        [0, 0, 1, new_size * 0.5],
                        [0, 0, 0, 1]])
    T_new_inv = T_new_inv.view(1, 4, 4)
    T_new_inv = T_new_inv.repeat(batch_size, 1, 1)

    # add the actual shift x and y dimension
    x_shift = params[:, 3].view(batch_size, 1, 1)
    y_shift = params[:, 4].view(batch_size, 1, 1)
    z_shift = params[:, 5].view(batch_size, 1, 1)

    # translation vector
    ones = torch.ones_like(x_shift)
    zeros = torch.zeros_like(x_shift)
    T_translate = torch.cat([torch.cat([ones, zeros, zeros, x_shift], axis=2),
                                torch.cat([zeros, ones, zeros, y_shift], axis=2),
                                torch.cat([zeros, zeros, ones, z_shift], axis=2),
                                torch.cat([zeros, zeros, zeros, ones], axis=2)], axis=1)

    

    total_M = torch.matmul(torch.matmul(torch.matmul(torch.matmul(T_new_inv.float(), T_translate.float()), scale_matrix.float()), transformation_matrix.float()), T.float())

    total_M = torch.inverse(total_M)
    total_M = total_M[:, :, 0:3]  # (batch_size, 3, 4)
    grid = create_voxel_grid(height=new_size, width=new_size, depth=new_size)  # (3, 128 * 128 * 128)
    grid_reshape = grid.view(1, int(grid.shape[0]), int(grid.shape[1]))  # (1, 3, 128 * 128 * 128)
    grid = grid_reshape.repeat(batch_size, 1, 1)  # (batch_size, 3, 128 * 128 * 128)
    grid_transform = torch.matmul(total_M.float(), grid.float())  # (batch_size, 4, 128 * 128 * 128)
    x_s_flat = grid_transform[:, 0, :].reshape(-1)  # (batch_size * 128 * 128 * 128)
    y_s_flat = grid_transform[:, 1, :].reshape(-1)  # (batch_size * 128 * 128 * 128)
    z_s_flat = grid_transform[:, 2, :].reshape(-1)  # (batch_size * 128 * 128 * 128)
    out_size = [batch_size, n_channels, new_size, new_size, new_size]
    input_transform = interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat, out_size)
    target = input_transform.view(batch_size, n_channels, new_size, new_size, new_size)
    


    return target, grid_transform


def create_voxel_grid(height, width, depth):
    # print('create voxel grid part start')
    # maybe x, y swapping
    x_tensor = torch.arange(0, width)
    y_tensor = torch.arange(0, height)
    z_tensor = torch.arange(0, depth)
    x_t, y_t, z_t = torch.meshgrid(x_tensor, y_tensor, z_tensor)

    # x_t = (128, 128, 128)

    # flatten
    input_size = x_t.shape[0]
    x_t_flat = x_t.reshape(1, -1) # (1, 128 * 128 * 128)
    y_t_flat = y_t.reshape(1, -1)  # (1, 128 * 128 * 128)
    z_t_flat = z_t.reshape(1, -1)  # (1, 128 * 128 * 128)


    grid = torch.cat([x_t_flat, y_t_flat, z_t_flat], axis=0)  # (3, 128 * 128 * 128)
    return grid

def interpolate(voxel_array, x, y, z, out_size):
    # Maybe this function has many bugs
    # out_size = (batch_size, channels, height, width, depth)
    print('interpolation part start')
    batch_size, n_channels, height, width, depth = voxel_array.shape
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    z = z.type(torch.float32)


    out_channels, out_height, out_width, out_depth = out_size[1:]

    
    max_y = height - 1
    max_x = width - 1
    max_z = depth - 1

    

    # do sampling
    x0 = torch.floor(x).type(torch.int32)
    x1 = x0 + 1

    y0 = torch.floor(y).type(torch.int32)
    y1 = y0 + 1

    z0 = torch.floor(z).type(torch.int32)
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)

    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)

    z0 = torch.clamp(z0, 0, max_z)
    z1 = torch.clamp(z1, 0, max_z)

    
    batch_range = torch.arange(batch_size) * width * height * depth
    # print('batch range shape:', batch_range.shape)
    n_repeats = out_height * out_width * out_depth
    rep = torch.ones([1, n_repeats], dtype=torch.int32)  # (1, 128 * 128 * 128)
    base = torch.matmul(batch_range.reshape(-1, 1).float(), rep.float())
    base = base.reshape(-1)
    # base shape => (100, 128 * 128 * 128)
    
    # print('find z element')
    # find the z element
    # print(base.shape, z0.shape)
    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height

    # find the y element based on z
    base_z0_y0 = base_z0 + y0 * width
    base_z1_y1 = base_z1 + y1 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width

    # print('find index element')
    # find the x element based on y, z for z=0
    idx_a = base_z0_y0 + x0
    idx_b = base_z0_y1 + x0
    idx_c = base_z0_y0 + x1
    idx_d = base_z0_y1 + x1

    # find the x element based on y, z for z=1
    idx_e = base_z1_y0 + x0
    idx_f = base_z1_y1 + x0
    idx_g = base_z1_y0 + x1
    idx_h = base_z1_y1 + x1

    idx_a = idx_a.type(torch.int64)
    idx_b = idx_b.type(torch.int64)
    idx_c = idx_c.type(torch.int64)
    idx_d = idx_d.type(torch.int64)

    idx_e = idx_e.type(torch.int64)
    idx_f = idx_f.type(torch.int64)
    idx_g = idx_g.type(torch.int64)
    idx_h = idx_h.type(torch.int64)



    # print('assign voxel flatten')
    voxel_flat = voxel_array.permute(0, 2, 3, 4, 1).reshape(-1, n_channels)  # (batch_size * height * width * depth, channels)
    Ia = voxel_flat[idx_a]  # (batch_size * height * width * depth, 1)
    Ib = voxel_flat[idx_b]
    Ic = voxel_flat[idx_c]
    Id = voxel_flat[idx_d]
    Ie = voxel_flat[idx_e]
    If = voxel_flat[idx_f]
    Ig = voxel_flat[idx_g]
    Ih = voxel_flat[idx_h]


    # convert float
    x0_f = x0.type(torch.float32)
    x1_f = x1.type(torch.float32)
    y0_f = y0.type(torch.float32)
    y1_f = y1.type(torch.float32)
    z0_f = z0.type(torch.float32)
    z1_f = z1.type(torch.float32)

    # print('slice weight coefficient')
    # first slice at z=0
    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y) * (z1_f - z)), 1)  # tf.expand_dims((), 1)
    wb = torch.unsqueeze(((x1_f - x) * (y - y0_f) * (z1_f - z)), 1)  # tf.expand_dims((), 1)
    wc = torch.unsqueeze(((x - x0_f) * (y1_f - y) * (z1_f - z)), 1)
    wd = torch.unsqueeze(((x - x0_f) * (y - y0_f) * (z1_f - z)), 1)

    # first slice at z=1
    we = torch.unsqueeze(((x1_f - x) * (y1_f - y) * (z - z0_f)), 1)
    wf = torch.unsqueeze(((x1_f - x) * (y - y0_f) * (z - z0_f)), 1)
    wg = torch.unsqueeze(((x - x0_f) * (y1_f - y) * (z - z0_f)), 1)
    wh = torch.unsqueeze(((x - x0_f) * (y - y0_f) * (z - z0_f)), 1)

    output = Ia * wa + Ib * wb + Ic * wc + Id * wd + Ie * we + If * wf + Ig * wg

    return output


# create argument called view_in
def generate_random_rotation_translation(batch_size, elevation_low=10, elevation_high=170,
                                                    azimuth_low=0, azimuth_high=359, 
                                                    transX_low=-3, transX_high=3,
                                                    transY_low=-3, transY_high=3,
                                                    transZ_low=-3, transZ_high=3,
                                                    scale_low=1.0, scale_high=1.0,
                                                    with_translation=False, with_scale=False):

    params = np.zeros((batch_size, 6))
    column = np.arange(0, batch_size)
    azimuth = np.random.randint(elevation_low, elevation_high, (batch_size)).astype(np.float) * np.pi / 180.0
    elevation = (90.0 - np.random.randint(elevation_low, elevation_high, (batch_size)).astype(np.float)) * np.pi / 180.0

    params[column, 0] = azimuth
    params[column, 1] = elevation

    if with_scale:
        params[column, 2] = float(np.random.uniform(scale_low, scale_high))
    else:
        params[column, 2] = 1.0

    if with_translation:
        shift_x = transX_low + np.random.random(batch_size) * (transX_high - transX_low)
        shift_y = transY_low + np.random.random(batch_size) * (transY_high - transY_low)
        shift_z = transZ_low + np.random.random(batch_size) * (transZ_high - transZ_low)

        params[column, 3] = shift_x
        params[column, 4] = shift_y
        params[column, 5] = shift_z

    return params
    





