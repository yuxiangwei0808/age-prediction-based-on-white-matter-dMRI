import numpy as np
import random
import torch


def rand_points(size, lam, dim):
    L = size[dim]
    cut_rat = np.sqrt(1. - lam)
    number_of_points = int(L * cut_rat)

    old_points = random.sample(range(0, L), number_of_points)
    new_points = random.sample(range(0, L), number_of_points)

    return old_points, new_points
        

def cutmix_discrete(data, dim='2d', axis=None):
    batch_size = data.shape[0]
    rand_index = torch.randperm(batch_size).cuda()
    lam = np.random.beta(1, 1)

    if dim =='1d':
        old_points, new_points = rand_points(data.shape, lam, axis)
        if axis == 1:
            data[:, new_points, :] = data[:, old_points, :]   
        elif axis == 2:
            data[:, :, new_points] = data[:, :, old_points]
    elif dim == '2d':
        # old_points_2, new_points_2 = rand_points(data.shape, lam, dim=2)
        # for i in range(len(new_points_2)):
        #     old_points_1, new_points_1 = rand_points(data.shape, lam, dim=1)
        #     feature_idx_old = old_points_2[i]
        #     feature_idx_new = new_points_2[i]
        #     data[:, new_points_1, feature_idx_new] = data[:, old_points_1, feature_idx_old]
        old_points_1, new_points_1 = rand_points(data.shape, lam, dim=1)
        for i in range(len(new_points_1)):
            old_points_2, new_points_2 = rand_points(data.shape, lam, dim=2)
            feature_idx_old = old_points_1[i]
            feature_idx_new = new_points_1[i]
            temp = data[rand_index, :, :]
            data[:, feature_idx_new, new_points_2] = temp[:, feature_idx_old, old_points_2]
    return data



    
