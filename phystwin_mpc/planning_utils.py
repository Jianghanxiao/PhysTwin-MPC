import torch


def batch_chamfer_dist(xyz, xyz_gt):
    xyz_gt = xyz_gt[None]
    dist1 = torch.sqrt(torch.sum((xyz[:, :, None] - xyz_gt[:, None]) ** 2, dim=3))
    dist2 = torch.sqrt(torch.sum((xyz_gt[:, None] - xyz[:, :, None]) ** 2, dim=3))
    chamfer = torch.mean(torch.min(dist1, dim=1).values, dim=1) + torch.mean(torch.min(dist2, dim=1).values, dim=1)
    return chamfer
