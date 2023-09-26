import torch


def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    points, calibrations = samples, calibs
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

'''

show_pts = orthogonal(sample_tensor, calib_tensor[0], transforms=None).cpu().numpy()
show_label = label_tensor.cpu().numpy()


xy = show_pts[:, :2, :]
x = show_pts[:, 0, :]
y = show_pts[:, 1, :]
z = show_pts[:, 2, :]

print("x max:%.4f,min:%.4f,mean:%.4f"%(x.max(),x.min(),x.mean()))
print("y max:%.4f,min:%.4f,mean:%.4f"%(y.max(),y.min(),y.mean()))
print("z max:%.4f,min:%.4f,mean:%.4f"%(z.max(),z.min(),z.mean()))


import trimesh
import pyrender
import numpy as np
# show surface

show_data = np.concatenate((show_pts[0],show_label[0]),0).transpose(1,0)
distance = 0.5
data_show = show_data[np.where(show_data[:,3]<distance)]
points_show,sdf_show =  data_show[:,:3], data_show[:,3]
colors = np.zeros(points_show.shape)
colors[sdf_show < 0.5, 2] = 1
colors[sdf_show > 0.5, 0] = 1
cloud = pyrender.Mesh.from_points(points_show, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


'''

def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
