import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from models.minkunet import MinkUNet
from dataloader.SemanticKITTIDataLoader import SemanticKITTIDataLoader
import torch
import MinkowskiEngine as ME

sparse_models = {
    'MinkUNet': MinkUNet,
}

data_loaders = {
    'SemanticKITTI': SemanticKITTIDataLoader,
}

color_map = {
    0: [0, 0, 0],
    1: [245, 150, 100],
    2: [245, 230, 100],
    3: [150, 60, 30],
    4: [180, 30, 80],
    5: [255, 0, 0],
    6: [30, 30, 255],
    7: [200, 40, 255],
    8: [90, 30, 150],
    9: [255, 0, 255],
    10: [255, 150, 255],
    11: [75, 0, 75],
    12: [75, 0, 175],
    13: [0, 200, 255],
    14: [50, 120, 255],
    15: [0, 175, 0],
    16: [0, 60, 135],
    17: [80, 240, 150],
    18: [150, 240, 255],
    19: [0, 0, 255],
}

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def list_segments_points(p_coord, p_feats, labels):
    c_coord = []
    c_feats = []

    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue

            batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:,:]
            segment_coord[:,0] = seg_batch_count
            seg_batch_count += 1

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)

    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)

    device = torch.device("cuda")

    return ME.SparseTensor(
                features=seg_feats,
                coordinates=seg_coord,
                device=device,
            )

def select_cluster(c, y, cluster):
    colors = np.ones((len(y),3))*.5#plt.get_cmap("prism")(y / (y.max() if y.max() > 0 else 1))
    colors[y == cluster] = [1.,0.,0.]
    pcd_gt = o3d.geometry.PointCloud()
    c_ = c.copy()
    c_[:,2] += 1500
    pcd_gt.points = o3d.utility.Vector3dVector(c_)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd_gt])

    select = input()

    return select == 'n'

def find_nearest_neighbors(pcd, points):
    nearest_points = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[p], 8)
        nearest_points += list(np.asarray(idx))

    return nearest_points

def visualize_pcd_clusters(p, p_corr, p_slc, gt, cmap="viridis", center_point=None, quantize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p[:,:3])

    labels = p[:, -1]
    colors = plt.get_cmap(cmap)(labels)

    # labels = p_slc[:, -1][:,np.newaxis]
    # colors = np.concatenate((labels, labels, labels), axis=-1)
    
    # if center_point is not None:
    #     lbl = np.argsort(labels)
    #     colors[lbl[-20:],:3] = [1., 0., 0.]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_corr = o3d.geometry.PointCloud()
    pcd_corr.points = o3d.utility.Vector3dVector(p_corr[:,:3])

    labels = p_corr[:, -1]
    colors = plt.get_cmap(cmap)(labels)
    
    if center_point is not None:
        lbl = np.argsort(labels)
        colors[lbl[-20:],:3] = [1., 0., 0.]
    pcd_corr.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_slc = o3d.geometry.PointCloud()
    pcd_slc.points = o3d.utility.Vector3dVector(p_slc[:,:3])

    labels = p_slc[:, -1]
    colors = plt.get_cmap(cmap)(labels)
    
    if center_point is not None:
        lbl = np.argsort(labels)
        colors[lbl[-20:],:3] = [1., 0., 0.]
    pcd_slc.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd = pcd.voxel_down_sample(voxel_size=5)
    pcd_corr = pcd_corr.voxel_down_sample(voxel_size=5)
    pcd_slc = pcd_slc.voxel_down_sample(voxel_size=5)
    gt = gt.voxel_down_sample(voxel_size=5)

    colors_gt = np.asarray(gt.colors).copy()
    colors_gt[:,0] = np.maximum(colors_gt[:,0], 0.2)
    colors = np.asarray(pcd.colors)

    colors[:,0] = colors_gt[:,0]*colors[:,0]
    colors[:,1] = colors_gt[:,0]*colors[:,1]
    colors[:,2] = colors_gt[:,0]*colors[:,2]
    #colors = plt.get_cmap(cmap)(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd, pcd_corr, pcd_slc, gt])
