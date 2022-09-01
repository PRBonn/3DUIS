import open3d as o3d
import numpy as np
from utils.corr_utils import crop_region
import matplotlib.pyplot as plt

pred = np.load('./output/3DUIS/08/raw_pred/000000.npy')
points = pred[:,:3]
pred = pred[:,-1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

colors_pred = np.zeros((len(pred), 4))
colors_cluster = np.zeros((len(pred), 4))
flat_indices = np.unique(pred)
max_instance = len(flat_indices)
colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

for idx in range(len(flat_indices)):
    colors_pred[pred == idx] = colors_instance[int(idx)]

colors_pred[pred == 0] = [0.,0.,0.,0.]

pcd.colors = o3d.utility.Vector3dVector(colors_pred[:,:3])

o3d.visualization.draw_geometries([pcd])
