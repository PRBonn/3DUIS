import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import multivariate_normal
from scipy.special import softmax
import networkx as nx
import os

def vis_pcd_saliency_corr(points, slc, corr, instance, corr_points=None):
    if corr_points is not None:
        points = points[corr_points]
        # slc = slc[corr_points]
        #corr = corr[corr_points]
        instance = instance[corr_points]

    # pcd with instances
    pcd_ins = o3d.geometry.PointCloud()
    pcd_ins.points = o3d.utility.Vector3dVector(points[:, :3])
    colors_ins = np.ones((len(points), 3)) * .5
    colors_ins = plt.get_cmap("prism")(instance)
    colors_ins[instance == 0] = [0.,0.,0.,0.]
    pcd_ins.colors = o3d.utility.Vector3dVector(colors_ins[:, :3])

    # pcd with saliency
    pcd_slc = o3d.geometry.PointCloud()
    slc_points = points.copy()
    slc_points[:,2] += 400
    pcd_slc.points = o3d.utility.Vector3dVector(slc_points[:, :3])
    colors_slc = plt.get_cmap("viridis")(slc)
    pcd_slc.colors = o3d.utility.Vector3dVector(colors_slc[:, :3])

    # pcd with correlation
    pcd_cor = o3d.geometry.PointCloud()
    cor_points = points.copy()
    cor_points[:,2] += 200
    pcd_cor.points = o3d.utility.Vector3dVector(cor_points[:, :3])
    colors_cor = np.ones((len(points), 3)) * .5
    colors_cor = plt.get_cmap("prism")(corr)
    colors_cor[corr == 0 ] = [0.,0.,0.,0.]
    pcd_cor.colors = o3d.utility.Vector3dVector(colors_cor[:, :3])

    o3d.visualization.draw_geometries([pcd_ins, pcd_slc, pcd_cor])

def crop_region(points, cluster_label, curr_cluster, growth_region=0.):
    cls_points = np.where(cluster_label == curr_cluster)[0]
    
    cluster_center = points[cls_points].mean(axis=0)

    #center_dists = np.sqrt(np.sum((points - cluster_center)**2, axis=-1))
    #farthest_point = np.argmax(center_dists[cls_points])

    # look for points around the cluster with radius = farthest_dist + growth_region
    max_xyz  = np.max(np.abs(points[cls_points] - cluster_center), axis=0) + growth_region

    # drop wrong labels
    if np.sum(max_xyz > 300.):
        return np.zeros((len(points),)).astype(bool)

    upper_idx = np.sum((points[:,:3] < cluster_center + max_xyz).astype(np.int32), 1) == 3
    lower_idx = np.sum((points[:,:3] > cluster_center - max_xyz).astype(np.int32), 1) == 3

    return ((upper_idx) & (lower_idx))

def parse_scan_file(filename):
    scan_info = filename.split('/')
    scan_file = scan_info[-1].split('.')[0]
    seq_num = scan_info[-3]

    return scan_info, scan_file, seq_num

def dump_preds(pred, seq, scan_file, config, pred_id=''):
    dump_path = os.path.join(config['experiment']['output_dir'], config['experiment']['id'], seq, pred_id)
    if not os.path.isdir(dump_path):
        os.makedirs(dump_path)
    dump_path = os.path.join(dump_path, scan_file)
    np.save(dump_path, pred)

def get_ground_labels(batch_data):
    # get ground labels and voxelize it
    scan_info, scan_file, seq_num = parse_scan_file(batch_data['scan_file'])
    ground_file = f'./Datasets/SemanticKITTI/assets/patchwork/{seq_num}/{scan_file}.label'
    ground_labels = np.fromfile(ground_file, dtype=np.uint32)
    ground_labels.reshape((-1))
    #ground_labels = np.delete(ground_labels, batch_data['clean_points'], axis=0)
    ground_labels = ground_labels[batch_data['vox_map']]

    return ground_labels

def get_cluster_correlation(points, feats, corr_points, corr_center):
    center_dists = np.sqrt(np.sum((points[corr_points] - corr_center)**2, axis=-1))
    corr_center = np.argmin(center_dists)

    cluster_embed = feats
    cluster_corr = np.corrcoef(cluster_embed)
    
    return cluster_corr, corr_center

def get_cluster_saliency(x, out, grad_point, corr_points=None):
    if corr_points is None:
        score = torch.mean(torch.mean(out.F[grad_point], dim=-1))
    else:
        score = torch.mean(torch.mean(out.F[corr_points][grad_point], dim=-1))
    x.F.retain_grad()
    score.backward(retain_graph=True)

    slc, _ = torch.max(torch.abs(x.F.grad), dim=-1)
    slc = torch.tanh((slc - slc.mean()) / slc.std())
    slc = slc.cpu().detach().numpy()

    return slc

def cosine_similarity(a, b):
    dot = np.multiply(a,b)
    dot = np.sum(dot, axis=-1)
    a_ = np.sqrt(np.sum(a**2, axis=-1))
    b_ = np.sqrt(np.sum(b**2, axis=-1))

    return 1. - (dot / (a_ * b_))

def euclidean(a, b):
    return np.linalg.norm(a-b)

def manhattan(a, b):
    return np.sum(np.abs(a-b))

def compute_pdfs(feats, inliers):
    comps = np.array([]).reshape(0,feats.shape[-1])

    # compute pdf
    for i in inliers:
        comps = np.vstack([comps, feats[i,:]])

    comps = np.vstack([comps, comps]) if len(comps) == 1 else comps

    mu, sigma = np.mean(comps, axis=0), np.cov(comps.T)

    return (mu, sigma)

def retrieve_knn(points, p, k):
    dists = np.sqrt(np.sum((points - points[p])**2, axis=-1))
    #dists = np.sqrt(np.sum((points - p)**2, axis=-1))

    return np.argsort(dists)[1:(k+1)]

def retrieve_knn_feats(feats, p, k):
    dot = feats@feats[p]
    a_ = np.sqrt(np.sum(feats**2, axis=-1))
    b_ = np.sqrt(np.sum(feats[p]**2, axis=-1))

    dists = 1. - (dot / (a_ * b_))

    return np.argsort(dists)[1:(k+1)]

def affinity(sim, sigma):
    return np.exp(-sim / (2 * (sigma**2)))

def sample_source(source_seeds, points, k=8):
    source = []
    for seed in source_seeds:
        source += retrieve_knn(points, seed, k).tolist()

    return np.asarray(source)

dissimilarity_function = {
    'euclidean': euclidean,
    'cosine': cosine_similarity,
    'manhattan': manhattan,
}

def build_graph(feats, slc, points, center, source_num, params, ground, cluster_idx):
    G = nx.Graph()
    G.add_node('source')
    G.add_node('sink')

    # select seeds sampling method
    if params['graphcut']['source_sampling'] == 'feats_distance':
        source_seeds = retrieve_knn_feats(feats, center, max(1, int(source_num/params['graphcut']['source_factor'])))
        source = sample_source(source_seeds, points)
        # seeds outside the cluster are discarded
        source = source[np.in1d(source, cluster_idx)]
    elif params['graphcut']['source_sampling'] == 'saliency':
        source_seeds = np.argsort(slc[:,0])[-max(1,int(source_num/params['graphcut']['source_factor'])):]
        source = sample_source(source_seeds, points)

        # seeds outside the cluster are discarded
        source = source[np.in1d(source, cluster_idx)]
    elif params['graphcut']['source_sampling'] == 'saliency_distance':
        dot = feats@feats[center]
        a_ = np.sqrt(np.sum(feats**2, axis=-1))
        b_ = np.sqrt(np.sum(feats[center]**2, axis=-1))

        dists = 1. - (dot / (a_ * b_))
        slc[:,0] = 0.5 * slc[:,0] + 0.5 * dists
        source_seeds = np.argsort(slc[:,0])[-max(1,int(source_num/params['graphcut']['source_factor'])):]
        source = sample_source(source_seeds, points)

    # remove ground points from sampled foreground seeds if any
    if params['graphcut']['ground_remove']['seed_sampling']:
        ground_idx = np.where(ground == 9)[0]
        source = source[~np.in1d(source, ground_idx)]

    # remove proposal points from background seeds if any
    sink_num = max(1, int((len(points) - source_num)/params['graphcut']['sink_factor']))
    sink = np.argsort(slc[:,0])[:sink_num]
    sink = sink[~np.in1d(sink, cluster_idx)]

    # define the points probs for foreground and background
    src_probs = np.ones((len(points),))*1e-20
    snk_probs = np.ones((len(points),))*1e-20

    # set the select seeds with high probability
    src_probs[source] = 1.
    snk_probs[sink] = 1.

    lambda_ = params['graphcut']['lambda']
    omega_ = params['graphcut']['omega']
    sigma = params['graphcut']['sigma']

    if params['graphcut']['visualization']:
        graph_pcd = np.zeros((0,4))

    # build graph
    for i in range(len(points)):
        # edges between non-terminal and terminal vertex
        G.add_node(f'{i}')
        G.add_edge('source', f'{i}')
        G['source'][f'{i}']['capacity'] = -lambda_ * np.log(src_probs[i])

        G.add_edge(f'{i}', 'sink')
        G[f'{i}']['sink']['capacity'] = -lambda_ * np.log(snk_probs[i])

        # edges between non-terminal and non-terminal vertex
        neighbor_index = retrieve_knn(points, i, params['graphcut']['k_neighbors'])

        if params['graphcut']['visualization']:
            edge_color = 0.
            edge_coord = points[i]

        # compute the edges weights between each point and its K neighbors
        for k in neighbor_index:

            G.add_edge(f'{i}', f'{k}')

            dissim_ik = dissimilarity_function[params['graphcut']['similarity_func']](feats[i], feats[k])
            diff_ik = affinity(dissim_ik, sigma)
            G[f'{i}'][f'{k}']['capacity'] = omega_ * diff_ik

            if params['graphcut']['visualization']:
                edge_color += omega_ * diff_ik

        if params['graphcut']['visualization']:
            edge_color /= params['graphcut']['k_neighbors']
            edge_point = np.hstack((edge_coord, edge_color))
            graph_pcd = np.vstack((graph_pcd, edge_point))

    if params['graphcut']['visualization']:
        print(f'Edges: max: {graph_pcd[:,-1].max()}\tmin: {graph_pcd[:,-1].min()}\tmean: {graph_pcd[:,-1].mean()}')
        pcd = o3d.geometry.PointCloud()
        graph_pcd[:,2] += 300
        pcd.points = o3d.utility.Vector3dVector(graph_pcd[:,:3])
        graph_pcd[:,-1] = (graph_pcd[:,-1] - graph_pcd[:,-1].mean()) / graph_pcd[:,-1].std()
        colors = plt.get_cmap('viridis')(graph_pcd[:,-1])
        pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
        o3d.visualization.draw_geometries([pcd])

    return G

def graph_cut(G):
    _, partition = nx.minimum_cut(G, "source", "sink")
    reachable, non_reachable = partition

    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    ins_points = []
    for px in list(cutset):
        if px[1] == 'sink':
            continue
        ins_points.append(int(px[1]))

    return np.asarray(ins_points)
