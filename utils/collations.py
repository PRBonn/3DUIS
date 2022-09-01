import numpy as np
import MinkowskiEngine as ME
import torch

def array_to_sequence(batch_data):
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ME.SparseTensor(
                features=seg_feats,
                coordinates=seg_coord,
                device=device,
            )

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    return ME.TensorField(
            features=p_feats,
            coordinates=p_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device,
        )

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    #_, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    # if len(mapping) > num_points:
    #     if deterministic:
    #         # for reproducibility we set the seed
    #     np.random.seed(42)
    #     mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord, p_feats, labels

def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j


class SparseCollation:
    def __init__(self, resolution, num_points, instance_labels='gt'):
        self.resolution = resolution
        self.num_points = num_points
        self.instance_labels = instance_labels

    def __call__(self, list_data):
        batch_data = list_data[0]#list(zip(*list_data))

        points = batch_data['points_cluster']
        sem_labels = batch_data['semantic_label']
        ins_labels = batch_data['instance_label']

        p = np.asarray(points)

        p_feats = []
        p_coord = []
        p_cluster = []

        #for p in points:
        # p[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
        coord_p, feats_p, cluster_p = point_set_to_coord_feats(p[:,:-1], p[:,-1], self.resolution, self.num_points)
        p_coord.append(coord_p)
        p_feats.append(feats_p)

        p_feats = np.asarray(p_feats)
        p_coord = np.asarray(p_coord)

        segment = np.asarray(cluster_p)

        # if not segment_contrast segment_i and segment_j will be an empty list
        return {'coord': p_coord, 'feats': p_feats, 'cluster': segment, 'semantic_label': sem_labels, 'instance_label': ins_labels if self.instance_labels == 'gt' else ins_labels,
                    'scan_file': batch_data['scan_file']}


