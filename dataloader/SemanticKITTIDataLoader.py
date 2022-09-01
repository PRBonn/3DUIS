import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from utils.data_map import *
from utils.pcd_preprocess import *
import MinkowskiEngine as ME
import torch
import json
import os

warnings.filterwarnings('ignore')

class SemanticKITTIDataLoader(Dataset):
    def __init__(self, root,  split='train'):
        self.root = root
        self.augmented_dir = 'augmented_views_patchwork'

        if not os.path.isdir(os.path.join(self.root, 'assets', self.augmented_dir)):
            os.makedirs(os.path.join(self.root, 'assets', self.augmented_dir))

        self.seq_ids = {}
        self.seq_ids['train'] = [ '00' , '01', '02', '03', '04', '05', '06', '07', '09', '10']
        self.seq_ids['validation'] = ['08']
        self.seq_ids['test'] = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.split = split

        assert (split == 'train' or split == 'validation' or split == 'test')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list(split)

        print('The size of %s data is %d'%(split,len(self.points_datapath)))

    def datapath_list(self, split):
        self.points_datapath = []
        self.labels_datapath = []
        self.instance_datapath = []

        for seq in self.seq_ids[split]:
            point_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            try:
                label_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'labels')
                point_seq_label = os.listdir(label_seq_path)
                point_seq_label.sort()
                self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
            except:
                pass

            try:
                instance_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'instances')
                point_seq_instance = os.listdir(instance_seq_path)
                point_seq_instance.sort()
                self.instance_datapath += [ os.path.join(instance_seq_path, instance_file) for instance_file in point_seq_instance ]
            except:
                pass

    def __len__(self):
        return len(self.points_datapath)

    def __getitem__(self, index):
        # we need to preprocess the data to get the cuboids and guarantee overlapping points
        # so if we never have done this we do and save this preprocessing
        # index = 1491
        # print(self.points_datapath[index])
        seq_num = self.points_datapath[index].split('/')[-3]
        fname = self.points_datapath[index].split('/')[-1].split('.')[0]
        cluster_path = os.path.join(self.root, 'assets', self.augmented_dir, seq_num)
        if os.path.isfile(os.path.join(cluster_path, fname + '.npy')):
            # if the preprocessing is done and saved already we simply load it
            points_set = np.load(os.path.join(cluster_path, fname + '.npy'))
            # Px5 -> [x, y, z, i, c] where i is the intesity and c the Cluster associated to the point

        else:
            # if not we load the full point cloud and do the preprocessing saving it at the end
            points_set = np.fromfile(self.points_datapath[index], dtype=np.float32)
            points_set = points_set.reshape((-1, 4))

            # remove ground and get clusters from point cloud
            points_set = clusterize_pcd(points_set, self.points_datapath[index])
            #visualize_pcd_clusters(points_set, points_set, points_set)

            # Px5 -> [x, y, z, i, c] where i is the intesity and c the Cluster associated to the point
            if not os.path.isdir(cluster_path):
                os.makedirs(cluster_path)
            np.save(os.path.join(cluster_path, fname), points_set)

        if self.split != 'test':
            labels = np.fromfile(self.labels_datapath[index], dtype=np.uint32)
            labels = labels.reshape((-1))
            sem_labels = labels & 0xFFFF
            #sem_labels = np.vectorize(learning_map.get)(sem_labels)
            #sem_labels = np.expand_dims(sem_labels, axis=-1)
            unlabeled = None#sem_labels[:,0] == 0
            # sem_labels = np.delete(sem_labels, unlabeled, axis=0)
            # points_set = np.delete(points_set, unlabeled, axis=0)

            #ins_labels = labels >> 16
            #ins_labels = np.delete(ins_labels, unlabeled, axis=0)

            ins_labels = labels >> 16
        else:
            sem_labels = None
            ins_labels = None

        # .ins files already have the unlabeled data removed
        # ins_labels = np.delete(ins_labels, unlabeled, axis=0)

        # now the point set returns [x,y,z,i,c] always
        return {'points_cluster': points_set, 'semantic_label': sem_labels, 'instance_label': ins_labels, 'scan_file': self.points_datapath[index]}

