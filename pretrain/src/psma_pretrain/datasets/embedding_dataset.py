import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .build import DATASETS

from .data_process import Ply2Embedding, Ply2Embedding_torch, read_ply_to_point_cloud

@DATASETS.register_module()
class EmbeddingDataset(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.use_normals = config.USE_NORMALS
        self.csv_file = config.get('CSV_FILE', None)  # CSV file path is optional
        split = config.subset
        self.subset = config.subset
        self.kpoints = config.kpoints
        self.pretrain = False

        # Load labels from CSV if provided
        if self.csv_file:
            print(f"Get csv file from {self.csv_file}")
            self.labels = pd.read_csv(self.csv_file, index_col='uniprot_id')['ogt'].to_dict()
        else:
            self.pretrain = True
            self.labels = None

        # Generate data paths
        if self.pretrain:
            self.datapath = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.ply')]
        else: 
            self.datapath = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.npy')]

        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        file_path = self.datapath[index]
        uniprot_id = os.path.splitext(os.path.basename(file_path))[0]
        if self.pretrain:
            point_set = read_ply_to_point_cloud(file_path) # Point Cloud
        else:
            # Load point set
            point_set = np.load(file_path).astype(np.float32) # Patches Embedding

        # Reshape point set if needed
        if point_set.shape[1] != self.kpoints:
            raise ValueError('Point set shape does not match config.N_POINTS')

        # Get label
        label = -1
        if self.labels:
            label = self.labels.get(uniprot_id, -1)
            if label == -1:
                raise ValueError('Label not found for uniprot_id: {}'.format(uniprot_id))
            label = np.array([label]).astype(np.int32)

        # If normals are disabled, truncate to XYZ + normals (first 6 channels).
        # if not self.use_normals:
        #     point_set = point_set[:, :, 0:6]

        return point_set, label

    def __getitem__(self, index):
        points, label = self._get_item(index)
        current_points = torch.from_numpy(points).float()
        return 'EmbeddingDataset', 'sample', (current_points, label)
