import os
import numpy as np
import torch

import glob

root_path = '/home/pengzhen/code/pointcloud_dataset_set/dataset/s3dis'

files = sorted(glob.glob(os.path.join(root_path, "learned_superpoint_graph_segmentations/*.npy")))

for file in files:
    chunks = file.split("/")[-1].split(".")
    area = chunks[0]
    room = chunks[1]

    spp = np.load(file, allow_pickle=True).item()["segments"]

    if not os.path.exists(os.path.join(root_path, "ISBNet_superpoints")):
        os.makedirs(os.path.join(root_path, "ISBNet_superpoints"))

    torch.save((spp), f"/home/pengzhen/code/pointcloud_dataset_set/dataset/s3dis/ISBNet_superpoints/{area}_{room}.pth")
