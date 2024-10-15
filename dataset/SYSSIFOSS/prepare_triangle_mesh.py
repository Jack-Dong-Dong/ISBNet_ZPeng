import numpy as np
import torch

import open3d as o3d
import os
import segmentator
from tqdm import tqdm

import configargparse
import glob
import sys
import natsort
import datetime

sys.path.append(".")

parser = configargparse.ArgumentParser()
parser.add_argument(
    # "--data_dir", type=str, default="/home/pengzhen/code/pointcloud_dataset_set/dataset/SYSSIFOSS", help="Path to the original data"
    "--data_dir", type=str, default="/home/pengzhen/code/pointcloud_dataset_set", help="Path to the original data"
)

def set_meshfile(mesh_file):

    print('------------load mesh_file from:', mesh_file, '----------------')
    # label = torch.load(mesh_file)
    # xyz = np.array(label[0])
    # colors = np.array(label[1])
    print('------------init pcd','----------------')
    pcd = o3d.io.read_point_cloud('/home/pengzhen/code/pointcloud_dataset_set/ISBNet_meshfile/BR01.ply')
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.estimate_normals()
    print('------------computing----------------')
    # pcd.voxel_down_sample(voxel_size=0.5)
    # return pcd
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    print('-------------- creat ----------------')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    return mesh




def prepare_superpoint(data_dir):
    preprocess_dir = os.path.join(data_dir, "ISBNet_preprocess")
    meshfile_dir = os.path.join(data_dir, "ISBNet_meshfile")


    if not os.path.exists(meshfile_dir):
        os.makedirs(meshfile_dir)

    msehfile = set_meshfile('')
    o3d.io.write_triangle_mesh(meshfile_dir + "/" + "2.ply", msehfile)

    # preprocess_data_lists = glob.glob(preprocess_dir + "/*")
    # for preprocess_data in preprocess_data_lists:
    #     os.makedirs(meshfile_dir, exist_ok=True)
    #     scene_name = preprocess_data.split("/")[-1].split("_")[0]
    #     print(scene_name)
    #
    #     msehfile = set_meshfile(preprocess_data)
    #     print('------------save mesh_file to:', meshfile_dir + "/" + scene_name + ".ply", '----------------')
    #     # o3d.io.write_point_cloud(meshfile_dir + "/" + scene_name + ".ply", msehfile)
    #     o3d.io.write_triangle_mesh(meshfile_dir + "/" + scene_name + "2.ply", msehfile)

cfg = parser.parse_args()
prepare_superpoint(cfg.data_dir)
