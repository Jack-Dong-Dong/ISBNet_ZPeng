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
    "--data_dir", type=str, default="/home/pengzhen/code/pointcloud_dataset_set/dataset/SYSSIFOSS", help="Path to the original data"
)

list_of_species_colors = np.array([
    (174, 199, 232),  # AbiAlb
    (152, 223, 138),  # AceCam
    (31, 119, 180),  # AcePse
    (255, 187, 120),  # BetPen
    (188, 189, 34),  # CarBet
    (140, 86, 75),  # FagSyl
    (255, 152, 150),  # FraExc
    (214, 39, 40),  # JugReg
    (197, 176, 213),  # LarDec
    (148, 103, 189),  # PicAbi
    (196, 156, 148),  # PinSyl
    (23, 190, 207),  # PruAvi
    (178, 76, 76), # PruSer
    (247, 182, 210),  # PseMen
    (66, 188, 102),  # QuePet
    (219, 219, 141),  # QueRob
    (140, 57, 197),  # QueRub
    (202, 185, 52),  # RobPse
    (51, 176, 203),  # SalCap
    (200, 54, 131),  # SorTor
    (92, 193, 61),  # TilSpe
    (78, 71, 183),  # TsuHet
    (0, 0, 0),  # clutter
])


def get_single_tree_data(data_dir):

    single_tree_txt = np.loadtxt(data_dir)
    xyz = single_tree_txt[:, :3]
    colors = np.zeros((len(single_tree_txt), 3), dtype=np.uint64)
    color_ins = single_tree_txt[:, 3]
    for i, color_index in enumerate(color_ins.astype(int)):
         # 确保索引合法
        colors[i] = list_of_species_colors[color_index]

    return xyz, colors

def set_mesh_file(xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))

    return mesh

def compute_triangle_mesh(data_dir):
    annotations_dir = os.path.join(data_dir, 'Annotations')
    meshfile_dir = os.path.join(data_dir, 'mesh_file')
    if not os.path.exists(meshfile_dir):
        os.makedirs(meshfile_dir)

    annotations_file_lists = glob.glob(annotations_dir + '/*')
    for annotations_file_list in tqdm(annotations_file_lists, position=0):
        if '.txt' in annotations_file_list:
            single_tree_name = annotations_file_list.split('/')[-1].split('.')[0]
            xyz, colors = get_single_tree_data(annotations_file_list)
            mesh_file = set_mesh_file(xyz, colors)
            o3d.io.write_triangle_mesh(meshfile_dir + "/" + single_tree_name + ".ply", mesh_file)



def preprocess_triangle_mesh_single(data_dir):
    file_lists_dir = os.path.join(data_dir, "SYSSIFOSS_v1.0_Aligned_version")
    file_lists = glob.glob(file_lists_dir + '/*')
    # compute_triangle_mesh(file_lists[0])

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"SYSSIFOSS_prepare_data_{timestamp}.log"

    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
        log_line = f"-----处理项目: SYSSIFOSS-single-meshFile生成-----\n"
        log_file.write(log_line)

    scene_list = []

    file_lists = glob.glob(file_lists_dir + "/*")
    for file_list in file_lists:
        txt_lists = glob.glob(file_list + "/*")
        for txt_list in txt_lists:
            if '.txt' in txt_list:
                scene_list.append(file_list.split("/")[-1])

    scene_list = natsort.natsorted(scene_list)

    for file_list in file_lists:
        data_lists = glob.glob(os.path.join(file_list + '/*'))
        for data_list in data_lists:
            if 'Annotations' in data_list:
                compute_triangle_mesh(file_list)


cfg = parser.parse_args()
preprocess_triangle_mesh_single(cfg.data_dir)
