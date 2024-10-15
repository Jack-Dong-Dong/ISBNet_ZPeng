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

def read_mesh_file(mesh_dir):
    single_tree_mesh_lists = glob.glob(mesh_dir + "/*")
    mesh_merge = o3d.geometry.TriangleMesh()
    for single_mesh_txt in single_tree_mesh_lists:
        mesh = o3d.io.read_triangle_mesh(single_mesh_txt)
        mesh_merge += mesh

    return mesh_merge

def merge_mesh(data_dir):
    file_lists_dir = os.path.join(data_dir, "SYSSIFOSS_v1.0_Aligned_version")
    save_lists_dir = os.path.join(data_dir, 'ISBNet_meshfile')

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"SYSSIFOSS_merge_mesh_{timestamp}.log"

    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
        log_line = f"-----处理项目: SYSSIFOSS合并mesh文件-----\n"
        log_file.write(log_line)

    scene_list = []

    file_lists = glob.glob(file_lists_dir + "/*")
    for file_list in file_lists:
        scene_list.append(file_list.split("/")[-1])

    scene_list = natsort.natsorted(scene_list)


    if not os.path.exists(save_lists_dir):
        os.makedirs(save_lists_dir)
    save_dir_list = glob.glob(save_lists_dir + "/*")

    for scene_name in tqdm(scene_list, position=0):
        with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
            log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-----\n"
            log_file.write(log_line)

        scene_pth = os.path.join(save_lists_dir, f"{scene_name}_meshfile.ply")
        if scene_pth in save_dir_list:
            with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
                log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-aleady----\n"
                log_file.write(log_line)
        else:
            mesh_dir = os.path.join(file_lists_dir, scene_name, 'mesh_file')
            mesh_data = read_mesh_file(mesh_dir)
            o3d.io.write_triangle_mesh(save_lists_dir + "/" + scene_name + "_meshfile.ply", mesh_data)

            with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
                log_line = f"--mesh_path---{save_lists_dir}--\n"
                log_file.write(log_line)
            

cfg = parser.parse_args()
merge_mesh(cfg.data_dir)
