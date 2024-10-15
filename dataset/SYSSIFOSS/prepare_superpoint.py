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

def get_superpoint(mesh_file, log_file_name):

    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
        log_line = f"--superpoint_path---{mesh_file}--\n"
        log_file.write(log_line)

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()
    return superpoint




def prepare_superpoint(data_dir):
    preprocess_dir = os.path.join(data_dir, "ISBNet_meshfile")
    superpoint_dir = os.path.join(data_dir, "ISBNet_superpoint")

    if not os.path.exists(preprocess_dir):
        return KeyError("Preprocess directory does not exist")

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"SYSSIFOSS_prepare_superpoint_{timestamp}.log"

    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
        log_line = f"-----处理项目: SYSSIFOSS superpoint预处理-----\n"
        log_file.write(log_line)

    if not os.path.exists(superpoint_dir):
        os.makedirs(superpoint_dir)
    save_dir_list = glob.glob(superpoint_dir + "/*")

    preprocess_data_lists = glob.glob(preprocess_dir + "/*")
    preprocess_data_lists = natsort.natsorted(preprocess_data_lists)
    for preprocess_data in tqdm(preprocess_data_lists, position=0):

        print('------------preprocess_data:', preprocess_data, '----------------')
        scene_name = preprocess_data.split("/")[-1].split("_")[0]

        with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
            log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-----\n"
            log_file.write(log_line)

        scene_pth = os.path.join(superpoint_dir, f"{scene_name}_superpoint")
        if scene_pth in save_dir_list:
            with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
                log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-aleady----\n"
                log_file.write(log_line)
        else:
            os.makedirs(superpoint_dir, exist_ok=True)

            spp = get_superpoint(preprocess_data, log_file_name)

            torch.save(spp, os.path.join(superpoint_dir, f"{scene_name}_superpoint.pth"))

cfg = parser.parse_args()
prepare_superpoint(cfg.data_dir)
