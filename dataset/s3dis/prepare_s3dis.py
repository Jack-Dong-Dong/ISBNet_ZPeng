import numpy as np
import torch
from scipy.spatial import KDTree
from tqdm import tqdm

import configargparse
import glob
import natsort
import os
import sys
import datetime


sys.path.append(".")

parser = configargparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="/home/pengzhen/code/pointcloud_dataset_set/dataset/s3dis/Stanford3dDataset_v1.2_Aligned_Version", help="Path to the original data"
)

S3DIS_SEMANTICS_COLORS = np.array(
    [
        (174, 199, 232),  # ceiling
        (152, 223, 138),  # floor
        (31, 119, 180),  # wall
        (255, 187, 120),  # column
        (188, 189, 34),  # beam
        (140, 86, 75),  # window
        (255, 152, 150),  # door
        (214, 39, 40),  # table
        (197, 176, 213),  # chair
        (148, 103, 189),  # bookcase
        (196, 156, 148),  # sofa
        (23, 190, 207),  # board
        (178, 76, 76),
    ]  # clutter
)

INS_COLORS = np.array(
    [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(1000)]
)


ID2NAME = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter",
}
ID2NAME = [ID2NAME[i] for i in range(13)]
NAME2ID = {}
for i in range(13):
    NAME2ID[ID2NAME[i]] = i


def get_labels(scene_name, scene_data, data_dir, log_file_name):
    area = scene_name.split(".")[0]
    name = scene_name.split(".")[1]
    instance_pths = glob.glob(data_dir + "/" + area + "/" + name + "/Annotations/*.txt")
    # print('instance_pths', instance_pths)

    scene_pts = scene_data[:, :3]  # Scene point cloud
    pt_tree = KDTree(scene_pts)

    error = 0
    instances = np.zeros((len(scene_data), 1), dtype=np.int32) - 1
    semantics = np.zeros((len(scene_data), 1), dtype=np.float32) - 1

    # Use nearest neighbor to find corresponding point indexes in the scenes PC of instances
    for instance_id, pth in enumerate(instance_pths):
        class_name = pth.split("/")[-1].split("_")[0]
        if not (class_name in NAME2ID.keys()):
            if class_name == "stairs":
                class_name = "clutter"
        semantic_id = NAME2ID[class_name]
        # Load instance point cloud

        # save logs
        with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
            log_line = f"--instances_path---{pth}--\n"
            log_file.write(log_line)

        instance_data = np.loadtxt(pth)
        instance_pts = instance_data[:, :3]
        # instance_colors = instance_data[:, 3:]
        # Find corresponding indices in the scene points
        dist, pt_indexs = pt_tree.query(instance_pts, k=1)
        instances[pt_indexs] = instance_id
        semantics[pt_indexs] = semantic_id
        error += dist.sum()

    decided = (instances >= 0)[:, 0]

    # For some points are not annotated, use the label from nearby points
    pt_tree = KDTree(scene_pts[decided])
    dist, decided_indexs = pt_tree.query(scene_pts[~decided], k=1)

    instances[~decided] = instances[decided][decided_indexs]
    semantics[~decided] = semantics[decided][decided_indexs]

    assert (instances.min()) >= 0
    assert (semantics.min()) >= 0

    # Avoiding duplicate instances -> instance ids are contiguous from 0
    remap_id = np.array(range(instances.max() + 1))
    for new_id, old_id in enumerate(np.unique(instances)):
        remap_id[old_id] = new_id
    instances = remap_id[instances].astype(np.float32)
    unique_instances = np.unique(instances)

    assert np.all(unique_instances == range(len(unique_instances)))

    return instances, semantics


def read_scene_txt(name, data_dir):
    area = name.split(".")[0]
    name = name.split(".")[1]

    pts = np.loadtxt(os.path.join(data_dir + "/" + area, name, name + ".txt"))

    return pts


def preprocess_s3dis(data_dir):
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"s3dis_prepare_data_{timestamp}.log"

    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
        log_line = f"-----处理项目: S3DIS数据预处理-----\n"
        log_file.write(log_line)
    
    scene_list = []
    for i in range(1, 7):
        area = data_dir + "/Area_" + str(i)
        tmp = glob.glob(area + "/*")
        for scene_name in tmp:
            scene_name = scene_name.split("/")[-2] + "." + scene_name.split("/")[-1]
            if '.txt' in scene_name:
                continue
            else:
                scene_list.append(scene_name)

    scene_list = natsort.natsorted(scene_list)

    save_dir_list = glob.glob("/home/pengzhen/code/pointcloud_dataset_set/dataset/s3dis/ISBNet_preprocess/*")

    for scene_name in tqdm(scene_list, position=0):

        with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
            log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-----\n"
            log_file.write(log_line)

        area = scene_name.split(".")[0]
        name = scene_name.split(".")[1]
        save_dir = "/home/pengzhen/code/pointcloud_dataset_set/dataset/s3dis/ISBNet_preprocess"
        scene_pth = os.path.join(save_dir, f"{area}_{name}_inst_nostuff")
        if scene_pth in save_dir_list:
            with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
                log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-aleady----\n"
                log_file.write(log_line)
        else: 
            os.makedirs(save_dir, exist_ok=True)

            scene_data = read_scene_txt(scene_name, data_dir)
            instances, semantics = get_labels(scene_name, scene_data, data_dir, log_file_name)

            torch.save(
                (
                    scene_data[:, :3].astype(np.float32),
                    scene_data[:, 3:6].astype(np.float32),
                    semantics.reshape(-1).astype(np.int32),
                    instances.reshape(-1).astype(np.int32),
                ),
                scene_pth,
            )


    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
            log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} ------done-----\n"
            log_file.write(log_line)


cfg = parser.parse_args()
preprocess_s3dis(cfg.data_dir)
