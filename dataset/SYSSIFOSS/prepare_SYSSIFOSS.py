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
    "--data_dir", type=str, default="/home/pengzhen/code/pointcloud_dataset_set/dataset/SYSSIFOSS", help="Path to the original data"
    # "--data_dir", type=str, default="/home/pengzhen/code/pointcloud_dataset_set/dataset/s3dis/Stanford3dDataset_v1.2_Aligned_Version", help="Path to the original data"
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

list_of_species_NAME2ID = {
    'AbiAlb': 0,
    'AceCam': 1,
    'AcePse': 2,
    'BetPen': 3,
    'CarBet': 4,
    'FagSyl': 5,
    'FraExc': 6,
    'JugReg': 7,
    'LarDec': 8,
    'PicAbi': 9,
    'PinSyl': 10,
    'PruAvi': 11,
    'PruSer': 12,
    'PseMen': 13,
    'QuePet': 14,
    'QueRob': 15,
    'QueRub': 16,
    'RobPse': 17,
    'SalCap': 18,
    'SorTor': 19,
    'TilSpe': 20,
    'TsuHet': 21,
    'clutter': 22,
}

list_of_species_ID2NAME = {
    0: 'AbiAlb',
    1: 'AceCam',
    2: 'AcePse',
    3: 'BetPen',
    4: 'CarBet',
    5: 'FagSyl',
    6: 'FraExc',
    7: 'JugReg',
    8: 'LarDec',
    9: 'PicAbi',
    10: 'PinSyl',
    11: 'PruAvi',
    12: 'PruSer',
    13: 'PseMen',
    14: 'QuePet',
    15: 'QueRob',
    16: 'QueRub',
    17: 'RobPse',
    18: 'SalCap',
    19: 'SorTor',
    20: 'TilSpe',
    21: 'TsuHet',
    22: 'clutter',
}

list_of_species_ID2NAME = [list_of_species_ID2NAME[i] for i in range(23)]
list_of_species_NAME2ID = {}
for i in range(23):
    list_of_species_NAME2ID[list_of_species_ID2NAME[i]] = i

INS_COLORS = np.array(
    [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(1000)]
)

def get_labels(scene_name, scene_data, data_dir, log_file_name):
    
    instance_pths = glob.glob(data_dir + "/" + scene_name + "/Annotations/*.txt")
    # print('instance_pths', instance_pths)

    scene_pts = scene_data[:, :3]  # Scene point cloud
    pt_tree = KDTree(scene_pts)

    error = 0
    instances = np.zeros((len(scene_data), 1), dtype=np.int32) - 1
    semantics = np.zeros((len(scene_data), 1), dtype=np.float32) - 1
    colors = np.zeros((len(scene_data), 3), dtype=np.uint64)

    # Use nearest neighbor to find corresponding point indexes in the scenes PC of instances
    for instance_id, pth in enumerate(instance_pths):
        class_name = pth.split("/")[-1].split("_")[0]
        if not (class_name in list_of_species_NAME2ID.keys()):
            class_name = "clutter"
        semantic_id = list_of_species_NAME2ID[class_name]
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
        colors[pt_indexs] = list_of_species_colors[semantic_id]
        error += dist.sum()

    decided = (instances >= 0)[:, 0]

    # For some points are not annotated, use the label from nearby points
    pt_tree = KDTree(scene_pts[decided])
    dist, decided_indexs = pt_tree.query(scene_pts[~decided], k=1)

    instances[~decided] = instances[decided][decided_indexs]
    semantics[~decided] = semantics[decided][decided_indexs]
    colors[~decided] = colors[decided][decided_indexs]

    assert (instances.min()) >= 0
    assert (semantics.min()) >= 0
    assert (colors.min()) >= 0

    # Avoiding duplicate instances -> instance ids are contiguous from 0
    remap_id = np.array(range(instances.max() + 1))
    for new_id, old_id in enumerate(np.unique(instances)):
        remap_id[old_id] = new_id
    instances = remap_id[instances].astype(np.float32)
    unique_instances = np.unique(instances)

    assert np.all(unique_instances == range(len(unique_instances)))

    return instances, semantics, colors


def read_scene_txt(name, data_dir):

    pts = np.loadtxt(os.path.join(data_dir + "/" + name, name + ".txt"))

    return pts


def preprocess_SYSSIFOSS(data_dir):
    file_lists_dir = os.path.join(data_dir, "SYSSIFOSS_v1.0_Aligned_version")

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"SYSSIFOSS_prepare_data_{timestamp}.log"

    with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
        log_line = f"-----处理项目: SYSSIFOSS数据预处理-----\n"
        log_file.write(log_line)

    scene_list = []
    
    file_lists = glob.glob(file_lists_dir + "/*")
    for file_list in file_lists:
        txt_lists = glob.glob(file_list + "/*")
        for txt_list in txt_lists:
            if '.txt' in txt_list:
                scene_list.append(file_list.split("/")[-1])
    
    scene_list = natsort.natsorted(scene_list)

    save_dir = os.path.join(data_dir, "ISBNet_preprocess")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir_list = glob.glob(save_dir + "/*")

    for scene_name in tqdm(scene_list, position=0):

        with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
            log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-----\n"
            log_file.write(log_line)

        scene_pth = os.path.join(save_dir, f"{scene_name}_inst_nostuff")
        if scene_pth in save_dir_list:
            with open('./logs/' + log_file_name, 'a', encoding='utf-8') as log_file:
                log_line = f"----{now.strftime('%Y-%m-%d %H:%M:%S')} - 处理目录: {scene_name}-aleady----\n"
                log_file.write(log_line)
        else: 
            os.makedirs(save_dir, exist_ok=True)

            scene_data = read_scene_txt(scene_name, file_lists_dir)
            instances, semantics, colors = get_labels(scene_name, scene_data, file_lists_dir, log_file_name)

            torch.save(
                (
                    scene_data[:, :3].astype(np.float32),
                    colors.reshape(-1, 3).astype(np.float32),
                    semantics.reshape(-1).astype(np.int32),
                    instances.reshape(-1).astype(np.int32),
                ),
                scene_pth,
            )


cfg = parser.parse_args()
preprocess_SYSSIFOSS(cfg.data_dir)