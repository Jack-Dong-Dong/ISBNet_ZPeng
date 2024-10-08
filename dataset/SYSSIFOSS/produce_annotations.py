import open3d as o3d
import numpy as np
import os
import glob
import laspy
from tqdm import tqdm

root_path = '/home/pengzhen/code/pointcloud_dataset_set/dataset/SYSSIFOSS/SYSSIFOSS_v1.0_Aligned_version'
file_lists = glob.glob(os.path.join(root_path + '/*'))

list_of_species = {
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
}

def megre_data_from_single_trees(data_list):
    """

    :param data_list: 
    """
    # 结果文件存储
    annotationDir = os.path.join(data_list, 'Annotations')
    os.makedirs(annotationDir, exist_ok=True)
    # 读取单树数据,值读取ULS、TLS文件
    data_list = os.path.join(data_list, 'single_trees')
    single_trees_lists = glob.glob(os.path.join(data_list + '/*'))
    for single_tree in tqdm(single_trees_lists, position=0, leave=True):
        [SPECIESID, PLOTID, TREEID] = single_tree.split('/')[-1].split('_')
        # print(SPECIESID, PLOTID, TREEID)
        single_tree_files = glob.glob(os.path.join(single_tree + '/*.laz'))
        all_xyz = []
        for single_tree_laz in single_tree_files:
            if 'ULS' in single_tree_laz or 'TLS' in single_tree_laz:
                # print(single_tree_laz)
                with laspy.open(single_tree_laz) as laz_file:
                    laz = laz_file.read()
                    xyz = np.vstack((laz.x, laz.y, laz.z)).transpose()
                    all_xyz.append(xyz)
        if all_xyz:
            combined_xyz = np.vstack(all_xyz)
            combined_xyz = np.around(combined_xyz, decimals=6)
            # print(xyz)
            classification = list_of_species[SPECIESID]
            data_with_classification = np.hstack((combined_xyz, np.full((combined_xyz.shape[0], 1), classification)))
            
            output_file_path = os.path.join(annotationDir, f'{SPECIESID}_{PLOTID}_{TREEID}.txt')
            np.savetxt(output_file_path, data_with_classification, fmt='%.6f %.6f %.6f %d')
            # x y z classification Individual


def megre_data_from_annotations(data_list):
    """

    :param data_list: 
    """
    annotationDir = os.path.join(data_list, 'Annotations')
    annotation_trees_lists = glob.glob(os.path.join(annotationDir + '/*'))
    PLOTNAME = data_list.split('/')[-1]
    all_xyz = []
    for annotation_tree in tqdm(annotation_trees_lists, position=0, leave=True):
        data = np.loadtxt(annotation_tree)
        xyz = data[:, :3]
        all_xyz.append(xyz)

    if all_xyz:
        combined_xyz = np.vstack(all_xyz)
        combined_xyz = np.around(combined_xyz, decimals=6)

        output_file_path = os.path.join(data_list, f'{PLOTNAME}.txt')
        np.savetxt(output_file_path, combined_xyz, fmt='%.6f %.6f %.6f')



if __name__ == "__main__":
    for file_list in file_lists:
        if 'batchData' not in file_list:
            data_lists = glob.glob(os.path.join(file_list + '/*'))
            for data_list in data_lists:
                if 'single_trees' in data_list:
                    print(file_list)
                    megre_data_from_single_trees(file_list)

    for file_list in file_lists:
        if 'batchData' not in file_list:
            data_lists = glob.glob(os.path.join(file_list + '/*'))
            for data_list in data_lists:
                if 'single_trees' in data_list:
                    print(file_list)
                    megre_data_from_annotations(file_list)