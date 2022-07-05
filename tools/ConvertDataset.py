#!/usr/bin/env python
# coding: utf-8

# In[10]:


import open3d as o3d
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from glob import glob
import os
import tqdm
import multiprocessing
from joblib import Parallel, delayed

# In[11]:


def rgbd_to_pointcloud(rgb_path, depth_path, label_path):

    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    mat_labels = scipy.io.loadmat(label_path)

    labels = mat_labels['gt_label']
    color_np_image = np.array(color_raw)
    depth_np_image = np.array(depth_raw)
    shape = color_np_image.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if labels[i][j] == 0:
                depth_np_image[i][j] = 0
            color_np_image[i][j] = (labels[i][j],labels[i][j],labels[i][j])

    color_raw = o3d.geometry.Image(color_np_image)
    depth_raw = o3d.geometry.Image(depth_np_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    return pcd


# In[14]:


def save_pcd_to_file(pcd, output_dataset_folder, class_label, class_id, object_id):
    
    class_folder_path = f'{output_dataset_folder}/{class_label}'
    if not os.path.exists(class_folder_path):
        os.makedirs(class_folder_path)
        os.makedirs(f'{class_folder_path}/points')
        os.makedirs(f'{class_folder_path}/points_label')
        
    with open(f'{class_folder_path}/points/{class_id}{object_id}.pts', "w") as file_points:
        with open(f'{class_folder_path}/points_label/{class_id}{object_id}.seg', "w") as file_labels:
            for point, label in zip(pcd.points,pcd.colors):
                label = round(label[0]*255)
                file_points.write("{} {} {}\n".format(point[0],point[1],point[2]))
                file_labels.write("{}\n".format(label))


# In[ ]:





# In[15]:


output_dataset_folder = "new_dataset"
if os.path.exists(output_dataset_folder):
     os.rmdir(output_dataset_folder)
os.makedirs(output_dataset_folder)

def worker_f(path):
    for label_path in tqdm.tqdm(glob(f"{path}/*label.mat"), leave=False):
        s = label_path.split('/')[4].split('_')
        class_label = s[0]
        class_id = s[1]
        object_id = s[2]
        depth_path = label_path[:-9]+'depth.png'
        rgb_path = label_path[:-9]+'rgb.jpg'
        pcd = rgbd_to_pointcloud(rgb_path, depth_path, label_path)
        save_pcd_to_file(pcd, output_dataset_folder, class_label, class_id, object_id)

num_cores = multiprocessing.cpu_count()
inputs = tqdm.tqdm(glob("dataset/part-affordance-dataset/tools/*"))
Parallel(n_jobs=num_cores)(delayed(worker_f)(i) for i in inputs)

    


# In[ ]:




