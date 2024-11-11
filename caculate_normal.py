import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def get_normal_map_by_point_cloud(depth, fov):
    height, width = depth.shape

    fov_radians = np.deg2rad(fov)
    height, width = depth_map.shape
    focal_length = width / (2 * np.tan(fov_radians / 2))
    fx = focal_length  
    fy = focal_length 
    cx = (width - 1) / 2.0  
    cy = (height - 1) / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def normalization(data):
        mo_chang = np.sqrt(
            np.multiply(data[:, :, 0], data[:, :, 0])
            + np.multiply(data[:, :, 1], data[:, :, 1])
            + np.multiply(data[:, :, 2], data[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return data / mo_chang

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))
    pts_3d = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, height, width))
    f = (
        pts_3d_world[:, 1 : height - 1, 2 : width]
        - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
    )
    t = (
        pts_3d_world[:, 2:height, 1 : width - 1]
        - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
    )
    normal_map = -np.cross(f, t, axisa=0, axisb=0)
    normal_map = normalization(normal_map)
    new_normal_nap = np.zeros((height, width, 3))
    new_normal_nap[1:height-1, 1:width-1, :] = normal_map
    return new_normal_nap



def depthToPoint(fov, depth):
    height, width = depth.shape
    fov_radians = np.deg2rad(fov)

    focal_length = width / (2 * np.tan(fov_radians / 2))
    fx = focal_length  
    fy = focal_length 
    cx = (width - 1) / 2.0  
    cy = (height - 1) / 2.0  

    x, y = np.meshgrid(range(width), range(height))
    z = depth 
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    x_3d = x_3d.astype(np.float32)
    y_3d = y_3d.astype(np.float32)


    point_cloud = np.stack((x_3d, y_3d, z), axis=-1)

    return point_cloud

def process_normal(normal):
    normal = normal * 2.0 - 1.0
    normal = normal[:,:,np.newaxis,:]
    normalizer = np.sqrt(normal @ normal.transpose(0,1,3,2))
    normalizer = np.squeeze(normalizer, axis=-2)
    normalizer = np.clip(normalizer, 1.0e-20, 1.0e10)
    normal = np.squeeze(normal, axis=-2)
    normal = normal / normalizer 
    return normal

save_vis = False
save_normal = True
padding = False
folder = "/home/disk2/dataset/albedo"

depth_folder = os.path.join(folder, "depth")
for filename in os.listdir(depth_folder):
    filename_1 = filename.split(".")[0]
    depth_path = os.path.join(depth_folder, filename)
    try:
        depth_map = np.load(depth_path)
    except FileNotFoundError:
        continue

    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = cv2.GaussianBlur(depth_map, (5,5), 0.0)
    depth_map = cv2.medianBlur(depth_map, 5)

    fov = 60
    pos_image_data = depthToPoint(fov, depth_map)
    normal_image_data = get_normal_map_by_point_cloud(depth_map, fov)
    normal_image_data = cv2.GaussianBlur(normal_image_data, (5,5), 0.0).astype(np.float32)

    pos_image_data[:,:,2] = -pos_image_data[:,:,2]
    normal_image_data[:,:,2] = -normal_image_data[:,:,2]
    normal_image_data = (normal_image_data + 1) * 0.5
    pos_image_data = pos_image_data[:,:,(2,1,0)]
    normal_image_data = normal_image_data[:,:,(2,1,0)]

    # save normal npy
    if save_normal:
        normal_image_data_npy = np.transpose(normal_image_data, (2, 0, 1))
        normal_path = os.path.join(folder, "normal", filename)
        np.save(normal_path, normal_image_data_npy)

    if save_vis:
        from PIL import Image
        depth_vis_path = os.path.join(folder, "depth_vis", filename_1 + '.png')
        normal_vis_path = os.path.join(folder, "normal_vis", filename_1 + '.png')
        plt.imsave(depth_vis_path, depth_map, cmap='viridis')
        scale_factor = 255
        normal_image_data= (normal_image_data * scale_factor).astype(np.uint8)
        img = Image.fromarray(normal_image_data)
        img.save(normal_vis_path)

    print(f"{filename} finish\n")