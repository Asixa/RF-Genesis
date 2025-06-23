from tqdm import tqdm

import torch
import numpy as np
from .radar import Radar
from PIL import Image
import math
torch.set_default_device('cuda')

def calculate_environment_points(environment_pir):
    """
    environment_pir: (H, W, 3) torch tensor, assumed to be on the correct device (e.g., CUDA)
    Returns: (H*W, 3) point cloud tensor in camera space
    """
    H, W, _ = environment_pir.shape
    device = environment_pir.device

    distance = environment_pir[:, :, 0] * 5 + 5  # [H, W]

    fov_rad = math.radians(60)
    fx = W / (2 * math.tan(fov_rad / 2))
    fy = fx
    cx = W / 2
    cy = H / 2

    j = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)  # rows
    i = torch.arange(0, W, device=device).view(1, -1).expand(H, W)  # cols

    x = (i - cx) / fx  # [H, W]
    y = (j - cy) / fy
    z = torch.ones_like(x, device=device)

    xyz = torch.stack((x, y, z), dim=-1) * distance.unsqueeze(-1)  # [H, W, 3]
    points = xyz.reshape(-1, 3)  # [H*W, 3]
    return points

def create_interpolator(_frames, _pointclouds, environment_pir, frame_rate=30, remove_zeros = True):
    num_frames = len(_frames)   
    total_time = num_frames / frame_rate
    frames = _frames.copy()
    pointclouds = _pointclouds.copy()

    if environment_pir != None:
        # redue the size of environment PIR to reduce the memory usages
        environment_pir = environment_pir.resize((64, 64), resample=Image.Resampling.BILINEAR)
        environment_pir = torch.tensor(np.array(environment_pir),dtype=torch.float32)/255.0
        environment_points = calculate_environment_points(environment_pir)
        environment_intensity = environment_pir[:,:,1].flatten()
    def interpolator(time):
            if time < 0 or time > total_time:
                raise ValueError("Invalid time value")
            
            frame_index = int(time * frame_rate)
            if frame_index == num_frames:
                return frames[-1]
            
            t = (time * frame_rate) % 1 # fractional part of time
            frame1 = frames[frame_index]
            frame2 = frames[frame_index + 1]

            pointcloud1 = pointclouds[frame_index]
            pointcloud2 = pointclouds[frame_index + 1]



            zero_depth_frame1 = frame1[:,:, 1] == 0  # zero depth pixels
            zero_depth_frame2 = frame2[:,:, 1] == 0

            zero_depth_frame1_flat = zero_depth_frame1.reshape(-1)
            zero_depth_frame2_flat = zero_depth_frame2.reshape(-1)


            frame1[zero_depth_frame1] = frame2[zero_depth_frame1] # replace zero depth pixels with the other frame
            frame2[zero_depth_frame2] = frame1[zero_depth_frame2]

            pointcloud1[zero_depth_frame1_flat] = pointcloud2[zero_depth_frame1_flat] # replace zero depth pixels with the other frame
            pointcloud2[zero_depth_frame2_flat] = pointcloud1[zero_depth_frame2_flat]


            interpolated_frame = frame1 * (1 - t) + frame2 * t
            interpolated_pointcloud = pointcloud1 * (1 - t) + pointcloud2 * t

            flatten_pir  = interpolated_frame.reshape(-1, 3)

            intensity = flatten_pir[:,0]
            depth = flatten_pir[:,1]
            
            mask = (depth > 0.1) & (intensity > 0.1)

            if environment_pir != None:
                combined_intensity = torch.cat((environment_intensity, intensity[mask]), dim=0)
                combined_pointcloud = torch.cat((environment_points, interpolated_pointcloud[mask]), dim=0)
            else:
                combined_intensity = intensity[mask]
                combined_pointcloud = interpolated_pointcloud[mask]
            # return flatten_pir[:,1], interpolated_pointcloud[mask]
            return combined_intensity, combined_pointcloud
        
    
    return interpolator




def generate_signal_frames(body_pirs,body_auxs,envir_pir, radar_config):
    interpolator = create_interpolator(body_pirs,body_auxs,envir_pir, frame_rate=30)
    total_motion_frames = len(body_pirs)

    radar = Radar(radar_config)

    total_radar_frame = int(total_motion_frames / 30 * radar.frame_per_second)
    frames = []
    for i in tqdm(range(total_radar_frame), desc="Generating radar frames"):
        frame_mimo = radar.frameMIMO(interpolator,i*1.0/radar.frame_per_second)
        frames.append(frame_mimo.cpu().numpy())
    frames = np.array(frames)
    return frames