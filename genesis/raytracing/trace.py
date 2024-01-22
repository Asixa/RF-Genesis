
from .radar import Radar
from .pathtracer import RayTracer
from tqdm import tqdm

import torch
import numpy as np
torch.set_default_device('cuda')



def trace(filename):
    smpl_data = np.load(filename, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    max_distance = np.max(root_translation[:,2])+2
    body_offset = np.array([0,1,3])
    sensor_origin = np.array([0,0,0])
    sensor_target = np.array([0,0,-5])

    raytracer = RayTracer()
    PIRs = []
    pointclouds = []
    total_motion_frames = len(root_translation)

    for frame_idx in tqdm(range(0, total_motion_frames), desc="Rendering Body PIRs"):
        raytracer.update_pose(smpl_data['pose'][frame_idx], smpl_data['shape'][0], np.array(root_translation[frame_idx]) -  body_offset)
        PIR, pc = raytracer.trace()
        PIRs.append(torch.from_numpy(PIR).cuda())
        pointclouds.append(torch.from_numpy(pc).cuda())

    return PIRs, pointclouds

