

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2
from tqdm import tqdm

from genesis.raytracing.radar import Radar 
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud,rangeFFT,dopplerFFT,process_pc
from smplpytorch.pytorch.smpl_layer import SMPL_Layer



# SMPL 
def display_smpl(
        model_info,
        model_faces=None,
        with_joints=False,
        kintree_table=None,
        ax=None,
        batch_idx=0,
        translation=None,
        ):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = model_info['verts'][batch_idx], model_info['joints'][
        batch_idx]
    if translation is not None:
        verts += translation
        joints += translation
    
    if model_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
    else:
        mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if with_joints:
        draw_skeleton(joints, kintree_table=kintree_table, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_zlim(-1, 3)
    ax.view_init(azim=-90, elev=100)
    ax.view_init(azim=30, elev=30, roll = 105)
    ax.set_title('SMPL model', fontsize=20)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return ax


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=False):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    colors = []
    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
    for c in kintree_colors:
        colors += left_right_mid[c]
    # For each 24 joint
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                [joints3D[j1, 1], joints3D[j2, 1]],
                [joints3D[j1, 2], joints3D[j2, 2]],
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_numbers:
            ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax



def draw_smpl_on_axis(pose,shape,translation=None, ax=None):
    pose = torch.tensor(pose).unsqueeze(0)
    shape = torch.tensor(shape).unsqueeze(0)
    smpl_layer = SMPL_Layer(center_idx=0,gender='male',model_root='models/smpl_models')
    verts, Jtr = smpl_layer(pose, th_betas=shape)

    display_smpl(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,translation = translation, ax = ax)
    

# Plotting Pointclouds
def draw_poinclouds_on_axis(pc,ax, tx,rx,elev,azim,title):
    pc = np.transpose(pc)
    ax.scatter(-pc[0], pc[1], pc[2], c=pc[4], cmap=plt.hot())
    if tx is not None:
        ax.scatter(tx[:,0], tx[:,2], tx[:,1], c="green", s= 50, marker =',', cmap=plt.hot())
    if rx is not None:
        ax.scatter(rx[:,0], rx[:,2], rx[:,1], c="orange", s= 50, marker =',', cmap=plt.hot())
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0, 6)
    ax.set_zlim(-0.5, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=20)

def draw_doppler_on_axis(radar_frame,pointcloud_cfg, ax):
    range_fft = rangeFFT(radar_frame,pointcloud_cfg.frameConfig)
    doppler_fft = dopplerFFT(range_fft,pointcloud_cfg.frameConfig)
    dopplerResultSumAllAntenna = np.sum(doppler_fft, axis=(0,1))
    ax.imshow(np.abs(dopplerResultSumAllAntenna))
    ax.set_title("Doppler FFT", fontsize=20)

def draw_combined(i,pointcloud_cfg,radar_frames,pointclouds,smpl_data):
    smpl_frame_id = i               # 30FPS
    radar_frame_id = int(i/3)       # 10FPS

    poses = smpl_data["pose"]
    shape = smpl_data['shape']
    root_translation = smpl_data['root_translation']


    fig= plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    draw_smpl_on_axis(poses[smpl_frame_id],shape,root_translation[smpl_frame_id],ax1)


    ax2 = fig.add_subplot(132, projection='3d')
    draw_poinclouds_on_axis(pointclouds[radar_frame_id],ax2, None,None,30,-30,"Point clouds")


    ax3 = fig.add_subplot(133)
    draw_doppler_on_axis(radar_frames[radar_frame_id],pointcloud_cfg, ax3)


    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) 
    return data


def save_video(radar_cfg_file, radar_frames_file, smpl_data_file, output_file):
    radar = Radar(radar_cfg_file)
    pointcloud_cfg = PointCloudProcessCFG(radar)
    radar_frames = np.load(radar_frames_file)
    smpl_data = np.load(smpl_data_file,allow_pickle=True)

    # Process the pointclouds
    pointclouds = []
    for frame in radar_frames:
        pc = process_pc(pointcloud_cfg, frame)
        pointclouds.append(pc)
    
    # Write the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = output_file
    out = cv2.VideoWriter(video_filename, fourcc, 30, (1200, 600))
    for i in tqdm(range(smpl_data["pose"].shape[0]-2)):
        frame = draw_combined(i,pointcloud_cfg,radar_frames,pointclouds,smpl_data)
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_data)
    out.release()