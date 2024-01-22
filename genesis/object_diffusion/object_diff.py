import sys
import os
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
from termcolor import colored
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from transforms3d.axangles import axangle2mat
import sys
sys.path.append("ext/mdm/")
from visualize.vis_utils import joints2smpl,npy2obj
from model.rotation2xyz import Rotation2xyz
import utils.rotation_conversions as geometry

def euler_to_axis_angle(euler_angles):
    """ Converts a set of Euler angles to axis-angle representation."""
    axis_angle_params = np.zeros_like(euler_angles)

    for i in range(euler_angles.shape[0]):
        for j in range(euler_angles.shape[1]):
            euler = euler_angles[i, j]
            r = R.from_euler('xyz', euler)
            axis_angle = r.as_rotvec()
            axis_angle_params[i, j] = axis_angle

    return axis_angle_params

def process(out_dir):
    filename = out_dir+"/obj_diff_raw.npy"
    print(colored("---[RFGen.ObjDiff]:Runing SMPLify, it may take a few minutes.---", 'yellow'))
    print(colored("---[RFGen.ObjDiff]:This may be optimized in future updates.---", 'yellow'))
    data = np.load(filename,allow_pickle=True)
    motion = data[None][0]['motion'].transpose(0,3, 1, 2)
    
    num_frames = motion.shape[1]
    device='0'
    cuda=True
    
    os.chdir("ext/mdm")
    j2s = joints2smpl(num_frames=num_frames, device_id=device, cuda=cuda)
    os.chdir("../..")
    
    motion_tensor, opt_dict = j2s.joint2smpl(motion[0]) 
    thetas = motion_tensor[0, :-1, :, :num_frames]   
                                                # So basicly this would be the posture of SMPL, 
                                                # it is rot6d, but you can convert it to rotation matrix
                                                # see rotation2xyz
    root_translation = motion_tensor[0, -1, :3, :].cpu().numpy().transpose(1,0)


    thetas_matrix = thetas.transpose(2, 0).transpose(1, 2)
    thetas_matrix = geometry.rotation_6d_to_matrix(thetas_matrix)
    thetas_vec3 = geometry.matrix_to_euler_angles(thetas_matrix,"XYZ")
    thetas_vec3 = thetas_vec3.cpu().numpy()
    final_thetas = euler_to_axis_angle(thetas_vec3)
    smpl_params = final_thetas.reshape(final_thetas.shape[0], -1)
    
    shape_params =np.zeros(10) 
    np.savez(out_dir+'/obj_diff.npz',pose=smpl_params,shape=shape_params, root_translation = root_translation,gender="male")
    


def generate(prompt, out_dir):

    os.chdir("ext/mdm/")
    subprocess.run(
        ['python', '-m', 'sample.generate_rfgen', '--model_path', './save/humanml_trans_enc_512/model000200000.pt', 
         '--text_prompt', prompt, 
         '--output_dir', "../../"+out_dir, 
         '--num_samples', '1', '--num_repetitions', '1'])
    os.chdir("../..")
    process(out_dir)
    

