import numpy as np
import mitsuba as mi
import drjit as dr
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import numpy as np
torch.set_default_device('cuda')



def get_smpl_layer():
    return SMPL_Layer(center_idx=0,gender='male',model_root='../models/smpl_models').cuda()

def apply_translation(vertices, translation_matrix):
    ones = torch.ones(vertices.shape[0], 1)
    homogeneous_vertices = torch.cat([vertices, ones], dim=1)
    transformed_vertices = torch.matmul(homogeneous_vertices, translation_matrix.T)  # We don't transpose the matrix here because it is already transposed by mitsuba
    transformed_vertices_3d = transformed_vertices[:, :3]
    return transformed_vertices_3d


def call_smpl_layer(pose_params, shape_params,body, need_face = False,transform = None): # Torch Tensor pose_params: (Batch, 72), shape_params: (Batch, 10)


    vertices, Jtr = body(pose_params, th_betas=shape_params) # (Batch, 6890, 3), (Batch, 24, 3)

    # remove the batch dimension
    if len(vertices.shape)==3:
        if vertices.shape[0]==1:
            vertices=vertices[0]
        else:
            raise NotImplementedError("mesh batch is supported, yet")

    if transform is not None:
        vertices = apply_translation(vertices, transform)

    # convert torch to drjit
    vertices_mi=mi.TensorXf(vertices.cpu().numpy())
    
    if not need_face:
        return vertices_mi
    if need_face:
        faces_mi=mi.TensorXf(body.th_faces.cpu().numpy())
        return vertices_mi, faces_mi



