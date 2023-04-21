import numpy as np
import torch
import torch.nn as nn
from lbs import lbs, batch_rodrigues
from utils import to_tensor
import pickle


class SMPL:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')

        self.J_regressor = to_tensor(model_data['J_regressor'])
        self.weights = to_tensor(model_data['weights'])
        self.posedirs = to_tensor(model_data['posedirs'])
        self.v_template = to_tensor(model_data['v_template'])
        self.shapedirs = to_tensor(model_data['shapedirs'])

    def __call__(self, betas, pose):
        full_pose = torch.cat([pose[:, :3], torch.zeros(pose.shape[0], 1, device=pose.device), pose[:, 3:]], dim=1)
        v_shaped = self.v_template + torch.matmul(self.shapedirs, betas.unsqueeze(-1)).squeeze()
        J = torch.matmul(self.J_regressor, v_shaped)
        pose_cube = batch_rodrigues(full_pose.view(-1, 3)).view(-1, 24, 3, 3)
        Rs = pose_cube[:, 1:] - pose_cube[:, :-1]
        poseRs = Rs.contiguous().view(Rs.shape[0], -1)
        rot_matrices = torch.cat([torch.eye(3, device=pose.device).unsqueeze(0), poseRs], dim=1)
        rot_matrices = torch.cumsum(rot_matrices, dim=1)
        joint_rot_matrices = torch.matmul(rot_matrices.view(-1, 3, 3), self.J_regressor.unsqueeze(-1)).view(-1, 24, 3)
        parent_rot_matrices = joint_rot_matrices.clone()
        parent_rot_matrices[:, 1:] = parent_rot_matrices[:, self.parent_ids[1:]]
        rest_T = torch.cat([joint_rot_matrices, J.unsqueeze(-1)], dim=2)
        zeros = torch.zeros((rest_T.shape[0], 4, 4), device=rest_T.device)
        rest_T = torch.cat([rest_T, zeros], dim=1)
        rest_T[:, 3, 3] = 1.0
        rest_T[:, 1:, 3] += J.unsqueeze(-1)
        results = []
        for i in range(rot_matrices.shape[0]):
            T = rest_T[i]
            for j in range(1, self.kintree_table.shape[1]):
                parent = self.parent_ids[j]
                Tj = torch.matmul(parent_rot_matrices[i, parent], torch.inverse(parent_rot_matrices[i, j]))
                T[j] = torch.matmul(Tj, T[j])
                T[j, :, 3] += torch.matmul(Tj[:, :, :3], self.offsets[j].unsqueeze(-1)).squeeze()
            results.append(torch.matmul(T[:24], v_shaped[i].unsqueeze(-1)).squeeze() + T[24:, :3, 3])
        results = torch.stack(results)
        return results

