import numpy as np
import torch

def batch_rodrigues(theta):
    """
    Calculates rotation matrices from rotation vectors using the Rodrigues formula.

    Args:
        theta: A tensor of shape (N, 3) representing the rotation vectors.

    Returns:
        A tensor of shape (N, 3, 3) representing the rotation matrices.
    """
    l2norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l2norm, -1)
    normalized = torch.div(theta, angle)

    angle = angle * (np.pi / 180.0)

    axis = normalized
    sina = torch.sin(angle)
    cosa = torch.cos(angle)

    # Rodrigues formula for rotation matrices
    outer = torch.matmul(torch.unsqueeze(axis, 2), torch.unsqueeze(axis, 1))
    R = (torch.eye(3).unsqueeze(0).repeat(theta.size(0), 1, 1) * cosa.view(-1, 1, 1)) + \
        ((1 - cosa.view(-1, 1, 1)) * outer) + \
        (torch.cross(axis, torch.eye(3).unsqueeze(0).repeat(theta.size(0), 1, 1), dim=1) * sina.view(-1, 1, 1))

    return R

def lbs(vertices, pose, shape, J_regressor, parent, lbs_weights):
    num_joints = J_regressor.shape[0]

    # Get the 3D joint locations using pose and shape parameters
    pose = pose.reshape((-1, 3))
    shape = shape.reshape((-1, 1))
    vertices = vertices + torch.matmul(lbs_weights, shape).reshape((-1, 3))
    J = torch.matmul(J_regressor, vertices)

    # Calculate the global joint locations using the parent-child relationships
    parent_indices = parent.cpu().numpy()
    parent_indices[0] = -1
    T = np.zeros((num_joints, 4, 4))
    for j in range(num_joints):
        R = batch_rodrigues(pose[j:j+1]).view((3, 3))
        if parent_indices[j] == -1:
            T[j, :3, :3] = R
            T[j, :3, 3] = J[j]
        else:
            T[j, :3, :3] = R
            T[j, :3, 3] = J[j] - J[parent_indices[j]]
            T[j] = torch.matmul(T[parent_indices[j]], T[j])

    # Calculate the transformed joint locations and the final 3D vertex positions
    T[:, :3, 3] = T[:, :3, 3] + J.unsqueeze(2).repeat(1, 1, 4)[:, :3, 3]
    A = torch.matmul(lbs_weights, T.reshape((num_joints, 16))[:, :3]).reshape((-1, 4, 4))
    vertices = torch.matmul(A, vertices.unsqueeze(2)).squeeze()

    return vertices
