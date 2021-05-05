# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn.functional as F


# PyTorch-backed implementations

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.contiguous().view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def qinv(q: torch.Tensor):
    """
    Invert quaternions
    Expect a tensor of shape (*, 4)
    Returns a tensor of shape (*, 4)
    """
    ori_shape = q.shape
    q = q.contiguous().view(-1, 4)
    q_norm = torch.norm(q, p=2, dim=1)
    q_inv = q.clone().contiguous().view(-1, 4)
    q_inv[:, 1:] *= -1
    q_inv /= q_norm.unsqueeze(1)
    return q_inv.reshape(*ori_shape)

def qinv_np(q):
    ori_shape = q.shape
    q = q.reshape(-1, 4)
    q_norm = np.linalg.norm(q, ord=2, axis=1)
    q_inv = np.copy(q).reshape(-1, 4)
    q_inv[:, 1:] *= -1
    q_inv /= np.expand_dims(q_norm, axis=1)
    return q_inv.reshape(*ori_shape)

def rotation_from_to(v_from: torch.Tensor, v_to: torch.Tensor):
    """
    Calculate the shortest rotation from two vectors
    Argument:
     -- Both v_from, v_to is of shape (*, 3)
    """
    assert v_from.shape == v_to.shape
    assert v_from.shape[-1] == 3 and v_to.shape[-1] == 3

    ori_shape = v_from.shape
    out_shape = list(ori_shape)[:-1] + [4]
    v_from = v_from.view(-1, 3)
    v_to = v_to.view(-1, 3)
    dot = torch.sum(v_from * v_to, dim=1)
    xyz = torch.cross(v_from, v_to)
    w = torch.norm(v_from, dim=1, p=2) * torch.norm(v_to, dim=1, p=2) + dot
    w = w.unsqueeze(1)
    rotations = torch.cat([w, xyz], dim=1)
    rotations = F.normalize(rotations, p=2, dim=1)
    return rotations.view(out_shape)


def geodesic_distance(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """
    calculate geodesic distance from v_from to v_to in the form of delta cosine value
    :param v_from: (*, 4)
    :param v_to: (*, 4)
    :return: (*)
    """
    assert v_from.shape == v_to.shape
    assert v_from.shape[-1] == v_to.shape[-1] == 4 or v_from.shape[-1] == v_to.shape[-1] == 3

    ori_shape = v_from.shape
    v_from = v_from.contiguous().view(-1, v_from.shape[-1])
    v_to = v_to.contiguous().view(-1, v_to.shape[-1])

    if ori_shape[-1] == 4:
        v_from_inv = qinv(v_from)
        terms = torch.bmm(v_from_inv.view(-1, 4, 1), v_to.view(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        w = torch.clamp(w, -1, 1)
        distance = torch.sub(1, w).view(ori_shape[:-1])
    else:
        v_from_inv = v_from.clone()
        v_from_inv[:, 1:] *= -1
        terms = v_from_inv * v_to
        w = terms[:, 0] - terms[:, 1] - terms[:, 2]
        w = torch.clamp(w, -1, 1)
        distance = torch.sub(1, w).view(ori_shape[:-1])

    return distance


# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise Exception("Unknown axis in order")
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)


def rotation_from_to_np(v_from, v_to, use_gpu=False):
    if use_gpu:
        v_from = torch.from_numpy(v_from).cuda()
        v_to = torch.from_numpy(v_to).cuda()
        return rotation_from_to(v_from, v_to).cpu().numpy()
    else:
        v_from = torch.from_numpy(v_from)
        v_to = torch.from_numpy(v_to)
        return rotation_from_to(v_from, v_to).numpy()

def average_quaternion(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    calculate the average of quaternions
    :param q: (W, J, 4)
    :param w: (W)
    :return: (J, 4)
    """
    w = w.view(-1, 1, 1, 1)
    joint_num = q.shape[1]
    q = q.reshape(-1, 4, 1)
    q_t = q.reshape(-1, 1, 4)
    qq_t = torch.bmm(q, q_t)
    qq_t = qq_t.reshape(w.shape[0], joint_num, 4, 4)
    averages = []
    m = torch.sum(w * qq_t, dim=0)
    for jm in m:
        eigenvalues, eigenvectors = torch.eig(jm, eigenvectors=True)
        q_avg = eigenvectors[:, torch.argmax(eigenvalues[:, 0])]
        averages.append(q_avg)
    return torch.stack(averages)

def average_quaternion_np(q, w = None, use_gpu = False):
    """
    q: (W, 4)
    w: (w)
    """
    if w is None:
        w = np.ones(len(q), dtype = np.float32) / len(q)
    
    if use_gpu:
        q_torch = torch.from_numpy(q).cuda().unsqueeze(1)
        w_torch = torch.from_numpy(w).cuda()
        avg = average_quaternion(q_torch, w_torch)
        avg = avg.cpu().numpy()
    else:
        q_torch = torch.from_numpy(q).unsqueeze(1)
        w_torch = torch.from_numpy(w)
        avg = average_quaternion(q_torch, w_torch)
        avg = avg.numpy()

    return avg