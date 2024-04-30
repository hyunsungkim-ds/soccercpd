from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, distance_matrix


# apply Delaunay triangulation to the given player coordinates to obtain the role-adjacency matrix
def delaunay_edge_mat(coords):
    tri_pts = Delaunay(coords).simplices
    edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
    edge_mat = np.zeros((coords.shape[0], coords.shape[0]))
    edge_mat[edges[:, 0], edges[:, 1]] = 1
    return np.clip(edge_mat + edge_mat.T, 0, 1)


# Hamming distance between two permutations of the same shape
def hamming_dist(perm1, perm2):
    return (perm1 != perm2).astype(int).sum()


# Manhattan distance between two matrices of the same shape
def manhattan_dist(mat1, mat2):
    return np.abs(mat1 - mat2).sum()


def most_common(player_roles):
    try:
        counter = Counter(player_roles[player_roles.notna()])
        return counter.most_common(1)[0][0]
    except IndexError:
        return np.nan


def compute_delaunay_dists(form1: pd.Series, form2: pd.Series):
    cost_mat = distance_matrix(form1["coords"], form2["coords"])
    _, perm = linear_sum_assignment(cost_mat)
    edge_mat1 = form1["edge_mat"]
    edge_mat2 = form2["edge_mat"][perm][:, perm]
    return np.abs(edge_mat1 - edge_mat2).sum()


def seconds_to_time_str(x: float):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes:02d}:{seconds:02d}"
