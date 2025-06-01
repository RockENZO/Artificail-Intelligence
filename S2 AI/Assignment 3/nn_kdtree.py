import sys
import numpy as np
import pandas as pd

STUDENT_ID = "1880714"
DEGREE = "UG"

class KDNode:
    def __init__(self, point, label, d, val, left, right):
        self.point = point
        self.label = label
        self.d = d
        self.val = val
        self.left = left
        self.right = right

def build_kdtree(points, labels, D, depth=0, print_first_split=False):
    if len(points) == 0:
        return None
    if len(points) == 1:
        d_val = (D + depth) % points.shape[1]
        return KDNode(point=points[0], label=labels[0], d=d_val, val=points[0][d_val], left=None, right=None)
    
    M = points.shape[1]
    d = (D + depth) % M
    sorted_idx = np.argsort(points[:, d])
    points_sorted = points[sorted_idx]
    labels_sorted = labels[sorted_idx]
    
    median_idx = len(points_sorted) // 2
    median_val = points_sorted[median_idx][d]
    node_point = points_sorted[median_idx]
    node_label = labels_sorted[median_idx]
    node_val = node_point[d]

    # Exclude the median point itself from both subtrees
    left_indices = []
    right_indices = []
    for i in range(len(points_sorted)):
        if i == median_idx:
            continue
        if points_sorted[i][d] < median_val:
            left_indices.append(i)
        elif points_sorted[i][d] > median_val:
            right_indices.append(i)
        else:  # points_sorted[i][d] == median_val
            if int(STUDENT_ID[-1]) % 2 == 1:  # odd student ID
                left_indices.append(i)
            else:  # even student ID
                right_indices.append(i)

    left_points = points_sorted[left_indices]
    left_labels = labels_sorted[left_indices]
    right_points = points_sorted[right_indices]
    right_labels = labels_sorted[right_indices]
  
    if print_first_split and depth == 0:
        dots = '.' * D
        print(f"{dots}l{len(left_points)}")
        print(f"{dots}r{len(right_points)}")
    
    left_sub = build_kdtree(left_points, left_labels, D, depth+1)
    right_sub = build_kdtree(right_points, right_labels, D, depth+1)
    return KDNode(point=node_point, label=node_label, d=d, val=node_val, left=left_sub, right=right_sub)

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def nn_search(node, target, best=None):
    if node is None:
        return best
    dist = euclidean_dist(target, node.point)
    if best is None or dist < best[0]:
        best = (dist, node.label)
    d = node.d
    if target[d] < node.val:
        best = nn_search(node.left, target, best)
        if best is None or abs(target[d] - node.val) < best[0]:
            best = nn_search(node.right, target, best)
    else:
        best = nn_search(node.right, target, best)
        if best is None or abs(target[d] - node.val) < best[0]:
            best = nn_search(node.left, target, best)
    return best

def main():
    train_path, test_path, D = sys.argv[1], sys.argv[2], int(sys.argv[3])
    train_df = pd.read_csv(train_path, sep=r'\s+', engine='python')
    test_df = pd.read_csv(test_path, sep=r'\s+', engine='python')
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.values
    tree = build_kdtree(X_train, y_train, D, print_first_split=True)
    for x in X_test:
        _, label = nn_search(tree, x)
        print(int(label))

if __name__ == "__main__":
    main()