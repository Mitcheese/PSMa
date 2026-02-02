import os
import torch
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm


def read_ply_to_point_cloud(file_path):
    ply_data = PlyData.read(file_path)
    vertices = ply_data['vertex']
    point_cloud = np.stack([vertices[name] for name in ['x', 'y', 'z', 'nx', 'ny', 'nz', 'iface', 'hbond', 'hphob']], axis=1)
    return point_cloud

def normalize_point_cloud(point_cloud):
    centroid = np.mean(point_cloud[:, :3], axis=0)
    centered_point_cloud = point_cloud.copy()
    centered_point_cloud[:, :3] -= centroid
    max_distance = np.max(np.abs(centered_point_cloud[:, :3]))
    if max_distance > 0:
        centered_point_cloud[:, :3] /= max_distance
    return centered_point_cloud

def farthest_point_sampling(points, num_samples):
    """
    Select points using farthest point sampling (FPS).
    :param points: numpy array of points (N x 3 for x, y, z coordinates).
    :param num_samples: number of samples to select.
    :return: sampled points (num_samples x 3).
    """
    num_points = points.shape[0]
    distances = np.ones(num_points) * np.inf
    farthest_points = np.zeros((num_samples, 3))
    farthest = np.random.randint(len(points))
    for i in range(num_samples):
        farthest_points[i] = points[farthest]
        centroid = points[farthest]
        dist = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return farthest_points

def nearest_neighbors(farthest_points, all_points, k):
    """
    Find k-nearest neighbors for each point in farthest_points.
    :param farthest_points: numpy array of selected points (N x 3).
    :param all_points: original numpy array of all points (M x 9).
    :param k: number of nearest neighbors to find.
    :return: indices of k-nearest neighbors for each point in farthest_points.
    """
    tree = cKDTree(all_points[:, :3])  # considering only x, y, z for neighbors
    _, indices = tree.query(farthest_points, k=k)
    return indices

def create_feature_vector(all_points, num_samples, k):
    """
    Create feature vector of shape (N x K x 9) using FPS and KNN.
    :param all_points: numpy array of all points (M x 9).
    :param num_samples: number of samples for FPS.
    :param k: number of neighbors for KNN.
    :return: feature vector of shape (N x K x 9).
    """
    # Apply FPS
    fps_points = farthest_point_sampling(all_points[:, :3], num_samples)

    # Find KNN for each point in FPS
    knn_indices = nearest_neighbors(fps_points, all_points, k)

    # Gather the features of K neighbors for each point
    feature_vector = all_points[knn_indices].reshape(num_samples, k, -1)

    return feature_vector

def sort_by_distance(groups):
    """
    Sort the points in each group by their distance from the origin (0, 0, 0).
    :param groups: numpy array of shape (N x K x 9) representing groups of points.
    :return: numpy array with sorted groups.
    """
    sorted_groups = np.zeros(groups.shape)
    for i, group in enumerate(groups):
        # Calculate distances from the origin
        distances = np.linalg.norm(group[:, :3], axis=1)
        sorted_indices = np.argsort(distances)
        sorted_groups[i] = group[sorted_indices]
    return sorted_groups

def Ply2Embedding(file_path, num_samples=128, k=64):
    if file_path.endswith('.ply'):
        point_cloud = read_ply_to_point_cloud(file_path)
        normalized_point_cloud = normalize_point_cloud(point_cloud)
        feature_vector = create_feature_vector(normalized_point_cloud, num_samples, k)
        sorted_feature_vector = sort_by_distance(feature_vector)

        return sorted_feature_vector
    else:
        raise TypeError('Not a Ply file.')

def farthest_point_sampling_torch(points, num_samples):
    num_points = points.shape[0]
    distances = torch.full((num_points,), float('inf'), device=points.device)
    farthest_points_indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    farthest = torch.randint(0, num_points, (1,), dtype=torch.long).item()
    for i in range(num_samples):
        farthest_points_indices[i] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()
    return points[farthest_points_indices]

def nearest_neighbors_torch(farthest_points, all_points, k):
    distances = torch.cdist(farthest_points, all_points[:, :3])
    _, indices = torch.topk(distances, k, largest=False, sorted=True)
    return indices

def sort_by_distance_torch(groups):
    sorted_groups = torch.zeros_like(groups)
    for i, group in enumerate(groups):
        distances = torch.norm(group[:, :3], dim=1)
        sorted_indices = torch.argsort(distances)
        sorted_groups[i] = group[sorted_indices]
    return sorted_groups

def Ply2Embedding_torch(file_path, num_samples=128, k=64):
    if file_path.endswith('.ply'):
        # Assuming read_ply_to_point_cloud outputs a numpy array
        point_cloud = read_ply_to_point_cloud(file_path)
        # Convert to PyTorch tensor and move to GPU
        point_cloud_torch = torch.tensor(point_cloud, dtype=torch.float32).cuda()
        
        # Normalize
        centroid = torch.mean(point_cloud_torch[:, :3], axis=0)
        point_cloud_torch[:, :3] -= centroid
        max_distance = torch.max(torch.abs(point_cloud_torch[:, :3]))
        if max_distance > 0:
            point_cloud_torch[:, :3] /= max_distance
        
        # FPS and KNN
        fps_points = farthest_point_sampling_torch(point_cloud_torch[:, :3], num_samples)
        knn_indices = nearest_neighbors_torch(fps_points, point_cloud_torch, k)
        feature_vector = point_cloud_torch[knn_indices].view(num_samples, k, -1)
        
        # Sort by distance
        sorted_feature_vector = sort_by_distance_torch(feature_vector)
        
        # Convert to NumPy array and move to CPU at the end
        sorted_feature_vector_np = sorted_feature_vector.cpu().numpy()
        
        return sorted_feature_vector_np
    else:
        raise TypeError('Not a Ply file.')

def process_and_save_ply_files(directory, save_directory, num_samples, k):
    num = 0
    for filename in tqdm(os.listdir(directory)):
        if num < 0:
            num += 1
            continue
        # print(filename)
        if filename.endswith('.ply'):
            file_path = os.path.join(directory, filename)
            point_cloud = read_ply_to_point_cloud(file_path)
            normalized_point_cloud = normalize_point_cloud(point_cloud)
            feature_vector = create_feature_vector(normalized_point_cloud, num_samples, k)

            # Sorting the points in each group by distance from the origin
            sorted_feature_vector = sort_by_distance(feature_vector)

            # Saving the feature vector
            # save_name = filename.split('_')[0] + '.npy'
            # np.save(os.path.join(save_directory, save_name), sorted_feature_vector)

def process_and_save_ply_files_torch(directory, save_directory, num_samples, k):
    os.makedirs(save_directory, exist_ok=True)
    
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.ply'):
            file_path = os.path.join(directory, filename)
            try:
                sorted_feature_vector_np = Ply2Embedding_torch(file_path, num_samples, k)

                # save_path = os.path.join(save_directory, filename.replace('.ply', '_features.npy'))
                # np.save(save_path, sorted_feature_vector_np)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
if __name__ == '__main__':
    data_root = os.environ.get('PSMA_DATA_ROOT', '/inspurfs/group/gaoshh/chenqy/pro_rna')
    surface_dir = os.path.join(data_root, 'protein_surface')
    embedding_dir = os.path.join(data_root, 'protein_embedding')
    process_and_save_ply_files(surface_dir, embedding_dir, num_samples=128, k=64)
    # process_and_save_ply_files_torch(surface_dir, embedding_dir, num_samples=128, k=64)
    pass
