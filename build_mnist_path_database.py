import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
from shortest_path import shortest_path


def load_mnist_data(csv_path, n_samples=None):
    # Load MNIST from CSV (assuming first column is label, rest are pixels)
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values  

    # Subsample if n_samples is specified
    if n_samples is not None:
        if n_samples > X.shape[0]:
            raise ValueError(f"n_samples {n_samples} exceeds dataset size {X.shape[0]}")
        X = X[:n_samples]
        y = y[:n_samples]
    
    return X, y


def get_shortest_path_df(shortest_path_finder, 
                         start_idx, 
                         end_idx,
                         start_label=None,
                         end_label=None,
                         path_id=None):
    path = shortest_path_finder.get_shortest_path(start_idx, end_idx)

    # turn into df with columns: index, next index path, path id, start label in path, end label in path
    indicies = []
    next_indicies = []
    for i in range(len(path)-1):
        indicies.append(path[i])
        next_indicies.append(path[i+1])
    path_df = pd.DataFrame({
        'index': indicies,
        'next_index': next_indicies,
    })
    path_df['path_id'] = path_id if path_id is not None else f"{start_idx}_to_{end_idx}"
    path_df['start_label'] = start_label if start_label is not None else 'unknown'
    path_df['end_label'] = end_label if end_label is not None else 'unknown'
    return path_df


def main(number_of_samples=None, 
         distance_model='pca', 
         k_neighbors=20, 
         n_paths=100000):
    # load the MNIST data
    X, y = load_mnist_data("mnist_data/mnist_train.csv",
                           n_samples=number_of_samples)

    # build the path finder instance from the data
    shortest_path_finder = shortest_path(
        X, distance_model=distance_model, k=k_neighbors)

    # create a list of paths between random start and end points
    start_end_details = []
    already_seen_ids = set()
    for i in tqdm(range(n_paths), desc="pre determining paths"):
        # get a valid start and end index
        while True:
            # check valid pair
            start_idx = np.random.randint(0, X.shape[0]-1)
            end_idx = np.random.randint(0, X.shape[0]-1)
            if start_idx == end_idx:
                continue
            start_label = y[start_idx]
            end_label = y[end_idx]
            if start_label == end_label:
                continue
            path_id = f"{start_idx}_{end_idx}"
            if path_id in already_seen_ids:
                continue
            break
        already_seen_ids.add(path_id)
        details = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_label': start_label,
            'end_label': end_label,
            'path_id': path_id
        }
        start_end_details.append(details)

    # calculate all paths multiprocessing
    path_dfs = []
    for single in tqdm(start_end_details, desc="creating paths"):
        path_df = get_shortest_path_df(shortest_path_finder,
                                       single['start_idx'],
                                       single['end_idx'],
                                       start_label=single['start_label'],
                                       end_label=single['end_label'],
                                       path_id=single['path_id'])
        path_dfs.append(path_df)

    # concatenate all path dfs into a single df
    all_paths_df = pd.concat(path_dfs, ignore_index=True)
    path = f"mnist_paths_datasets/mnist_paths_{distance_model}_k{k_neighbors}_samples{number_of_samples}.csv"
    os.makedirs("mnist_paths_datasets", exist_ok=True)
    all_paths_df.to_csv(path, index=False)
    print(f"Saved paths to {path}")


if __name__ == "__main__":
    number_of_samples = None  # 59999 or None for full dataset
    # number_of_samples = 2000  # for quick testing
    distance_model = 'pca'  # options: 'raw', 'pca', or 'tsne'
    k_neighbors = 20 # number of neighbors for k-NN graph
    n_paths = 100000  # number of random paths to create
    # n_paths = 10000  # number of random paths to create

    main(number_of_samples=number_of_samples,
         distance_model=distance_model,
         k_neighbors=k_neighbors,
         n_paths=n_paths)

    