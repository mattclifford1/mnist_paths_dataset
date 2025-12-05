import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
import time
from face_path import facelift_paths
from data_utils import load_mnist_data, save_dfs


def get_face_path_df(face_finder, 
                     start_idx, 
                         end_label,
                         start_label=None):
    path = face_finder.get_face_path(start_idx, end_label, _print=True)
    if path is None:
        return path
    end_idx = path[-1]

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
    path_df['path_id'] = f"{start_idx}_to_{end_idx}"
    path_df['start_label'] = start_label if start_label is not None else 'unknown'
    path_df['end_label'] = end_label if end_label is not None else 'unknown'
    return path_df


def main(number_of_samples=None, 
         n_paths=100000,
         labels=set(range(10)),
         parallel=False):
    # load the MNIST data
    print("Loading MNIST data...")
    X, y = load_mnist_data("mnist_data/mnist_train.csv",
                           n_samples=number_of_samples)
    print(f"Loaded MNIST data with {X.shape[0]} samples.")
    
    # build the path finder instance from the data
    face_finder = facelift_paths(X, y, parallel=parallel)

    # create a list of paths between random start and end points
    path_dfs = []
    data_size = X.shape[0]

    # max number of paths, in reality not all will make a valid path (FACE gets stuck on returns None path)
    possible_pairs = [(i, j) for i in range(data_size) for j in labels]
    # expected_max_paths = data_size * (len(labels)-1)
    expected_max_paths = len(possible_pairs)
    print(f"Creating {expected_max_paths} max paths with data size {data_size}...")
    # make all possible paths and shuffle that list, only works with the one path per start ind and end label version of FACE
    i = 0
    already_tried = []
    pbar = tqdm(total=expected_max_paths)
    for pair in possible_pairs:
        # get start and end of the path
        start_idx, end_label = pair
        start_label = y[start_idx]
        if start_label == end_label:
            pbar.update(1)
            continue

        # check the path is valid
        str_id = f'{start_idx} -> {end_label}'
        if str_id in already_tried:
            raise ValueError(f"Already tried path {str_id}, this should happen, check for bug!!!")

        # make the paths
        path_df = get_face_path_df(face_finder,
                                       start_idx,
                                       end_label,
                                       start_label=y[start_idx])
        # check the path
        already_tried.append(str_id)
        pbar.update(1)
        if path_df is None:
            continue
        path_dfs.append(path_df)

        # see if we need to save intermediate results
        if (i) % (n_paths/10) == 0:
            path = f"mnist_paths_datasets/mnist_paths_FACE_paths-{i}_datasize-{data_size}.csv"
            save_dfs(path_dfs, path)
        i += 1

    pbar.close()
    print(f"Created {len(path_dfs)} valid paths.")
    # concatenate all path dfs into a single df
    all_paths_df = pd.concat(path_dfs, ignore_index=True)
    data_size = X.shape[0]
    path = f"mnist_paths_datasets/mnist_paths_FACE_paths-{i}_datasize-{data_size}.csv"
    os.makedirs("mnist_paths_datasets", exist_ok=True)
    all_paths_df.to_csv(path, index=False)
    print(f"Saved paths to {path}")


if __name__ == "__main__":
    number_of_samples = None  # 59999 or None for full dataset
    number_of_samples = 10000  # for quick testing (can't go below 10000 for mnist?)
    n_paths = 100000  # number of random paths to create
    # n_paths = 1000  # number of random paths to create

    print(f'Path parameters: \nsamples={number_of_samples} \nn_paths={n_paths}')

    start_time = time.time()
    main(number_of_samples=number_of_samples,
         n_paths=n_paths,
         parallel=True,  # True uses more RAM so set to False for large dataset size
         )
    end_time = time.time()
    # show time in hours, minutes, seconds
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    