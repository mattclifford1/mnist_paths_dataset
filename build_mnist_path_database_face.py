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
    already_tried = []
    data_size = X.shape[0]
    print(f"Creating {n_paths} paths with data size {data_size}...")

    # max number of paths, in reality not all will make a valid path (FACE gets stuck on returns None path)
    expected_max_paths = n_paths * (len(labels)-1)
    # alt method: make all possible paths and shuffle that list, would only work with the one path per start ind and end label version of FACE
    i = 0
    while i < n_paths + 1:
        start_idx = np.random.randint(0, X.shape[0]-1)
        start_label = y[start_idx]
        # get random number in labels that is not start label
        possible_end_labels = list(labels - set([start_label]))
        if len(possible_end_labels) == 0:
            raise ValueError("No possible end labels found")
        end_label = random.choice(possible_end_labels)
        # path_id = f"{start_idx}_to_{end_label}"

        str_id = f'{start_idx} -> {end_label}'
        # don't redo already tried since only path is returned from start to end label only
        if str_id in already_tried:
            continue

        # make the paths
        path_df = get_face_path_df(face_finder,
                                       start_idx,
                                       end_label,
                                       start_label=y[start_idx])
        already_tried.append(str_id)

        # only for FACE where one path per start and end label is possible
        if len(already_tried) == expected_max_paths:
            print("Reached maximum number of unique paths possible.")
            break

        if path_df is None:
            continue

        path_dfs.append(path_df)


        # see if we need to save intermediate results
        if (i) % (n_paths/10) == 0:
            path = f"mnist_paths_datasets/mnist_paths_FACE_paths-{i}_datasize-{data_size}.csv"
            save_dfs(path_dfs, path)

        i += 1
    # concatenate all path dfs into a single df
    all_paths_df = pd.concat(path_dfs, ignore_index=True)
    data_size = X.shape[0]
    path = f"mnist_paths_datasets/mnist_paths_FACE_paths-{n_paths}_datasize-{data_size}.csv"
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
         parallel=False)
    end_time = time.time()
    # show time in hours, minutes, seconds
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    