import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
from face_path import facelift_paths
from sklearn.model_selection import train_test_split

def stratified_subsample(X, y, n_samples, random_state=42):
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state
    )
    return X_sub, y_sub


def load_mnist_data(csv_path, n_samples=None):
    # Load MNIST from CSV (assuming first column is label, rest are pixels)
    df = pd.read_csv(csv_path)
    # X = df.iloc[:, 1:].values.astype(np.float32)
    X = df.iloc[:, 1:].values.astype(np.uint8)
    y = df.iloc[:, 0].values  

    # Subsample if n_samples is specified
    if n_samples is not None:
        if n_samples > X.shape[0]:
            raise ValueError(f"n_samples {n_samples} exceeds dataset size {X.shape[0]}")

        X, y = stratified_subsample(X, y, n_samples)
    
    return X, y


def process_mnist_face(data_df):
    """ Process the MNIST dataset such that it is ready for model prediction.

    Args:
        data_df (DataFrame): a dataframe file read from raw input file

    Returns:
        X: features of all data instances as a dataframe
        y: labels of all data instances as a numpy array
    """

    # Pixel values need to be normalised between 0 and 1 for 
    # effective model training and distance calculation.
    X = (data_df.iloc[:, 1:]/255).astype("float32")
    y = np.asarray(data_df.iloc[:, 0])    

    # Sample a subset with 1000 instances for each digit.
    # Originally, each digit has 5000-6000 instances.
    # If we want to keep all instances, comment out the following code
    all_indices = []
    # for digit in [1,9]:
    for digit in np.arange(0,10):
        single_digit_indices = set(np.where(y == digit)[0][:1000])
        all_indices = set(all_indices).union(single_digit_indices)

    X = X.iloc[list(all_indices), :]
    y = y[list(all_indices)]

    return X, y


def get_face_path_df(face_finder, 
                     start_idx, 
                         end_label,
                         start_label=None):
    path = face_finder.get_face_path(start_idx, end_label)
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
         labels=set(range(10))):
    # load the MNIST data
    print("Loading MNIST data...")
    X, y = load_mnist_data("mnist_data/mnist_train.csv",
                           n_samples=number_of_samples)
    print(f"Loaded MNIST data with {X.shape[0]} samples.")
    
    # build the path finder instance from the data
    face_finder = facelift_paths(X, y)

    # create a list of paths between random start and end points
    start_end_details = []
    already_seen_ids = set()
    for i in tqdm(range(n_paths), desc="pre determining paths"):
        start_idx = np.random.randint(0, X.shape[0]-1)
        start_label = y[start_idx]
        # get random number in labels that is not start label
        possible_end_labels = list(labels - set([start_label]))
        if len(possible_end_labels) == 0:
            raise ValueError("No possible end labels found")
        end_label = random.choice(possible_end_labels)
        path_id = f"{start_idx}_to_{end_label}"
        already_seen_ids.add(path_id)
        details = {
            'start_idx': start_idx,
            'start_label': start_label,
            'end_label': end_label,
        }
        start_end_details.append(details)

    # calculate all paths multiprocessing
    path_dfs = []
    i = 0
    for single in tqdm(start_end_details, desc="creating paths"):
        path_df = get_face_path_df(face_finder,
                                       single['start_idx'],
                                       single['end_label'],
                                       start_label=single['start_label'])
        path_dfs.append(path_df)
        i += 1
        if i % 100 == 0:
            data_size = X.shape[0]
            save_dfs(path_dfs, data_size, n_paths=i)

    # concatenate all path dfs into a single df
    all_paths_df = pd.concat(path_dfs, ignore_index=True)
    data_size = X.shape[0]
    path = f"mnist_paths_datasets/mnist_paths_FACE_paths-{n_paths}_datasize-{data_size}.csv"
    os.makedirs("mnist_paths_datasets", exist_ok=True)
    all_paths_df.to_csv(path, index=False)
    print(f"Saved paths to {path}")

def save_dfs(dfs, data_size, n_paths):
    # concatenate all path dfs into a single df
    all_paths_df = pd.concat(dfs, ignore_index=True)
    path = f"mnist_paths_datasets/mnist_paths_FACE_paths-{n_paths}_datasize-{data_size}.csv"
    os.makedirs("mnist_paths_datasets", exist_ok=True)
    all_paths_df.to_csv(path, index=False)
    print(f"Saved paths to {path}")

if __name__ == "__main__":
    number_of_samples = None  # 59999 or None for full dataset
    number_of_samples = 10000  # for quick testing (can't go below 10000 for mnist?)
    # n_paths = 100000  # number of random paths to create
    n_paths = 1000  # number of random paths to create

    print(f'Path parameters: \nsamples={number_of_samples} \nn_paths={n_paths}')

    main(number_of_samples=number_of_samples,
         n_paths=n_paths)

    