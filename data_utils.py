import pandas as pd
import numpy as np
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
    df = pd.read_csv(csv_path, dtype=np.uint8)
    X = df.iloc[:, 1:]/255
    X = X.astype("float32")

    # X = df.iloc[:, 1:].values.astype(np.float32)
    
    y = df.iloc[:, 0].values  

    # Subsample if n_samples is specified
    if n_samples is not None:
        if n_samples > X.shape[0]:
            raise ValueError(f"n_samples {n_samples} exceeds dataset size {X.shape[0]}")

        X, y = stratified_subsample(X, y, n_samples)
    
    return X, y



#### from ed's repo
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
