import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.sparse import csr_matrix, csgraph
from numba import jit
from tqdm import tqdm
from joblib import Parallel, delayed

    
def get_weights_epsilon_parrelel(n_samples, X, distance_threshold,
                        distance_function, weight_function,
                        n_jobs=-1, use_tqdm=True):
    """Get the weight matrix in the epsilon graph (parallelized with tqdm)."""

    if not isinstance(X, np.ndarray):
        X = X.to_numpy()

    # Function to compute weights for one row
    def compute_row(i):
        x_i = X[i]
        row = np.zeros(n_samples)
        for j in range(n_samples):
            x_j = X[j]

            # Condition 2: custom conditions
            if not check_conditions(x_i, x_j):
                continue

            # Condition 1: distance threshold
            dist = distance_function(x_j, x_i)
            if dist <= distance_threshold and dist != 0:
                row[j] = weight_function(dist)
        return row

    # Wrap indices with tqdm if requested
    iterator = range(n_samples)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Building weight matrix (parallel)")

    # Run parallel over rows
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(compute_row)(i) for i in iterator)

    # Stack rows into full matrix
    W = np.vstack(results)

    return W


def get_weights_epsilon(n_samples, X, distance_threshold,                 
                  distance_function, weight_function):
    """Get the weight matrix in the epsilon graph

    Args:
        n_samples (int): number of rows in the dataframe
        X (pd.DataFrame): X
        distance_threshold (float): distance threshold
        distance_function (function(numpy.ndarray, numpy.ndarray) -> float):
                        calculate the distance between two points    
        weight_function (function(float) -> float): 
                        calculate the weight of an edge

    Returns:
        W: weight matrix of shape (n_samples x n_samples)
    """

    # Initialise the weight matrix
    W = np.zeros((n_samples, n_samples))

    if not isinstance(X, np.ndarray):
        X = X.to_numpy()

    # Calculate only the lower left of the weight matrix
    for i in tqdm(range(n_samples), desc="building weight matrix"):
        # get features of the ith data instance
        x_i = X[i]

        for j in range(n_samples):
            x_j = X[j]

            # In Line 2, Algorithm 1, there are 2 conditions to check
            # 1) distance between two points has to be smaller than or equal to distance_threshold.
            # 2) custom conditions c need to be satisfied.

            ## Condition 2: check custom conditions
            if not check_conditions(x_i, x_j):
                continue

            ## Condition 1: check distance
            dist = distance_function(x_j, x_i)
            if dist <= distance_threshold and dist != 0:
                W[i, j] = weight_function(dist)
                # W[j, i] = W[i, j]
            
    
    return W


def get_weights_knn(n_samples, X, distance_threshold, n_neighbours, 
                    distance_function, weight_function, directed = True):
    """Get the weight matrix in the knn graph

    Args:
        n_samples (int): number of rows in the dataframe
        X (Pd.DataFrame): X
        distance_threshold (float): distance threshold 
        n_neighbours (int): number of neighbours to be connected with
        distance_function (function(numpy.ndarray, numpy.ndarray) -> float):
                        calculate the distance between two points    
        weight_function (function(float) -> float): 
                        calculate the weight of an edge

    Returns:
        W: weight matrix of shape ((n_samples x n_samples)). 
    """
    K = np.zeros((n_samples, n_samples))
    W = K
    X = X.to_numpy()
    
    # find neighbours that are 1) within the distance threshold; 
    # 2) nearest k neighbours.    
    for i in range(n_samples):
        x_i = X[i]

        # Count the number of neighbours which are outside the distance-threshold
        counter = 0
        for j in range(n_samples):
            x_j = X[j]
            dist = distance_function(x_j, x_i)
            if (check_conditions(x_i, x_j) and dist <= distance_threshold and dist != 0):
                K[i, j] = dist
                W[i, j] = weight_function(dist)
            else:
                counter += 1
        
        # Once we get distance between neighbours within the threshold,
        # we only keep the top-k nearest points, then reset the rest to 0
        #TODO: edge conditions need to be checked, e.g. what if n is very large
        n_reset_neighbours = counter+n_neighbours+1
        reset_neighbours_idx = np.argsort(W[i, :])[n_reset_neighbours:]
        mask = np.ix_(reset_neighbours_idx)
        K[i, mask] = 0
        W[i, mask] = 0

    if directed:
        W = build_asymmetric_matrix(W, X, weight_function, distance_function)
    else:
        W = build_symmetric_matrix(W)
        
    return W


def get_weights_kde(n_samples, X, distance_threshold, 
                    distance_function, weight_function, density_scorer):
    """Get the weight matrix in the kde graph

    Args:
        n_samples (int): number of rows in the dataframe.
        X (pd.DataFrame): X
        distance_threshold (float): distance threshold
        distance_function (function(numpy.ndarray, numpy.ndarray) -> float):
                        Calculate the distance between two points.
        weight_function (function(float) -> float): 
                        Calculate the weight of an edge.
        density_scorer (function(array-like of shape(1, n_samples)) -> float): 
                        The function KernelDensity.score_samples() which returns the 
                        log-likelihood of this sample under the estimator.

    Returns:
        W: weight matrix of shape ((n_samples x n_samples)).
    """
    W = np.zeros((n_samples, n_samples))
    X = X.to_numpy()

    # Step 1: find neighbours within the distance threshold
    for i in range(n_samples):
        x_i = X[i]
        for j in range(n_samples):
            x_j = X[j]

            if not check_conditions(x_i, x_j):
                continue

            # The value of an edge is distance diminished by the density.
            dist = distance_function(x_j, x_i)
            if dist <= distance_threshold:
                mid_points = (x_i + x_j)/2
                # Density is the log-likelihood of the mid-point.
                density = density_scorer(mid_points.reshape(1, -1))
                # weight function = -np.log(x), so np.exp(density) has to be smaller than 1.
                W[i, j] = weight_function(np.exp(density)) * dist
                 
    
    return W

def get_kernel_density_estimator(X):
    """Get the best Kernel Density Estimator

    Args:
        X (pd.DataFrame): X

    Returns:
        kernel density estimator: the best fitted kernel density estimator 
    """
    # Assumption 1: the kernel to use is Gaussian
    # Assumption 2: use Euclidean distance (l2-norm) to calculate distance
    
    # bandwidth: the number of bins that divide the range of the data.
    bandwidths = np.logspace(-2, 0, 10)
    kd_estimators = GridSearchCV(
        estimator = KernelDensity(kernel="gaussian", metric="l2"),
        param_grid = {"bandwidth": bandwidths}
    )

    kd_estimators.fit(X)
    return kd_estimators.best_estimator_


def get_weight_matrix(params, parallel=True):
    """Return the correct weight matrix.

    Returns:
        weight_matrix: weight matrix of shape ((n_samples, n_samples))
    """
    weight_matrix = 0
    if params["method"] == "epsilon":
        if parallel:
            weight_matrix = get_weights_epsilon_parrelel(params["n_samples"], params["X"], 
                                                params["distance_threshold"], 
                                                params["distance_function"], 
                                                params["weight_function"])
        else:
            weight_matrix = get_weights_epsilon(params["n_samples"], params["X"], 
                                                params["distance_threshold"], 
                                                params["distance_function"], 
                                                params["weight_function"])
    # elif params["method"] == "kde":
    #     density_scorer = get_kernel_density_estimator(params["X"]).score_samples
    #     weight_matrix = get_weights_kde(params["n_samples"], params["X"], 
    #                                     params["distance_threshold"],
    #                                     params["distance_function"], 
    #                                     params["weight_function"],
    #                                     density_scorer)
    # elif params["method"] == "knn":
    #     weight_matrix = get_weights_knn(params["n_samples"], params["X"], 
    #                                     params["distance_threshold"], 
    #                                     params["n_neighbours"], 
    #                                     params["distance_function"], 
    #                                     params["weight_function"],
    #                                     params["directed"])

    return weight_matrix


def check_conditions(x_i, x_j):
    # check_conditions (function(numpy.ndarray, numpy.ndarray) -> Boolean): 
    #             check if additional custom conditions are satisfied
    return True


@jit(nopython=True)
def calculate_weighted_distance(v0, v1, penalty_term = 1.1):
    """Calculate the asymmetric distance between two instances

    Args:
        v0 (array): one data instance
        v1 (array): another data instance
        penalty_term (float, optional): how difficult to change features against the direction. Defaults to 1.1.

    Returns:
        weighted_diff (float): asymmetric distance between two instances.
    """
    diff = np.subtract(v0, v1)
    reweight_vector = np.where(diff>=0, 1, -penalty_term)
    weighted_diff = np.linalg.norm(diff*reweight_vector)
    return weighted_diff


def build_symmetric_matrix(kernel):
    for i in range(kernel.shape[0]):
        for j in range(i):
            kernel[j, i] = kernel[i,j]
    return kernel


def build_asymmetric_matrix(kernel, X, weight_func, distance_func):
    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel[i,j] != 0:
                x_i = X[i]
                x_j = X[j]
                dist = distance_func(x_i, x_j)
                kernel[j, i] = weight_func(dist)
    return kernel


def get_knn_weight_function(n_neighbours, n_features, n_samples):
    """Get the weight function to calculate weight of edge in knn graph.

    Args:
        n_neighbours (int): how significant you want for diminishing the distance between two points.
        n_features (int): number of features
        n_samples (int): number of data instances

    Returns:
        function: a weight function
    """
    # The volume of a d-dimensional unit sphere is pi**(d/2)/gamma(d/2+1)
    sphere_volume = math.pi**(n_features/2)/math.gamma(n_features/2 + 1)
    r = (n_neighbours/(n_samples * sphere_volume))

    weight_function = lambda x: -x*np.log(r/x)
    return weight_function


def get_epsilon_weight_function(distance_threshold, n_features):
    """Get the weight function to calculate weight of edge in epsilon graph.

    Args:
        distance_threshold (float): epsilon
        n_features (int): number of features

    Returns:
        function: a weight function
    """
    distance_threshold = 1
    weight_function = lambda x: -x*np.log((distance_threshold**n_features)/x)
    return weight_function


def construct_graph(weight_matrix):
    graph = csr_matrix(weight_matrix)
    return graph

def find_shortest_path(graph, start_point_idx, directed=True):
    """Get the shortest path from the start point to all other points.

    Args:
        graph (csr_matrix): a sparse matrix that contains the weight of the graph.
        start_point_idx (int): the index of the start point

    Returns:
        dist_matrix: the matrix of distances between graph nodes. It has shape (n_indices, n_nodes)
        predecessors: the matrix of predecessors to reconstruct the shortest paths. 
                It has shape (n_indices, n_nodes). Each entry predecessors[i, j] 
                gives the index of the previous node in the path from point i to point j.
    """

    # Compute the shortest distance path from start point to any other nodes.
    # dist_matrix has shape (n_indices, n_nodes).
    dist_matrix, predecessors = csgraph.dijkstra(
        csgraph=graph, directed=directed, indices=start_point_idx, return_predecessors=True
    )

    return dist_matrix, predecessors

def reconstruct_shortest_path(predecessors, start_point_idx, end_point_idx):
    """Get all the nodes along the path between the start point and the end point. 

    Args:
        predecessors (matrix of shape (1, n_nodes)): contain the previous node in the path.
        start_point_idx (int): the index of the start data point
        end_point_idx (int): the index of the end data point

    Returns:
        node_path (list): [start_point_idx, intermedium points index, end_point_idx]
    """

    if predecessors[end_point_idx] == start_point_idx:
        node_path = [end_point_idx]
    else:
        node_path = []
    intermedium_idx = end_point_idx
    count = 0
    max_count = 150
    while (predecessors[intermedium_idx] != start_point_idx):
        node_path.append(intermedium_idx)
        intermedium_idx = predecessors[intermedium_idx]
        count += 1
        if count > max_count:
            print("Warning: maximum path length exceeded.")
            return None
    if intermedium_idx != node_path[-1]:
        node_path.append(intermedium_idx)
    node_path.append(start_point_idx)
    
    return node_path[::-1]
    
def get_minimum_dist(dist_matrix):
    """get the shortest distance and its data index
    Args:
        dist_matrix (array): shape: 1 x n_nodes

    Returns:
        min_dist: minimum distance in the distance matrix
        min_dist_idx: index of the data point with the shortest dist
    """
    min_dist = np.min(np.ma.masked_where(dist_matrix==0, dist_matrix, copy=False)) 
    min_dist_idx = np.argmin(np.ma.masked_where(dist_matrix==0, dist_matrix, copy=False))
    return min_dist, min_dist_idx


def get_closest_cf_point(dist_matrix, predictions, y, target_class, class_labels, num_paths = 1, pred_threshold=0.55):
    """Get a set of CF points that satisfy the confidence threshold

    Args:
        dist_matrix (matrix(n_samples x n_samples)): edge weight/distance
        predictions (array): prediction probability 
        y (array): true label of X
        target_class (int): the destination CF class 
        class_labels (array): a list of original class labels 
        num_paths (int, optional): number of CF points. Defaults to 1.
        pred_threshold (float, optional): prediction threshold t_p. Defaults to 0.55.

    Returns:
        int: a list of CF points
    """
    assert num_paths > 0 and isinstance(num_paths, int), "only positive integers"
    end_point_idx = []
    path_count = 0
    for idx in np.argsort(np.ma.masked_where(dist_matrix==0, dist_matrix)):
        if (y[idx] == target_class and
        predictions[idx, class_labels.index(target_class)] >= pred_threshold): 
            end_point_idx.append(idx)
            if path_count >= num_paths-1:
                break
            else:
                path_count += 1
    return end_point_idx


# for branching factor calculation
def get_user_agency(sp_graph, start_point_idx, alternative_classes, predictions, y, class_labels, pred_threshold=0.55):
    dist_matrix, predecessors = find_shortest_path(sp_graph, start_point_idx)
    alt_class_dict = {}
    alt_path_dict = {}
    for alt_class in alternative_classes:
        alt_end_idx = get_closest_cf_point(dist_matrix, predictions, y, alt_class, class_labels, pred_threshold = pred_threshold)
        alt_end_dist = dist_matrix[alt_end_idx[0]]
        alt_class_dict[alt_class] = {alt_end_idx[0]: alt_end_dist}
        
        alt_end_path = reconstruct_shortest_path(predecessors, start_point_idx, alt_end_idx[0])
        alt_path_dict[alt_class] = {alt_end_idx[0]: alt_end_path}
    
    return alt_class_dict, alt_path_dict