from .library.facelift_helpers import *

class FaceLift:
    """
    Implementation of FACE and FaceLift

    Notes:
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "method": {"kde", "knn", "epsilon"},
            Decides the type of FACE.
        * "weight_function": function
            the function to calculate the weight of an edge in a graph.
        * "prediction_threshold": float [0 =< x =< 1]
            a lower bound on the prediction confidence from the predictor.
            (t_p in line 10, Algorithm 1)
        * "density_threshold": float [0 < x < 1]
            density threshold t_d used in line 10, Algorithm 1. 
            This hyper-parameter is used only in "kde" method.
        * "distance_threshold": float
            distance threshold used in line 2, Algorithm 1.
        * "n_neighbours": int [0< x < n_samples]
            k in equation (3) for k-NN graph.
        * "distance_function": function
            distance function d; currently support l1, l2 and asymmetric distances; 
            default is l2.
        * "penalty_term": float
            the difficulty of changing features against the directions.
    """
    
    def __init__(self, X, hyperparams) -> None:
        self.method = hyperparams["method"]
        self.distance_threshold = hyperparams["distance_threshold"]
        self.prediction_threshold = hyperparams["prediction_threshold"]
        self.penalty_term = hyperparams["penalty_term"]
        self.directed = hyperparams["directed"]
        self.n_samples, self.n_features = X.shape
        self.n_features = 1

        self.density_threshold = hyperparams["kde"]["density_threshold"]
        self.n_neighbours = hyperparams["knn"]["n_neighbours"]
        
        self.weight_function = hyperparams["method"]
        self.distance_function = hyperparams["distance_function"]
        
    
    # Check that the value of each parameter is valid.
    @property
    def method(self) -> str:
        """
        return the method to build the graph
        return: str
        """
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        if method in ["knn", "epsilon", "kde"]:
            self._method = method
        else:
            raise ValueError("Mode has to be either knn, epsilon, or kde.")

    @property
    def directed(self) -> int:
        """
        whether the graph is bidirectional or unidirectional
        return: boolean
        """
        return self._directed
    
    @directed.setter
    def directed(self, directed: bool) -> None:
        # Check that each instance has to be a positive integer
        if isinstance(directed, bool):
            self._directed = directed
        else:
            raise ValueError("The value should be either True or False")

    @property
    def penalty_term(self) -> float:
        return self._penalty_term
    
    @penalty_term.setter
    def penalty_term(self, penalty_term: float) -> None:
        if penalty_term > 0:
            self._penalty_term = penalty_term
        else:
            raise ValueError("Penalty term has to be positive.")
    
    @property
    def weight_function(self):
        """
        return the weight function to calculate the weight of a edge in the graph
        """
        return self._weight_function
    
    @weight_function.setter
    def weight_function(self, method):
        if method == "knn":
            self._weight_function = get_knn_weight_function(self.n_neighbours, self.n_features, self.n_samples)
        elif method == "epsilon":
            self._weight_function = get_epsilon_weight_function(self.distance_threshold, self.n_features)
        elif method == "kde":
            self._weight_function = lambda x: -np.log(x)
        else:
            raise ValueError("Weight function is not supported. Available distance functions are \"knn\", \"kde\" and \"epsilon\".")

    @property
    def prediction_threshold(self) -> float:
        """
        return the lower bound on the prediction confidence from the predictor
        """
        return self._prediction_threshold

    @prediction_threshold.setter
    def prediction_threshold(self, prediction_threshold) -> None:
        if prediction_threshold >= 0 and prediction_threshold <= 1:
            self._prediction_threshold = prediction_threshold
        else:
            raise ValueError("Prediction threshold should be in range [0,1] inclusive.")
        
    @property
    def density_threshold(self) -> float:
        """
        return density threshold t_d. used in "kde" method only.
        """
        return self._density_threshold

    @density_threshold.setter
    def density_threshold(self, density_threshold) -> None:
        assert density_threshold >= 0 and density_threshold <= 1, \
                "Density threshold must be in range [0,1]"
        self._density_threshold = density_threshold
    
    @property
    def distance_threshold(self) -> float:
        """
        distance threshold used in line 2, Algorithm 1
        """
        return self._distance_threshold
    
    @distance_threshold.setter
    def distance_threshold(self, distance_threshold) -> None:
        #TODO: what's the value range of epsilon? 
        # assert distance_threshold >= 0 and distance_threshold <= 1,\
        #        "Distance threshold should be in range [0,1] inclusive"
        self._distance_threshold = distance_threshold

    @property
    def n_neighbours(self) -> int:
        """
        return parameter num_neighbours in knn graph
        """
        return self._n_neighbours

    @n_neighbours.setter
    def n_neighbours(self, n_neighbours) -> None:
        assert isinstance(n_neighbours, int) and n_neighbours > 0, \
            "num_neighbours should be a positive integer"
        self._n_neighbours = n_neighbours
    
    @property
    def distance_function(self):
        """
        return the distance function d
        """
        return self._distance_function 
    
    @distance_function.setter
    def distance_function(self, distance_function):

        if distance_function == "l1":
            self._distance_function = lambda x_0, x_1: np.linalg.norm(x_0-x_1, ord=1)
        elif distance_function == "l2": 
            self._distance_function = lambda x_0, x_1: np.linalg.norm(x_0-x_1)
        elif distance_function == "weighted":
            self._distance_function = lambda x_0, x_1: calculate_weighted_distance(x_0, x_1, self.penalty_term)
        else:
            raise ValueError("Distance function is not supported. Available distance functions are \"l1\", \"l2\" and \"weighted\".")

    def init_weights_and_graph(self, X, y, class_labels, predictions):
        self.X = X
        self.y = y
        self.class_labels = class_labels
        self.predictions = predictions
        # self.n_samples, self.n_features = self.X.shape
        self.n_features = 1     # Just in MNIST dataset
        
        params = {
            "X": self.X,
            "method": self.method,
            "n_samples": self.n_samples,
            "distance_threshold": self.distance_threshold,
            "distance_function": self.distance_function,
            "weight_function": self.weight_function,
            "n_neighbours": self.n_neighbours,   # used only in "knn" method
            "prediction_threshold": self.prediction_threshold,
            "density_threshold": self.density_threshold,
            "penalty_term": self.penalty_term,
            "directed": self.directed,
        }
        
        print("Constructing the weight matrix...")
        weight_matrix = get_weight_matrix(params, parallel=False)
        print("Constructing graph...")
        self.graph = construct_graph(weight_matrix)
        print("Graph construction completed.")


    def _get_CF_dict_from_set_params(self, start_point_idx, cf_class, num_paths):
        dist_matrix, predecessors = find_shortest_path(self.graph, start_point_idx=start_point_idx)
        end_points_indices = get_closest_cf_point(dist_matrix, self.predictions, self.y, cf_class, self.class_labels, num_paths)

        cf_solutions = {}
        cf_solutions[start_point_idx] = {}
        for order, end_point_idx in enumerate(end_points_indices):
            shortest_path = reconstruct_shortest_path(predecessors, start_point_idx, end_point_idx)
            cf_solutions[start_point_idx][order] = shortest_path
        
        return cf_solutions


    def get_CF_path_from_set_params(self, start_point_idx, cf_class, num_paths=1):
        '''
        call this if you just want a list of inds of one path
        '''
        paths_dict = self._get_CF_dict_from_set_params(start_point_idx, cf_class, num_paths)[start_point_idx]
        print(paths_dict.keys())
        paths = paths_dict[0]
        paths_ints = []
        for p in paths:
            paths_ints.append(int(p))
        return paths_ints
    

    def get_counterfactuals(self, X, y, class_labels, predictions, start_point_idx, cf_class, num_paths):
        self.init_weights_and_graph(X, y, class_labels, predictions)
        cf_solutions = self._get_CF_dict_from_set_params(start_point_idx, cf_class, num_paths)
        return cf_solutions

        
