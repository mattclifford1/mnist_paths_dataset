import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'face_lift')))

from utils.utils import init_seed
from utils.configurator import Config
from predictors.predictors import mlp_classifier
from facelift.facelift_model import FaceLift


class facelift_paths:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.make_model()
        self.init_config()
        # Initialise the facelift explainer
        self.cf_explainer = FaceLift(self.X, self.config)
        self.cf_explainer.init_weights_and_graph(self.X, self.y, self.class_labels, self.predictions)

    def make_model(self):
        self.model = mlp_classifier(self.X, self.y).fitted_model
        self.predictions = self.model.predict_proba(self.X)
        self.class_labels = list(map(int, self.model.classes_))

    def init_config(self):
        # Add input arguments to the configuration file
        self.config = Config(predictor='mlp').face_config_dict
        init_seed(self.config["seed"])

    def get_face_path(self,
                      start_point_idx, 
                      cf_class):
        return self.cf_explainer.get_CF_path_from_set_params(start_point_idx, cf_class)



# if __name__ == '__main__':

#     # Load the processed dataset
#     X, y = load_dataset('mnist')

#     # init FACE
#     face_maker = facelift_paths(X, y)

#     # get path
#     path = face_maker.get_face_path(start_point_idx=1, cf_class=9)
#     print(path)


    

