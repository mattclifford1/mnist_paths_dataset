import sys
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'face_lift')))
from utils.utils import init_seed
from utils.configurator import Config
from predictors.predictors import mlp_classifier
from facelift.facelift_model import FaceLift


class facelift_paths:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_embed_2d = None
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
                      cf_class, 
                      _print=False):
        return self.cf_explainer.get_CF_path_from_set_params(start_point_idx, cf_class, _print=_print)

    def plot_path(self, path):
        if self.X_embed_2d is None:
            self.X_embed_2d = TSNE(n_components=2, 
                              perplexity=30,
                              random_state=42).fit_transform(self.X_embed)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.X_embed_2d[:, 0], self.X_embed_2d[:, 1],
                   c=self.y, cmap="tab10", alpha=0.3, s=10)

        # overlay path line
        path_coords = self.X_embed_2d[path]
        ax.plot(path_coords[:, 0], path_coords[:, 1], c="red", linewidth=3)

        # add MNIST digit thumbnails at each path point
        for idx, (x0, y0) in zip(path, path_coords):
            img = self.X[idx].reshape(28, 28)
            imagebox = OffsetImage(img, zoom=0.5, cmap="gray")
            ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
            ax.add_artist(ab)

        # highlight start and end
        ax.scatter(path_coords[0, 0], path_coords[0, 1],
                   c="green", s=120, label="Start")
        ax.scatter(path_coords[-1, 0], path_coords[-1, 1],
                   c="blue", s=120, label="End")
        ax.legend()

        ax.set_title(f"Shortest path in {self.distance_mode.upper()} space")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        plt.show()

    def plot_images_in_path(self, path):
        fig, axes = plt.subplots(1, len(path), figsize=(len(path)*1.2, 2))
        for ax, idx in zip(axes, path):
            ax.imshow(self.X[idx].reshape(28, 28), cmap="gray")
            ax.set_title(str(self.y[idx]))
            ax.axis("off")

        plt.tight_layout()
        plt.show()



# if __name__ == '__main__':

#     # Load the processed dataset
#     X, y = load_dataset('mnist')

#     # init FACE
#     face_maker = facelift_paths(X, y)

#     # get path
#     path = face_maker.get_face_path(start_point_idx=1, cf_class=9)
#     print(path)


    

