from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
from joblib import Parallel, delayed


class shortest_path():
    def __init__(self,
                 X,
                 distance_model='raw',
                 k=5):
        # Choose distance space: 'raw', 'pca', or 'tsne'
        self.distance_mode = distance_model  # options: 'raw', 'pca', 'tsne'
        self.X = X
        self.n_samples = X.shape[0]
        self.k = k
        self.make_embedding()
        self.make_knn()
        self.build_graph()

    def make_embedding(self):
        if self.distance_mode == 'raw':
            self.X_embed = self.X
        elif self.distance_mode == 'pca':
            self.X_embed = PCA(
                n_components=50, random_state=42).fit_transform(self.X)
        elif self.distance_mode == 'tsne':
            self.X_embed = TSNE(n_components=2, perplexity=30,
                                random_state=42).fit_transform(self.X)
        else:
            raise ValueError("distance_mode must be 'raw', 'pca', or 'tsne'")

    def make_knn(self):
        # Compute k-NN graph
        nbrs = NearestNeighbors(n_neighbors=self.k+1,
                                algorithm='auto').fit(self.X_embed)
        self.distances, self.indices = nbrs.kneighbors(self.X_embed)

    def _build_distance_edges(self, i):
        edges = []
        for j, d in zip(self.indices[i][1:], self.distances[i][1:]):
            mse = mean_squared_error(self.X_embed[i], self.X_embed[j])
            edges.append((i, j, mse))
        return edges

    def build_graph(self):
        # Parallel edge construction
        results = Parallel(n_jobs=-1)(delayed(self._build_distance_edges)(i)
                                      for i in tqdm(range(self.n_samples), desc="creating distance matrix"))
        self.graph = nx.Graph()
        for edge_list in results:
            for i, j, mse in edge_list:
                self.graph.add_edge(i, j, weight=mse)

    def get_shortest_path(self, start, end):
        path = nx.shortest_path(self.graph, source=start,
                                target=end, weight='weight')

        # path_length = nx.shortest_path_length(self.graph, source=start, target=end, weight='weight')
        # print("Path:", path)
        # print("Path length:", len(path))
        # print("Labels along path:", y[path])
        return path

    def plot_path(self, path, y):
        if self.distance_mode not in ['tsne']:
            X_embed_2d = TSNE(n_components=2, perplexity=30,
                              random_state=42).fit_transform(self.X_embed)
        else:
            X_embed_2d = self.X_embed

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(X_embed_2d[:, 0], X_embed_2d[:, 1],
                   c=y, cmap="tab10", alpha=0.3, s=10)

        # overlay path line
        path_coords = X_embed_2d[path]
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

    def plot_images_in_path(self, path, y=None):
        fig, axes = plt.subplots(1, len(path), figsize=(len(path)*1.2, 2))
        for ax, idx in zip(axes, path):
            ax.imshow(self.X[idx].reshape(28, 28), cmap="gray")
            if y is not None:
                ax.set_title(str(y[idx]))
            ax.axis("off")

        plt.tight_layout()
        plt.show()
