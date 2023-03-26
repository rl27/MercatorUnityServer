import math
from typing import List, Tuple
from params import hp, path_configs
from model_data.hyperbolic_generative_model import HyperbolicGenerativeModel
from model_data.GANzoo import PoincareGANzoo
from model_data.LAFITE import PoincareLAFITE
# from model_data.JTVAE import PoincareJTVAE
import numpy as np
from scipy.spatial.distance import pdist, squareform

# models = {'poincare': PoincareGANzoo}
models = {'poincare': PoincareLAFITE}
# models = {'poincare': PoincareJTVAE}

class ImageSampler:
    def __init__(self, starting_tiles=None):
        self.sigma = hp['sigma']
        self.alpha = hp['alpha']
        self.lscale = hp['lscale']
        assert issubclass(models[hp['model_family']], HyperbolicGenerativeModel)
        assert self.sigma > 0
        assert self.alpha > 0
        self.model_family = models[hp['model_family']]
        self.generative_model = self.model_family()

    def generate_image_from_sentence(self, sentence: str):
        assert len(sentence) > 0, "Sentence should not be empty"
        return self.generative_model.generate_image_from_sentence(sentence)

    def generate_images_for_megatile(self, world_data: List[Tuple[int, float, float]], tile_coords: List[Tuple[float, float]], latent_vectors: List[List[float]]):
        ### 1. read in old data with schema (tile_index, tile_x, tile_y, latent_vector)
        assert len(world_data) == len(latent_vectors), "Number of world coords should equal number of latent vectors"

        # ### 2. sample new latent space vectors
        # latent_space_vectors = self.sample_latent_vector(data_df=data_df, world_data=world_data, list_of_test_coords=tile_coords)

        # new_tile_records = []
        # ### 3. write images to folder
        # for i, v in enumerate(latent_space_vectors):
        #     tile_idx = i + len(data_df)
        #     im = self.generative_model.generate_image_from_latent_vector(v)
        #     im.save(join(path_configs['world_data_dir'], 'images', 'tile{tile_idx}.png'.format(tile_idx=tile_idx)), "PNG")
        #     if not isinstance(v, list):
        #         v = v.tolist()
        #     new_tile_records.append({'tile_index': tile_idx,
        #                              'tile_x': tile_coords[i][0],
        #                              'tile_y': tile_coords[i][1],
        #                              'latent_vector': v})

        # ### 4. write new data with schema (tile_index, tile_x, tile_y, latent_vector)
        # new_df = pd.DataFrame(new_tile_records)
        # data_df = pd.concat([data_df, new_df])
        # data_df.to_csv(self.path_to_world_data)


        images = []
        vectors = []

        ########## Initialization
        if len(world_data) == 0:

            # same as hyperboloid_distance, but takes in two vectors
            def dist2(a, b):
                x1, z1 = a
                x2, z2 = b
                y1 = math.sqrt(1 + x1**2 + z1**2)
                y2 = math.sqrt(1 + x2**2 + z2**2)
                minkDot = y1 * y2 - x1 * x2 - z1 * z2
                if minkDot < 1:
                    return 0
                return math.acosh(minkDot)

            dists = squareform(pdist(tile_coords, dist2))
            K = np.exp(-0.5*dists / self.lscale**2) + 1e-6*np.eye(len(tile_coords))
            cK = np.linalg.cholesky(K)
            noise = cK @ np.random.randn(len(tile_coords), self.model_family.latent_dim)
            noise = noise / np.linalg.norm(noise, axis=1)[:, np.newaxis]  # normalize
            ims = self.generative_model.generate_multiple(noise)
            for i, im in enumerate(ims):
                images.append(im)
                vectors.append(noise[i].tolist())

            return images, vectors

        ##########

        # Using a loop to generate each vector individually; may be more efficient to generate at same time?
        # Appending to world_data and latent_vectors to make this work
        for i in range(len(tile_coords)):

            v = self.sample_latent_vector(list_of_train_coords=world_data, list_of_test_coords=[tile_coords[i]], latent_vectors=latent_vectors)
            
            v = (v / np.linalg.norm(v)).tolist()  # normalize

            vectors.append(v[0])

            world_data.append(tile_coords[i])
            latent_vectors.append(v[0])

            im = self.generative_model.generate_image_from_latent_vector(v)
            images.append(im)
        
        # The code below generates everything at once.
        # new_latent_vectors = self.sample_latent_vector(list_of_train_coords=world_data, list_of_test_coords=tile_coords, latent_vectors=latent_vectors)
        # for i, v in enumerate(new_latent_vectors):
        #     im = self.generative_model.generate_image_from_latent_vector(v)
        #     images.append(im)
        #     vectors.append(v)

        return images, vectors

    # this implements the GP logic
    def sample_latent_vector(self, list_of_train_coords, list_of_test_coords, latent_vectors):
        """
        Let the number of training points (X: coords from data_df) be m.
        Let the number of test points (X_*: coords from list_of_test_coords) be n.
        Let the latent space dimensionality be d.
        We want to do posterior inference f(X_*) | f(X), X, X_*.

        1. compute the training covariance matrix K of size mxm.
        2. compute the train-test covariance matrix K_* of size mxn.
        3. compute the test covariance matrix K_** of size nxn.
        4. compute the posterior covariance SIGMA = K_** - K_*^T K^(-1) K_*.  This is the same for each of the d GPs.

        5. for each latent space dim:
            - compute the posterior mean MU = K_*^T K f
            - sample from the multivariate normal distribution N(MU, SIGMA).  Will be a vector of size n.

        6. Collect all d vectors of size n into an nxd matrix.  Transpose and re-slice into n vectors of size d.

        :return: a list of n vectors of size d representing the sampled latent space vectors.
        """

        m = len(list_of_train_coords)
        n = len(list_of_test_coords)
        d = self.model_family.latent_dim

        data = np.vstack([np.array(x) for x in latent_vectors])

        # 1. compute the training covariance matrix K of size mxm.
        train_cov = self.compute_covariance_matrix(list_of_train_coords, list_of_train_coords)
        train_cov += np.eye(train_cov.shape[0])*1e-8  # Prevent singular matrices
        # 2. compute the train-test covariance matrix K_* of size mxn.
        train_test_cov = self.compute_covariance_matrix(list_of_train_coords, list_of_test_coords)
        # 3. compute the test covariance matrix K_** of size nxn.
        test_cov = self.compute_covariance_matrix(list_of_test_coords, list_of_test_coords)

        # 4. compute the posterior covariance SIGMA = K_** - K_*^T K^(-1) K_*. This is the same for each of the d GPs.
        posterior_cov = test_cov - train_test_cov.T @ np.linalg.inv(train_cov) @ train_test_cov

        # 5. for each latent space dim:
        latent_slices = []
        for dim in range(d):
            # compute the posterior mean MU = K_*^T K^(-1) f
            f = data[:, dim]  # f = data.apply(lambda x: x[dim])
            posterior_mean = train_test_cov.T @ np.linalg.inv(train_cov) @ f
            # sample from the multivariate normal distribution N(MU, SIGMA).  Will be a vector of size n.
            test_sample_along_dim = np.random.multivariate_normal(mean=posterior_mean, cov=posterior_cov)
            latent_slices.append(test_sample_along_dim)

        # 6. Collect all d vectors of size n into an nxd matrix. Transpose and re-slice into n vectors of size d.
        #latent_matrix = np.concatenate(latent_slices).reshape((n, d))
        latent_matrix = np.array(latent_slices).T

        #latent_matrix = latent_matrix / np.sqrt(np.sum(latent_matrix**2, axis=1, keepdims=True))
        #latent_matrix = np.random.randn(*latent_matrix.shape)

        return latent_matrix.tolist()

    # closed form of distance formula on Poincare disk
    def geodesic_distance(self, x1, y1, x2, y2):
        euclidean_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        norm1 = math.sqrt(x1 ** 2 + y1 ** 2)
        norm2 = math.sqrt(x2 ** 2 + y2 ** 2)
        assert norm1 < 1
        assert norm2 < 1

        dot_product = x1 * x2 + y1 * y2

        numerator = euclidean_distance + math.sqrt((norm1 ** 2) * (norm2 ** 2) - 2 * dot_product + 1)
        denominator = math.sqrt((1 - norm1 ** 2) * (1 - norm2 ** 2))
        return 2 * math.log(numerator / denominator)

    # distance formula on the 3D hyperboloid, given only the non-vertical-axis coordinates
    def hyperboloid_distance(self, x1, z1, x2, z2):
        y1 = math.sqrt(1 + x1**2 + z1**2)
        y2 = math.sqrt(1 + x2**2 + z2**2)
        minkDot = y1 * y2 - x1 * x2 - z1 * z2
        if minkDot < 1:
            return 0
        return math.acosh(minkDot)

    # our kernel function
    # this is the secret sauce binding the hyperbolic geometry to the ML
    # k(x,x') = sigma^2 e^(-alpha * d(x, x')) is a PD kernel for any distance d with sigma^2, alpha > 0
    # got this from Didong Li
    def k(self, x1, y1, x2, y2):
        return (self.sigma ** 2) * math.exp(-1 * self.hyperboloid_distance(x1, y1, x2, y2) / self.alpha)
        # return math.exp(-0.5 * (((x1-x2)/self.alpha)**2 + ((y1-y2)/self.alpha)**2))


    def compute_covariance_matrix(self, coords1, coords2):
        """
        :param coords1: list of a coords
        :param coords2: list of b coords
        :return: return a covariance matrix of size axb
        """
        cov = np.zeros((len(coords1), len(coords2)))
        for row, (x1, y1) in enumerate(coords1):
            for col, (x2, y2) in enumerate(coords2):
                cov[row, col] = self.k(x1, y1, x2, y2)
        return cov

    def get_random_coords(self, d):
        """
        gets points uniformly distributed in the d-dim unit ball
        :return:
        """
        # u = np.random.normal(0, 1, d)
        # norm = np.sum(u ** 2) ** 0.5
        # r = np.random.random() ** (1 / d)
        # return r * u / norm
        return np.random.normal(0, 1, d)


if __name__ == "__main__":
    sampler = ImageSampler()

    first_set_of_tile_coords = [sampler.get_random_coords(2) for _ in range(20)]
    sampler.generate_images_for_megatile(first_set_of_tile_coords)
