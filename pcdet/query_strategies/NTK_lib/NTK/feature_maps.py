import math

from .. import utils
from .feature_data import *


class DataTransform:

    def __call__(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> FeatureData:
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        pieces = [self.forward(sub_data, sub_idxs)
                  for sub_idxs, sub_data in feature_data.iterate(idxs)]
        return ConcatFeatureData(pieces) if len(pieces) != 1 else pieces[0]

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        raise NotImplementedError()


class FeatureMap(DataTransform):

    def __init__(self, n_features: int, allow_precompute_features: bool = True):

        self.n_features = n_features
        self.allow_precompute_features = allow_precompute_features

    def get_n_features(self):
        #Returns the feature space dimension.
     
        return self.n_features

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:

        return TensorFeatureData(self.get_feature_matrix(feature_data, idxs))

    def precompute(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> Tuple['FeatureMap', FeatureData]:
       
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        if self.allow_precompute_features:
            return IdentityFeatureMap(n_features=self.n_features), self(feature_data, idxs)
        else:
            results = [self.precompute_soft_(sub_data, sub_idxs) for sub_idxs, sub_data in feature_data.iterate(idxs)]
            if len(results) == 1:
                return results[0]
            return results[0][0], ConcatFeatureData([r[1] for r in results])

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
    
        return self, feature_data[idxs]

    def posterior(self, feature_data: FeatureData, sigma: float, allow_kernel_space_posterior: bool = True) \
            -> 'FeatureMap':

        if self.n_features < 0 or (allow_kernel_space_posterior and self.n_features > max(1024, 3 * len(feature_data))):
            # compute the posterior in kernel space
            return KernelSpacePosteriorFeatureMap(feature_map=self, cond_data=feature_data, sigma=sigma)

        feature_matrix = self.get_feature_matrix(feature_data)
        eye = torch.eye(self.n_features, device=feature_matrix.device, dtype=feature_matrix.dtype)
        cov_matrix = feature_matrix.t().matmul(feature_matrix) + (sigma ** 2) * eye

        return SequentialFeatureMap(LinearFeatureMap(sigma * robust_cholesky_inv(cov_matrix).t()), [self])

    def get_feature_matrix(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> torch.Tensor:
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        return torch_cat([self.get_feature_matrix_impl_(sub_data, sub_idxs)
                          for sub_idxs, sub_data in feature_data.iterate(idxs)], dim=-2)

    def get_kernel_matrix(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                          idxs_1: Optional[Indexes] = None, idxs_2: Optional[Indexes] = None) -> torch.Tensor:
        idxs_1 = Indexes(feature_data_1.get_n_samples(), idxs_1)
        idxs_2 = Indexes(feature_data_2.get_n_samples(), idxs_2)
        return torch_cat([torch_cat([self.get_kernel_matrix_impl_(sub_data_1, sub_data_2, sub_idxs_1, sub_idxs_2)
                                     for sub_idxs_2, sub_data_2 in feature_data_2.iterate(idxs_2)], dim=-1)
                          for sub_idxs_1, sub_data_1 in feature_data_1.iterate(idxs_1)], dim=-2)

    def get_kernel_matrix_diag(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> torch.Tensor:
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        return torch_cat([self.get_kernel_matrix_diag_impl_(sub_data, sub_idxs)
                          for sub_idxs, sub_data in feature_data.iterate(idxs)], dim=-1)

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        raise NotImplementedError()

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:

        return self.get_feature_matrix(feature_data_1, idxs_1).matmul(
            self.get_feature_matrix(feature_data_2, idxs_2).t())

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        return (self.get_feature_matrix(feature_data, idxs) ** 2).sum(dim=-1)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        raise NotImplementedError()


class IdentityFeatureMap(FeatureMap):
    """
    This class represents the identity feature map phi(x) = x, and the linear kernel k(x, y) = x^T y.
    """
    def __init__(self, n_features: int):
        """
        :param n_features: Dimension of the inputs.
        """
        super().__init__(n_features=n_features)

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        return feature_data.get_tensor(idxs)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        # Gaussian sketch
        matrix = torch.randn(self.n_features, n_features)
        if config.get('sketch_norm', False):
            # it is possible to normalize the Gaussian vectors, which can make a difference for small input dimensions.
            matrix /= (matrix ** 2).sum(dim=0, keepdim=True).sqrt() / math.sqrt(self.n_features)
        matrix /= math.sqrt(n_features)
        return SequentialFeatureMap(LinearFeatureMap(matrix), [self])


class ReLUNTKFeatureMap(FeatureMap):
    """
    This feature map represents the Neural Tangent Kernel (Jacot et al., 2018) corresponding to a ReLU NN
    in Neural Tangent Parameterization. We implement the form of Lee et al. (2019), with factors sigma_w and sigma_b,
    and where the biases are initialized from N(0, 1).
    """
    # following SM 3 and 5 in
    # https://proceedings.neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html
    def __init__(self, n_layers=3, sigma_w_sq=2.0, sigma_b_sq=0.0):
        """
        :param n_layers: Number of layers of the corresponding NN
        :param sigma_w_sq: sigma_w**2 in the notation of Lee et al. (2019)
        :param sigma_b_sq: sigma_b**2 in the notation of Lee et al. (2019)
        """
        super().__init__(n_features=-1, allow_precompute_features=False)
        self.n_layers = n_layers
        self.sigma_w_sq = sigma_w_sq
        self.sigma_b_sq = sigma_b_sq

    def t_and_tdot_(self, a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_sqrt = a.sqrt()
        d_sqrt = d.sqrt()
        cos_theta = torch.clip(b / (a_sqrt * d_sqrt + 1e-30), min=-1.0, max=1.0)
        theta = torch.arccos(cos_theta)
        t = 1 / (2 * math.pi) * a_sqrt * d_sqrt * ((1 - cos_theta ** 2).sqrt() + (math.pi - theta) * cos_theta)
        tdot = 1 / (2 * math.pi) * (math.pi - theta)
        return t, tdot

    def diag_prop_(self, diag: torch.Tensor) -> torch.Tensor:
        # evaluate (k_l(x, x))_{x \in X} from diag = (k_{l-1}(x, x))_{x \in X}
        t = 0.5 * diag
        return self.sigma_w_sq * t + self.sigma_b_sq

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        feature_mat_1 = feature_data_1.get_tensor(idxs_1)
        feature_mat_2 = feature_data_2.get_tensor(idxs_2)
        kernel_mat = feature_mat_1.matmul(feature_mat_2.t())
        d_in = feature_mat_1.shape[1]
        diag_1 = self.sigma_w_sq / d_in * (feature_mat_1 ** 2).sum(dim=1) + self.sigma_b_sq
        diag_2 = self.sigma_w_sq / d_in * (feature_mat_2 ** 2).sum(dim=1) + self.sigma_b_sq
        nngp_mat = self.sigma_w_sq / d_in * kernel_mat + self.sigma_b_sq
        ntk_mat = nngp_mat
        for i in range(self.n_layers - 1):
            t, tdot = self.t_and_tdot_(diag_1[:, None], nngp_mat, diag_2[None, :])
            nngp_mat = self.sigma_w_sq * t + self.sigma_b_sq
            ntk_mat = nngp_mat + self.sigma_w_sq * ntk_mat * tdot
            diag_1 = self.diag_prop_(diag_1)
            diag_2 = self.diag_prop_(diag_2)
        return ntk_mat

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        feature_mat = feature_data.get_tensor(idxs)
        d_in = feature_mat.shape[1]
        nngp_diag = self.sigma_w_sq / d_in * (feature_mat ** 2).sum(dim=1) + self.sigma_b_sq
        ntk_diag = nngp_diag
        for i in range(self.n_layers - 1):
            nngp_diag = self.diag_prop_(nngp_diag)
            ntk_diag = nngp_diag + self.sigma_w_sq * 0.5 * ntk_diag
        return ntk_diag



class LaplaceKernelFeatureMap(FeatureMap):
    """
    Laplace kernel, k(x, y) = exp(-scale*||x-y||).
    """
    def __init__(self, scale: float = 1.0):
        # Scale parameter of the Laplace kernel. Larger scale yields a narrower kernel.
        super().__init__(n_features=-1, allow_precompute_features=False)
        self.scale = scale

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        feature_mat_1 = feature_data_1.get_tensor(idxs_1)
        feature_mat_2 = feature_data_2.get_tensor(idxs_2)

        sq_dist_mat = (feature_mat_1**2).sum(dim=-1)[:, None] + (feature_mat_2**2).sum(dim=-1)[None, :] \
                   - 2 * feature_mat_1 @ feature_mat_2.t()
        return torch.exp(-self.scale*torch.sqrt(sq_dist_mat))

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        feature_mat = feature_data.get_tensor(idxs)
        return torch.ones_like(feature_mat[:, 0])






class ToDoubleTransform(DataTransform):
    """
    Transforms data to float64 format
    """
    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        return SubsetFeatureData(feature_data.cast_to(torch.float64), idxs)


def robust_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """
    Implements a Cholesky decomposition.
    """
    eps = 1e-5 * matrix.trace() / matrix.shape[-1]
    L = None
    for i in range(10):
        try:
            L = torch.linalg.cholesky(matrix)
            break
        except RuntimeError:
            print('Increasing jitter for Cholesky decomposition', flush=True)
            matrix += eps * torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
            eps *= 2
    if L is None:
        raise RuntimeError('Could not Cholesky decompose the matrix')
    return L


def robust_cholesky_inv(matrix: torch.Tensor) -> torch.Tensor:
    # returns a matrix A such that matrix^{-1} = A^T A
    L = robust_cholesky(matrix)
    return L.inverse()