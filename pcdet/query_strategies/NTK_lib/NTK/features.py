from .feature_maps import *


class Features:

    def __init__(self, feature_map: FeatureMap, feature_data: FeatureData, diag: Optional[torch.Tensor] = None):

        self.feature_map = feature_map
        self.feature_data = feature_data
        self.diag = diag
        if diag is not None and not isinstance(diag, torch.Tensor):
            raise ValueError(f'diag has wrong type {type(diag)}')

    def precompute(self) -> 'Features':

        fm, fd = self.feature_map.precompute(self.feature_data)
        if self.diag is None:
            self.diag = fm.get_kernel_matrix_diag(fd)
        return Features(fm, fd, self.diag)

    def simplify(self) -> 'Features':

        return Features(self.feature_map, self.feature_data.simplify(), self.diag)


    def posterior_tfm(self, sigma: float = 1.0, allow_kernel_space_posterior: bool = True, **config) \
            -> 'FeaturesTransform':

        fm = self.feature_map.posterior(self.feature_data, sigma,
                                        allow_kernel_space_posterior=allow_kernel_space_posterior)
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, f.feature_data))

    def sketch_tfm(self, n_features: int, **config) -> 'FeaturesTransform':

        fm = self.feature_map.sketch(n_features, **config)
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, f.feature_data))


    def get_n_samples(self) -> int:
        return self.feature_data.get_n_samples()

    def __len__(self) -> int:
        return self.get_n_samples()

    def get_n_features(self) -> int:
        return self.feature_map.get_n_features()

    def get_device(self) -> str:
        return self.feature_data.get_device()

    def get_dtype(self) -> Any:
        return self.feature_data.get_dtype()

    def __getitem__(self, idxs: Union[int, slice, torch.Tensor]) -> 'Features':

        idxs = Indexes(self.get_n_samples(), idxs)
        return Features(self.feature_map, self.feature_data[idxs],
                        None if self.diag is None else self.diag[idxs.get_idxs()])

    def get_kernel_matrix_diag(self) -> torch.Tensor:

        if self.diag is None:
            self.diag = self.feature_map.get_kernel_matrix_diag(self.feature_data)
        return self.diag

    def get_kernel_matrix(self, other_features: 'Features') -> torch.Tensor:

        return self.feature_map.get_kernel_matrix(self.feature_data, other_features.feature_data)

    def get_feature_matrix(self) -> torch.Tensor:

        return self.feature_map.get_feature_matrix(self.feature_data)

    def get_sq_dists(self, other_features: 'Features') -> torch.Tensor:

        diag = self.get_kernel_matrix_diag()
        other_diag = other_features.get_kernel_matrix_diag()
        kernel_matrix = self.get_kernel_matrix(other_features)
        sq_dists = diag[:, None] + other_diag[None, :] - 2*kernel_matrix
        return sq_dists

    def batched(self, batch_size: int) -> 'Features':
        """
        Return a Features object that behaves as self,
        but where the feature data is virtually batched such that certain transformations are applied in batches.
        :param batch_size: Batch size of the batches. The last batch may be smaller.
        :return: Returns a Features object with batched feature data.
        """
        return Features(self.feature_map, self.feature_data.batched(batch_size), self.diag)



class FeaturesTransform:
    def __call__(self, features: Features) -> Features:
        raise NotImplementedError()


class LambdaFeaturesTransform(FeaturesTransform):
    def __init__(self, f: Callable[[Features], Features]):
        self.f = f

    def __call__(self, features: Features) -> Features:
        return self.f(features)


class SequentialFeaturesTransform(FeaturesTransform):
    
    def __init__(self, tfms: List[FeaturesTransform]):
        self.tfms = tfms

    def __call__(self, features: Features) -> Features:
        for tfm in self.tfms:
            features = tfm(features)
        return features


class PrecomputeTransform(FeaturesTransform):
    def __init__(self, batch_size: int = -1):
        self.batch_size = batch_size

    def __call__(self, features: Features) -> Features:
        if self.batch_size > 0:
            features = features.batched(self.batch_size)
        return features.precompute().simplify()


class BatchTransform(FeaturesTransform):
    def __init__(self, batch_size: int):

        self.batch_size = batch_size

    def __call__(self, features: Features) -> Features:
        return features.batched(self.batch_size)


