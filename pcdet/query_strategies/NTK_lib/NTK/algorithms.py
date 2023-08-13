import torch
import torch.nn as nn

from ..layers import *
from .features import *
from .selection import *


def select_batch(batch_size: int, models: List[nn.Module], data: Dict[str, FeatureData],
                 y_train: Optional[torch.Tensor],
                 base_kernel: str, kernel_transforms: List[Tuple[str, List]], selection_method: str,
                 precomp_batch_size: int = 32768, nn_batch_size=8192, **config) \
        -> Tuple[torch.Tensor, Dict[str, Any]]:

    bs = BatchSelectorImpl(models, data, y_train)
    return bs.select(base_kernel=base_kernel, kernel_transforms=kernel_transforms, selection_method=selection_method,
                     batch_size=batch_size, precomp_batch_size=precomp_batch_size, nn_batch_size=nn_batch_size,
                     **config)


class BatchSelectorImpl:
    def __init__(self, models: List[nn.Module], data: Dict[str, FeatureData], y_train: Optional[torch.Tensor]):

        self.data = data
        self.models = models
        self.features = {}  # will be computed in select()
        self.n_models = len(models)
        self.y_train = y_train
        self.has_select_been_called = False
        self.device = self.data['train'].get_device()

    def apply_tfm(self, model_idx: int, tfm: FeaturesTransform):
        for key in self.features:
            self.features[key][model_idx] = tfm(self.features[key][model_idx])

    def to_float64(self):
        for key in self.data:
            self.data[key] = self.data[key].cast_to(torch.float64)

    def select(self, base_kernel: str, kernel_transforms: List[Tuple[str, List]], selection_method: str,
               batch_size: int, precomp_batch_size: int = 32768, nn_batch_size=8192, **config) \
            -> Tuple[torch.Tensor, Dict[str, Any]]:
    
        if self.has_select_been_called:
            raise RuntimeError('select() can only be called once per BatchSelector object')
        self.has_select_been_called = True

        allow_tf32_before = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # do not use tf32 since it causes large numerical errors

        if config.get('allow_float64', False):
            use_float64 = (selection_method in ['maxdet', 'bait'])

            for tfm_name, tfm_args in kernel_transforms:
                if tfm_name in ['train', 'pool', 'acs-rf', 'acs-rf-hyper', 'acs-grad']:
                    use_float64 = True
        else:
            use_float64 = False

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        kernel_timer = utils.Timer()
        kernel_timer.start()

        if base_kernel == 'ntk':  # KECOR
            feature_maps = [ReLUNTKFeatureMap(n_layers=config.get('n_ntk_layers', len([
                l for l in model.modules() if isinstance(l, LayerGradientComputation)])),
                                              sigma_w_sq=config.get('weight_gain', 0.4)**2,
                                              sigma_b_sq=config.get('sigma_b', 0.0)**2) for model in self.models]
            if use_float64:
                self.to_float64()
        elif base_kernel == 'last':  # KECOR-LAST
            n_last_layers = config.get('n_last_layers', 1)
            feature_maps = []
            grad_dict = config.get('layer_grad_dict', {nn.Linear: LinearGradientComputation})
            for model in self.models:
                grad_layers = []
                for layer in model.modules():
                    if isinstance(layer, LayerGradientComputation):
                        grad_layers.append(layer)
                    elif type(layer) in grad_dict:
                        grad_layers.append(grad_dict[type(layer)](layer))
                feature_maps.append(create_grad_feature_map(model, grad_layers[-n_last_layers:],
                                                            use_float64=use_float64))
        elif base_kernel == 'linear': # KECOR-LINEAR
            feature_maps = [IdentityFeatureMap(n_features=self.data['train'].get_tensor(0).shape[-1]) for model in self.models]
            if use_float64:
                self.to_float64()
        elif base_kernel == 'laplace': # # KECOR-RBF
            feature_maps = [LaplaceKernelFeatureMap(scale=config.get('laplace_scale', 1.0)) for model in self.models]
            if use_float64:
                self.to_float64()
        else:
            raise ValueError(f'Unknown base kernel "{base_kernel}"')

        self.features = {key: [Features(fm, feature_data) for fm in feature_maps]
                         for key, feature_data in self.data.items()}

        if base_kernel in ['last', 'grad']:
            for i in range(self.n_models):
                # use smaller batch size for NN evaluation
                self.apply_tfm(i, BatchTransform(batch_size=nn_batch_size))

        for tfm_name, args in kernel_transforms:
            if tfm_name == 'train':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 2:
                        self.apply_tfm(i, self.features['train'][i].scale_tfm(factor=args[1]))
                    self.apply_tfm(i, self.features['train'][i].posterior_tfm(args[0], **config))
            elif tfm_name == 'pool':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 2:
                        self.apply_tfm(i, self.features['pool'][i].scale_tfm(factor=args[1]))
                    self.apply_tfm(i, self.features['pool'][i].posterior_tfm(args[0], **config))
            else:
                raise ValueError(f'Unknown kernel transform "{tfm_name}"')

        for i in range(self.n_models):
            self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        kernel_timer.pause()

        eff_dim = None

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        selection_timer = utils.Timer()
        selection_timer.start()

        # only pick first model (if multiple models were there, they should have been ensembled by now)
        self.features = {key: val[0] for key, val in self.features.items()}

        if selection_method == 'random':
            alg = RandomSelectionMethod(self.features['pool'], **config)
        elif selection_method == 'maxdet':
            sel_with_train = config.get('sel_with_train', None)
            n_select = batch_size
            n_features = self.features['pool'].get_n_features()
            maxdet_sigma = config.get('maxdet_sigma', 0.0)

            alg = MaxDetSelectionMethod(self.features['pool'], self.features['train'],
                                            noise_sigma=config.get('maxdet_sigma', 0.0), **config)
            # for baseline BAIT
        elif selection_method == 'bait':
            alg = BaitFeatureSpaceSelectionMethod(self.features['pool'], self.features['train'],
                                            noise_sigma=config.get('bait_sigma', 0.0), **config)
        else:
            raise ValueError(f'Unknown selection method "{selection_method}"')

        batch_idxs = alg.select(batch_size)

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        selection_timer.pause()

        results_dict = {'kernel_time': kernel_timer.get_result_dict(),
                        'selection_time': selection_timer.get_result_dict(),
                        'selection_status': alg.get_status()}

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32_before

        return batch_idxs, results_dict
