import torch.nn as nn

from .feature_maps import *


class LayerGradientComputation:

    def __init__(self, layer: nn.Module):

        self.layers = [layer]  # dirty hack to avoid infinite recursion in PyTorch if layer is self.
        super().__init__()   # in case this is used with multiple inheritance

    def get_layer(self) -> nn.Module:

        return self.layers[0]

    def get_feature_map(self) -> FeatureMap:

        raise NotImplementedError()

    def before_forward(self) -> None:

        raise NotImplementedError()

    def pop_feature_data(self) -> FeatureData:

        raise NotImplementedError()


class ModelGradTransform(DataTransform):

    def __init__(self, model: nn.Module, grad_layers: List[LayerGradientComputation]):

        self.model = model
        self.grad_layers = grad_layers
        self.requires_grad_list = [any([any([p is gl_p for gl_p in gl.get_layer().parameters()]) for gl in grad_layers])
                                   for p in model.parameters()]
        self.grad_params = [p for grad_layer in grad_layers for p in grad_layer.get_layer().parameters()]

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:

        for grad_layer in self.grad_layers:
            grad_layer.before_forward()

        # only set requires_grad=True for those parameters that need one
        requires_grad_before_list = [p.requires_grad for p in self.model.parameters()]
        for p, requires_grad in zip(self.model.parameters(), self.requires_grad_list):
            p.requires_grad = requires_grad

        old_training = self.model.training
        self.model.eval()
        X = feature_data.get_tensor(idxs)
        y = self.model(X)  # implicitly calls hooks that were set by l.before_forward()
        y.backward(torch.ones_like(y))
        with torch.no_grad():
            for p in self.model.parameters():
                p.grad = None

        self.model.train(old_training)

        data = ListFeatureData([layer_comp.pop_feature_data() for layer_comp in self.grad_layers])

        for p, value in zip(self.model.parameters(), requires_grad_before_list):
            p.requires_grad = value

        return data


def create_grad_feature_map(model: nn.Module, grad_layers: List[LayerGradientComputation],
                            use_float64: bool = False) -> FeatureMap:
    """
    Creates a feature map corresponding to phi_{grad} or phi_{ll}, depending on which layers are provided.
    :param model: Model to compute gradients of
    :param grad_layers: All layers of the model whose parameters we want to compute gradients of
    :param use_float64: Set to true if the gradient features should be converted to float64 after computing them
    :return: Returns a feature map corresponding to phi_{grad} for the given layers.
    """
    tfms = [ModelGradTransform(model, grad_layers)]
    if use_float64:
        tfms.append(ToDoubleTransform())
    return SequentialFeatureMap(SumFeatureMap([l.get_feature_map() for l in grad_layers]),
                                tfms)


# ----- Specific LayerGradientComputation implementation(s) for linear layers


class GeneralLinearGradientComputation(LayerGradientComputation):
    """
    Implements LayerGradientFeatures for general linear layers.
    It can also be used with the Neural Tangent Parameterization since it includes a weight factor and bias factor.
    (These are called sigma_w and sigma_b in the paper.)
    """
    def __init__(self, layer: nn.Module, in_features: int, out_features: int,
                 weight_factor: float = 1.0, bias_factor: float = 1.0):
        """
        :param layer: nn.Module object implementing a linear (fully-connected) layer,
        whose gradients should be computed.
        :param in_features: Input dimension of the layer.
        :param out_features: Output dimension of the layer.
        :param weight_factor: Factor sigma_w by which the weight matrix is multiplied in the forward pass.
        :param bias_factor: Factor sigma_w by which the bias is multiplied in the forward pass.
        """
        super().__init__(layer)
        self.in_features = in_features
        self.out_features = out_features
        self.weight_factor = weight_factor
        self.bias_factor = bias_factor
        self._input_data = None
        self._grad_output_data = None
        self._input_hook = None
        self._grad_output_hook = None

    def get_feature_map(self) -> FeatureMap:
        # gradients wrt to this layer are an outer product of the input and the output gradient,
        # so we can use a ProductFeatureMap
        # the +1 is for the bias
        return ProductFeatureMap([IdentityFeatureMap(n_features=self.in_features+1),
                                  IdentityFeatureMap(n_features=self.out_features)])

    def set_input_(self, value: torch.Tensor):
        # this is used to have a method to call in the hooks
        self._input_data = value

    def set_grad_output_(self, value: torch.Tensor):
        # this is used to have a method to call in the hooks
        self._grad_output_data = value

    def before_forward(self):
        # sets up hooks that store the input and grad_output
        self._input_hook = self.get_layer().register_forward_hook(
                lambda layer, inp, output, s=self: s.set_input_(inp[0].detach().clone()))
        self._grad_output_hook = self.get_layer().register_full_backward_hook(
                lambda layer, grad_input, grad_output, s=self: s.set_grad_output_(grad_output[0].detach().clone()))

    def pop_feature_data(self) -> FeatureData:
        # remove the hooks
        self._input_hook.remove()
        self._grad_output_hook.remove()
        # compute the adjusted input \tilde{x} from the paper
        inp = torch.cat([self.weight_factor * self._input_data,
                         self.bias_factor * torch.ones(self._input_data.shape[0], 1, device=self._input_data.device)],
                        dim=1)
        # feature data for the two IdentityFeatureMaps in the ProductFeatureMap, given by inputs and grad_outputs
        fd = ListFeatureData([TensorFeatureData(inp), TensorFeatureData(self._grad_output_data)])

        # allow to release memory earlier
        self._input_data = None
        self._grad_output_data = None

        return fd

