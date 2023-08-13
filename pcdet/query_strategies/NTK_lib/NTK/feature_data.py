from typing import *
import torch

from .. import utils


def torch_cat(tensors: List[torch.Tensor], dim: int):
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


class Indexes:

    def __init__(self, n_samples: int, idxs: Optional[Union[torch.Tensor, slice, int, 'Indexes']]):
        self.n_samples = n_samples
        if idxs is None:
            self.idxs = slice(0, n_samples)
        elif isinstance(idxs, Indexes):
            self.idxs = idxs.idxs
        elif isinstance(idxs, int):
            self.idxs = slice(idxs, idxs+1)
        elif isinstance(idxs, slice):
            if idxs.step is not None and idxs.step != 1:
                raise ValueError(f'Cannot handle slices with step size other than 1')
            start = 0 if idxs.start is None else idxs.start + (0 if idxs.start >= 0 else n_samples)
            stop = n_samples if idxs.stop is None else idxs.stop + (0 if idxs.stop >= 0 else n_samples)
            if stop <= start:
                raise ValueError(f'stop <= start not allowed for slices')
            self.idxs = slice(start, stop)
        elif isinstance(idxs, torch.Tensor):
            self.idxs = idxs
        else:
            raise ValueError(f'Cannot handle index type {type(idxs)}')

        if isinstance(self.idxs, slice):
            self.n_idxs = self.idxs.stop - self.idxs.start
        else:
            self.n_idxs = len(self.idxs)

    def __len__(self):
     
        return self.n_idxs

    def compose(self, other: 'Indexes') -> 'Indexes':
 
        if isinstance(self.idxs, torch.Tensor):
            new_idxs = self.idxs[other.idxs]
        elif isinstance(self.idxs, slice):
            if isinstance(other.idxs, torch.Tensor):
                new_idxs = other.idxs + self.idxs.start
            elif isinstance(other.idxs, slice):
                new_idxs = slice(self.idxs.start + other.idxs.start, self.idxs.start + other.idxs.stop)
            else:
                raise RuntimeError('other.idxs is neither slice nor torch.Tensor')
        else:
            raise RuntimeError('self.idxs is neither slice nor torch.Tensor')

        return Indexes(len(self), new_idxs)

    def get_idxs(self):
       
        return self.idxs

    def split_by_sizes(self, sample_sizes: List[int]) -> Iterable[Tuple[int, 'Indexes']]:
   
        if isinstance(self.idxs, slice):
            start = self.idxs.start
            stop = self.idxs.stop
            for i, sz in enumerate(sample_sizes):
                if start < sz and stop > 0:
                    yield i, Indexes(sz, slice(max(start, 0), min(stop, sz)))
                start -= sz
                stop -= sz
        else:
            raise NotImplementedError('indexing splitted data with a list of indexes is currently not supported')

    def is_all_slice(self) -> bool:
      
        return isinstance(self.idxs, slice) and self.idxs.start == 0 and self.idxs.stop == self.n_samples


class FeatureData:
    """
    Abstract base class for classes that represent data that serves as input to feature maps.
    """
    def __init__(self, n_samples: int, device: str, dtype):

        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype

    def get_n_samples(self) -> int:
        """
        :return: Returns the number of samples.
        """
        return self.n_samples

    def get_device(self) -> str:
        """
        :return: Returns the device that the data is on.
        """
        return self.device

    def get_dtype(self) -> Any:
        """
        :return: Returns the (torch) dtype that the feature data has.
        """
        return self.dtype

    def __len__(self) -> int:
        """
        :return: Returns the number of samples.
        """
        return self.get_n_samples()

    def __getitem__(self, idxs: Union[torch.Tensor, slice, int, 'Indexes']):

        idxs = Indexes(self.get_n_samples(), idxs)
        if idxs.is_all_slice():
            return self
        return SubsetFeatureData(self, idxs)

    def simplify(self, idxs: Optional[Union[torch.Tensor, slice, int, Indexes]] = None) -> 'FeatureData':

        idxs = Indexes(self.get_n_samples(), idxs)
        return self.simplify_impl_(idxs)

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':

        raise NotImplementedError()

    def simplify_multi_(self, feature_data_list: List['FeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':

        raise NotImplementedError()

    def iterate(self, idxs: Indexes) -> Iterable[Tuple[Indexes, 'FeatureData']]:

        yield idxs, self  # default implementation

    def __iter__(self) -> Iterable[Tuple[Indexes, 'FeatureData']]:

        return self.iterate(Indexes(n_samples=self.get_n_samples(), idxs=None))


    def get_tensor(self, idxs: Optional[Union[torch.Tensor, slice, int, Indexes]] = None) -> torch.Tensor:

        idxs = Indexes(self.get_n_samples(), idxs)
        return self.get_tensor_impl_(idxs)

    def get_tensor_impl_(self, idxs: Indexes) -> torch.Tensor:

        return torch_cat([feature_data.get_tensor(sub_idxs) for sub_idxs, feature_data in self.iterate(idxs)], dim=0)

    def cast_to(self, dtype) -> 'FeatureData':

        raise NotImplementedError()

    def to_indexes(self, idxs: Optional[Indexes]) -> Indexes:
    
        return idxs if idxs is not None else Indexes(self.get_n_samples(), None)


class EmptyFeatureData(FeatureData):

    def __init__(self, device: str, dtype):
        super().__init__(n_samples=0, device=device, dtype=dtype)

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return self

    def simplify_multi_(self, feature_data_list: List['EmptyFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        return EmptyFeatureData(device=self.device, dtype=self.dtype)

    def cast_to(self, dtype) -> 'FeatureData':
        return self


class TensorFeatureData(FeatureData):

    def __init__(self, data: torch.Tensor):
        """
        :param data: Tensor of shape [n_samples, ...], usually [n_samples, n_features]
        """
        super().__init__(n_samples=data.shape[-2], device=data.device, dtype=data.dtype)
        self.data = data

    def get_tensor_impl_(self, idxs: Indexes) -> torch.Tensor:
        return self.data[idxs.get_idxs()]

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return TensorFeatureData(self.get_tensor_impl_(idxs))

    def simplify_multi_(self, feature_data_list: List['TensorFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        return TensorFeatureData(torch_cat([fd.data[idxs.get_idxs()]
                                            for fd, idxs in zip(feature_data_list, idxs_list)], dim=0))

    def cast_to(self, dtype) -> 'FeatureData':
        return TensorFeatureData(self.data.type(dtype))


class ConcatFeatureData(FeatureData):
    
    def __init__(self, feature_data_list: List[FeatureData]):
        sample_sizes = [fd.get_n_samples() for fd in feature_data_list]
        super().__init__(n_samples=sum(sample_sizes), device=feature_data_list[0].get_device(),
                         dtype=feature_data_list[0].get_dtype())
        self.feature_data_list = feature_data_list
        self.sample_sizes = sample_sizes

    def iterate(self, idxs: Indexes) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        for i, sub_idxs in idxs.split_by_sizes(self.sample_sizes):
            for sub_idxs_2, feature_data in self.feature_data_list[i].iterate(sub_idxs):
                yield sub_idxs_2, feature_data

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        simplified = [self.feature_data_list[i].simplify(sub_idxs)
                      for i, sub_idxs in idxs.split_by_sizes(self.sample_sizes)]
        simplified = [fd for fd in simplified if not isinstance(fd, EmptyFeatureData)]
        if len(simplified) == 1:
            return simplified[0]
        elif len(simplified) == 0:
            return EmptyFeatureData(device=self.device)
        else:
            if utils.all_equal([type(fd) for fd in simplified]):
                return simplified[0].simplify_multi_(simplified, [Indexes(fd.get_n_samples(), None) for fd in simplified])
            else:
                raise RuntimeError(
                    'Attempting to concatenate different-typed simplified FeatureData objects during simplify')

    def simplify_multi_(self, feature_data_list: List['ConcatFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        if len(feature_data_list) == 0:
            return EmptyFeatureData(device=self.device)
        # use simplify() to implement simplify_multi_()
        return ConcatFeatureData([SubsetFeatureData(fd, idxs) for cfd, idxs in zip(feature_data_list, idxs_list)
                                  for fd in cfd.feature_data_list]).simplify()

    def cast_to(self, dtype) -> 'FeatureData':
        return ConcatFeatureData([fd.cast_to(dtype) for fd in self.feature_data_list])


class SubsetFeatureData(FeatureData):

    def __init__(self, feature_data: FeatureData, idxs: Indexes):
 
        super().__init__(n_samples=len(idxs), device=feature_data.get_device(), dtype=feature_data.get_dtype())
        self.idxs = idxs
        self.feature_data = feature_data

    def iterate(self, idxs: Optional[Indexes] = None) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        for sub_idxs, feature_data in self.feature_data.iterate(self.idxs.compose(idxs)):
            yield sub_idxs, feature_data

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return self.feature_data.simplify(self.idxs.compose(idxs))

    def simplify_multi_(self, feature_data_list: List['SubsetFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        return ConcatFeatureData([fd.simplify(idxs) for fd, idxs in zip(feature_data_list, idxs_list)]).simplify()

    def cast_to(self, dtype) -> 'FeatureData':
        return SubsetFeatureData(self.feature_data.cast_to(dtype), self.idxs)

class ListFeatureData(FeatureData):  # does not concatenate along batch dimension

    def __init__(self, feature_data_list: List[FeatureData]):

        super().__init__(n_samples=feature_data_list[0].get_n_samples(), device=feature_data_list[0].get_device(),
                         dtype=feature_data_list[0].get_dtype())
        self.feature_data_list = feature_data_list

    def get_tensor(self, idxs: Optional[Union[torch.Tensor, slice, int, Indexes]] = None) -> torch.Tensor:
        raise NotImplementedError(
            'get_tensor() cannot be called on ListFeatureData since it would need to return multiple tensors')

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return ListFeatureData([fd.simplify(idxs) for fd in self.feature_data_list])

    def simplify_multi_(self, feature_data_list: List['ListFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        if len(feature_data_list) == 0:
            return EmptyFeatureData(device=self.device, dtype=self.dtype)
        return ListFeatureData([ConcatFeatureData([fd.feature_data_list[i][idxs]
                                                   for fd, idxs in zip(feature_data_list, idxs_list)]).simplify()
                                for i in range(len(feature_data_list[0].feature_data_list))])

    def cast_to(self, dtype) -> 'FeatureData':
        return ListFeatureData([fd.cast_to(dtype) for fd in self.feature_data_list])
