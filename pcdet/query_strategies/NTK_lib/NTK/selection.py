from abc import ABC
import numpy as np

import torch
from typing import *

from .. import utils
from .features import *


class SelectionMethod:
    """
    Abstract base class for selection
    """
    def __init__(self):
        super().__init__()
        self.status = None  # can be used to report errors during selection

    def select(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_status(self) -> Optional:
        return self.status


class IterativeSelectionMethod(SelectionMethod):
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool, verbosity: int = 1,
                 **config):

        super().__init__()
        self.train_features = train_features
        self.pool_features = pool_features
        self.features = pool_features.concat_with(train_features) if sel_with_train else pool_features
        self.selected_idxs = []
        self.selected_arr = torch.zeros(self.pool_features.get_n_samples(), dtype=torch.bool,
                                        device=self.pool_features.get_device())
        self.with_train = sel_with_train
        self.verbosity = verbosity
        self.entropy_list = config['entropy_list']

    def prepare(self, n_adds: int):
        pass

    def get_scores(self) -> torch.Tensor:
        raise NotImplementedError()

    def add(self, new_idx: int):
        raise NotImplementedError()

    def get_next_idx(self) -> Optional[int]:
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        return torch.argmax(self.get_scores()).item()

    def select(self, batch_size: int) -> torch.Tensor:
        device = self.pool_features.get_device()

        self.prepare(batch_size + len(self.train_features) if self.with_train else batch_size)

        if self.with_train:
            # add training points first
            for i in range(len(self.train_features)):
                self.add(len(self.pool_features)+i)
                if (i+1) % 256 == 0 and self.verbosity >= 1:
                    print(f'Added {i+1} train samples to selection', flush=True)

        for i in range(batch_size):
            next_idx = self.get_next_idx()
            if next_idx is None or next_idx < 0 or next_idx >= len(self.pool_features) or self.selected_arr[next_idx]:
                # data selection failed
                # fill up with random remaining indices
                self.status = f'filling up with random samples because selection failed after n_selected = {len(self.selected_idxs)}'
                if self.verbosity >= 1:
                    print(self.status)
                n_missing = batch_size - len(self.selected_idxs)
                remaining_idxs = torch.nonzero(~self.selected_arr).squeeze(-1)
                new_random_idxs = remaining_idxs[torch.randperm(len(remaining_idxs), device=device)[:n_missing]]
                selected_idxs_tensor = torch.as_tensor(self.selected_idxs, dtype=torch.long,
                                                       device=torch.device(device))
                return torch.cat([selected_idxs_tensor, new_random_idxs], dim=0)
            else:
                self.add(next_idx)
                self.selected_idxs.append(next_idx)
                self.selected_arr[next_idx] = True
        return torch.as_tensor(self.selected_idxs, dtype=torch.long, device=torch.device(device))

class ForwardBackwardSelectionMethod(IterativeSelectionMethod, ABC):
    """
    for BAIT
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool, verbosity: int = 1,
                 overselection_factor: float = 1.0, **config):
        super().__init__(pool_features=pool_features, train_features=train_features, sel_with_train=sel_with_train,
                         verbosity=verbosity, **config)
        self.overselection_factor = overselection_factor

    def get_scores_backward(self) -> torch.Tensor:
        raise NotImplementedError()

    def get_next_idx_backward(self) -> Optional[int]:
        scores = self.get_scores_backward().clone()
        return torch.argmax(scores).item()

    def remove(self, idx: int):
        raise NotImplementedError()

    def select(self, batch_size: int) -> torch.Tensor:
        overselect_batch_size = min(round(self.overselection_factor * batch_size), len(self.pool_features))
        overselect_batch = super().select(overselect_batch_size)
        if batch_size == overselect_batch_size:
            return overselect_batch
        if self.status is not None:
            # selection failed
            return overselect_batch[:batch_size]

        for i in range(overselect_batch_size-batch_size):
            next_idx = self.get_next_idx_backward()
            if next_idx is None or next_idx < 0 or next_idx >= len(self.selected_idxs):
                self.status = f'removing the latest overselected samples because the backward step failed '\
                              f'after removing {i} samples'
                if self.verbosity >= 1:
                    print(self.status)
                self.selected_idxs = self.selected_idxs[:batch_size]
                break
            else:
                self.remove(next_idx)
                self.selected_arr[self.selected_idxs[next_idx]] = False
                del self.selected_idxs[next_idx]
        device = self.pool_features.get_device()
        return torch.as_tensor(self.selected_idxs, dtype=torch.long, device=torch.device(device))


class RandomSelectionMethod(SelectionMethod):
    def __init__(self, pool_features: Features, **config):
        super().__init__()
        self.pool_features = pool_features

    def select(self, batch_size: int) -> torch.Tensor:
        device = self.pool_features.get_device()
        generator = torch.Generator(device=device)
        return torch.randperm(self.pool_features.get_n_samples(),
                              device=self.pool_features.get_device(),
                              generator=generator)[:batch_size]


class MaxDetSelectionMethod(IterativeSelectionMethod):
    """
    as the maxdet part in kernel coding rate
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 noise_sigma: float = 0.0, **config):
      
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.entropy_list = config['entropy_list']
        self.entropy_sigma = config['entropy_sigma']
        self.noise_sigma = noise_sigma
        self.diag = self.features.get_kernel_matrix_diag() + self.noise_sigma**2 + self.entropy_sigma * torch.stack(self.entropy_list)
        self.l = None
        self.n_added = 0

    def prepare(self, n_adds: int):
        n_total = n_adds if self.l is None else n_adds + self.l.shape[1]
        new_l = torch.zeros(len(self.features), n_total, device=torch.device(self.features.get_device()), dtype=self.diag.dtype)
        if self.l is not None:
            new_l[:, :self.l.shape[1]] = self.l
        self.l = new_l

    def get_scores(self) -> torch.Tensor:
        return self.diag

    def get_next_idx(self) -> Optional[int]:
        # print('max score:', torch.max(self.get_scores()).item())
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        new_idx = torch.argmax(scores).item()
        if scores[new_idx] <= 0.0:
            print(f'Selecting index {len(self.selected_idxs)+1}: new diag entry nonpositive')
            # print(f'diag entry: {self.get_scores()[new_idx]}')
            # diagonal is zero or lower, would cause numerical errors afterwards
            return None
        # print(self.diag[new_idx])
        return new_idx

    def add(self, new_idx: int):
        # print('new_idx:', new_idx)
        l = None if self.l is None else self.l[:, :self.n_added]
        lTl = 0.0 if l is None else l.matmul(l[new_idx, :])
        mat_col = self.features[new_idx].get_kernel_matrix(self.features).squeeze(0)
        if self.noise_sigma > 0.0:
            mat_col[new_idx] += self.noise_sigma**2
        update = (1.0 / torch.sqrt(self.diag[new_idx])) * (mat_col - lTl)
        # shape: len(self.features)
        self.diag -= update ** 2
        # shape: (n-1) x len(self.features)
        self.l[:, self.n_added] = update
        # self.l = update[:, None] if self.l is None else torch.cat([self.l, update[:, None]], dim=1)
        # print('trace(ll^T):', (self.l**2).sum())

        self.n_added += 1

        self.diag[new_idx] = -np.Inf   # ensure that the index is not selected again


        
        
class BaitFeatureSpaceSelectionMethod(ForwardBackwardSelectionMethod):

    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 noise_sigma: float = 0.0, **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.noise_sigma = noise_sigma
        self.diag = self.features.get_kernel_matrix_diag().clone()
        self.feature_matrix = self.features.get_feature_matrix().clone()

        self.feature_cov_matrix = self.feature_matrix.t() @ self.feature_matrix
        if not sel_with_train:
            train_feature_matrix = train_features.get_feature_matrix()
            self.feature_cov_matrix += train_feature_matrix.t() @ train_feature_matrix
        self.scores_numerator = torch.einsum('ij,ji->i',
                                             self.feature_matrix,
                                             self.feature_cov_matrix @ self.feature_matrix.t())

    def get_scores(self) -> torch.Tensor:
        return self.scores_numerator / (self.diag + self.noise_sigma**2 + 1e-8)

    def get_next_idx(self) -> Optional[int]:
        scores = self.get_scores()
        scores[self.selected_idxs] = -np.Inf
        new_idx = torch.argmax(scores).item()
        if scores[new_idx].item() <= 0.0:
            if self.verbosity >= 1:
                print(f'Selecting index {len(self.selected_idxs)+1}: new score nonpositive')
            return None
        return new_idx

    def add(self, new_idx: int):
        diag_entry = self.diag[new_idx] + self.noise_sigma**2
        sqrt_diag_entry = torch.sqrt(diag_entry)
        beta = 1.0 / (sqrt_diag_entry * (sqrt_diag_entry + self.noise_sigma))
        phi_x = self.feature_matrix[new_idx]
        dot_prods = self.feature_matrix.matmul(phi_x)
        dot_prods_sq = dot_prods**2

        # update scores_numerator
        cov_phi = self.feature_cov_matrix @ phi_x
        phi_cov_phi = self.scores_numerator[new_idx].clone()
        # phi_cov_phi = torch.dot(phi_x, cov_phi)
        mult = 1/diag_entry
        self.scores_numerator -= 2 * mult * (self.feature_matrix @ cov_phi) * dot_prods
        self.scores_numerator += mult**2 * phi_cov_phi * dot_prods_sq
        # update feature_cov_matrix
        cov_phi_phit = cov_phi[:, None] * phi_x[None, :]
        phi_phit = phi_x[:, None] * phi_x[None, :]
        self.feature_cov_matrix -= beta * (cov_phi_phit + cov_phi_phit.t())
        self.feature_cov_matrix += beta**2 * phi_cov_phi * phi_phit
        # update feature matrix
        self.feature_matrix -= dot_prods[:, None] * (beta * phi_x[None, :])
        # update diag
        self.diag -= dot_prods_sq / diag_entry


    def get_scores_backward(self) -> torch.Tensor:
        den = (self.diag[self.selected_idxs] - self.noise_sigma ** 2)
        num = torch.clamp(self.scores_numerator[self.selected_idxs], min=0.0)
        scores = num / den
        scores[den >= 0.0] = -np.Inf
        return scores

    def get_next_idx_backward(self) -> Optional[int]:
        scores = self.get_scores_backward()
        new_idx = torch.argmax(scores).item()
        new_score = scores[new_idx].item()
        if new_score == -np.Inf or new_score >= 0.0:
            if self.verbosity >= 1:
                print(f'Backwards selecting index {len(self.selected_idxs)}: new score positive')
            return None
        return new_idx

    def remove(self, idx: int):
        features_idx = self.selected_idxs[idx]
        diag_entry = self.noise_sigma ** 2 - self.diag[features_idx]
        diag_entry = torch.clamp(diag_entry, min=1e-15)
        sqrt_diag_entry = torch.sqrt(diag_entry)
        beta = 1.0 / (sqrt_diag_entry * (sqrt_diag_entry + self.noise_sigma))
        phi_x = self.feature_matrix[features_idx]
        dot_prods = self.feature_matrix.matmul(phi_x)
        dot_prods_sq = dot_prods**2

        # update scores_numerator
        cov_phi = self.feature_cov_matrix @ phi_x
        phi_cov_phi = self.scores_numerator[features_idx].clone()
        mult = 1 / diag_entry
        self.scores_numerator += 2 * mult * (self.feature_matrix @ cov_phi) * dot_prods
        self.scores_numerator += mult ** 2 * phi_cov_phi * dot_prods_sq
        # update feature_cov_matrix
        cov_phi_phit = cov_phi[:, None] * phi_x[None, :]
        phi_phit = phi_x[:, None] * phi_x[None, :]
        self.feature_cov_matrix += beta * (cov_phi_phit + cov_phi_phit.t())
        self.feature_cov_matrix += beta ** 2 * phi_cov_phi * phi_phit
        # update feature matrix
        self.feature_matrix += dot_prods[:, None] * (beta * phi_x[None, :])
        # update diag
        self.diag += dot_prods_sq / diag_entry

