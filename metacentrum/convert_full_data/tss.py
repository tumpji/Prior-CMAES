import numpy as np
import math
import itertools

from abc import ABC, abstractmethod

class TSSbase(ABC):
    def __init__(self
                 , population
                 , mahalanobis_transf):
        self.population = population
        self.mahalanobis_transf = mahalanobis_transf

        assert isinstance(self.mahalanobis_transf, np.ndarray)
        assert len(self.mahalanobis_transf.shape) == 2
        assert self.mahalanobis_transf.shape[0] == self.mahalanobis_transf.shape[1]
        assert self.mahalanobis_transf.shape[0] > 0
        assert len(self.population.shape) == 2
        assert self.population.shape[1] == self.mahalanobis_transf.shape[0]

    @abstractmethod
    def __call__(self, archive_points, archive_evaluation, compute_distances=True, add_minimal_distances=False):
        assert len(archive_points.shape) == 2
        assert archive_points.shape[1] == self.population.shape[1]
        assert len(archive_evaluation.shape) == 1
        assert archive_evaluation.shape[0] == archive_points.shape[0]

        if compute_distances or add_minimal_distances:
            # (O, G, D)
            differences = self.population[np.newaxis, :, :] - archive_points[:, np.newaxis, :]
            # (O, G)
            distances = self.mahalanobis_distance(differences, self.mahalanobis_transf, do_not_square=True)
        else:
            distances = None

        if add_minimal_distances:
            return distances, {'minimal_distances': np.sqrt(np.amin(distances, axis=1))}
        else:
            return distances, {}

    @staticmethod
    def apply_mask_to_dictionary(mask, dictionary):
        for key in dictionary.keys():
            data = dictionary[key]
            if isinstance(data, np.ndarray) and len(data.shape) == 1 and len(data) == len(mask):
                dictionary[key] = data[mask]
            else:
                raise NotImplementedError(f'How to convert {key} ?')
        return dictionary

    @staticmethod
    def mahalanobis_distance(differences, mahalanobis_transf, do_not_square=False):
        N = differences.shape[:-1]
        D = differences.shape[-1]

        assert mahalanobis_transf.shape == (D, D)

        # centered_points = np.matmul(differences, mahalanobis_transf)
        centered_points = np.matmul(mahalanobis_transf, differences[..., np.newaxis])[..., 0]

        # # centered_points = np.matmul(differences, mahalanobis_transf)
        # # TODO check
        # centered_points = np.matmul(mahalanobis_transf, differences[..., np.newaxis])
        # centered_points = np.matmul(differences[..., np.newaxis, :], centered_points)[..., 0]
        # # pdb.set_trace()


        if do_not_square:
            return np.sum(np.square(centered_points), axis=-1)
        else:
            return np.sqrt(np.sum(np.square(centered_points), axis=-1))

class TSS0(TSSbase):
    # The hungry one...

    def __call__(self, archive_points, archive_evaluation, **kwargs):
        distances, other_info = super().__call__(archive_points, archive_evaluation, compute_distances=False, **kwargs)
        mask = np.ones(shape=(len(archive_points),), dtype=np.bool)
        return mask, other_info


class TSS2(TSSbase):
    def __init__(self
                 , population
                 , mahalanobis_transf
                 , maximum_distance
                 , maximum_number):
        super().__init__(population, mahalanobis_transf)

        self.maximum_distance = maximum_distance
        self.maximum_number = maximum_number
        assert isinstance(self.maximum_number, (int, float))
        assert isinstance(self.maximum_distance, (int, float))

    def __call__(self, archive_points, archive_evaluation, **kwargs):
        distances, other_info = super().__call__(archive_points, archive_evaluation, **kwargs)

        D = archive_points.shape[1]
        N = archive_points.shape[0]
        P = self.population.shape[0]

        # Firstly, do not limit number of neighbours
        mask = np.any(distances <= self.maximum_distance ** 2, axis=1)
        if np.sum(mask) <= self.maximum_number:
            return mask, self.apply_mask_to_dictionary(mask, other_info)

        # It's too much
        distances_indices = np.argsort(distances, axis=0)
        distances_sorted = np.stack(
            [distances[distances_indices[:, i], i] for i in range(P)], axis=1)
        distances_enable = distances_sorted <= self.maximum_distance ** 2

        # compute minimal size
        min_perin_size = math.floor(self.maximum_number / P)

        mask.fill(False)  # reuse mask array
        new_elems_ind = distances_indices[:min_perin_size, :][distances_enable[:min_perin_size, :]]
        mask[new_elems_ind] = True
        enabled_n = np.sum(mask)

        if enabled_n == self.maximum_number:
            return mask, self.apply_mask_to_dictionary(mask, other_info)
        else:
            # create another array
            new_mask = mask.copy()

        for perinp in itertools.count(start=min_perin_size):
            new_elems_ind = np.unique(distances_indices[perinp, :][distances_enable[perinp, :]])
            new_elems_ind = new_elems_ind[new_mask[new_elems_ind] == False]

            new_mask[new_elems_ind] = True

            enabled_o = enabled_n
            enabled_n += len(new_elems_ind)  # np.sum(new_mask)

            if enabled_n == self.maximum_number:
                return new_mask, self.apply_mask_to_dictionary(new_mask, other_info)
            elif enabled_n < self.maximum_number:
                mask, new_mask = new_mask, mask
                np.copyto(new_mask, mask)
            else:
                new_elems_eval = archive_evaluation[new_elems_ind]
                new_elems_ind_selection = np.argsort(new_elems_eval)[:self.maximum_number - enabled_o]
                selection = new_elems_ind[new_elems_ind_selection]

                mask[selection] = True
                return mask, self.apply_mask_to_dictionary(mask, other_info)
