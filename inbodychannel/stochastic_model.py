import numpy as np
import h5py
import os

from sklearn import mixture
from typing import Dict, Union, Tuple, List


class StochasticChannelModel:

    def __init__(self,
                 model_name: str):
        """
        Initialize the StochasticChannelModel.

        :type model_name: str
        :param model_name:  Name of the voxel model to use the stochastic model from. Can be either 'AustinMan',
                            'AustinWoman', 'Donna', 'Golem', 'Helga', 'Irene', or 'VisibleHuman'

        """
        self.all_cluster_indices = ['00', '01', '02', '03',
                                    '10', '11', '12', '13',
                                    '20', '21', '22', '23',
                                    '30', '31', '32', '33']
        self.model_name = model_name
        self.cluster = {}  # dictionary to store all the GMM of the different clusters

        # load the HDF5 file with the parameters of the stochastic model
        current_directory = os.path.dirname(__file__)
        filename = os.path.join(current_directory, 'stochastic_model_%s.hdf5' % model_name)
        with h5py.File(filename, 'r') as f:
            for cluster in self.all_cluster_indices:
                # create the GMM for each cluster
                gmm = mixture.GaussianMixture(n_components=5, covariance_type='full')
                # we need to trick the GaussianMixture to let it think it was trained..
                gmm.fit(np.random.rand(10, 5))
                # .. and then load the saved parameters from previous trainings
                gmm.weights_ = f['cluster%s/gmm/weights' % cluster][()]
                gmm.means_ = f['cluster%s/gmm/means' % cluster][()]
                gmm.covariances_ = f['cluster%s/gmm/covariances' % cluster][()]

                # add the gmm to the cluster
                self.cluster[cluster] = {}
                self.cluster[cluster]['GMM'] = gmm

                # add the constraint on alpha for each cluster
                self.cluster[cluster]['max_alpha'] = f['cluster%s/constraints/max_alpha' % cluster][()]

    def _delete_implausible_samples(self, cluster: str, samples: np.ndarray) -> np.ndarray:
        """
        Delete all samples that do not pass the plausibility test.

        :param cluster: string representing the current cluster.
        :param samples: input samples
        :return:
        """

        # delete samples with a distance smaller 10mm
        index = np.where(samples[:, 5] <= 0.01)
        samples = np.delete(samples, index, axis=0)

        # delete samples with values greater -0.3 for a, b or c (as this possibly leads to path gain > 1)
        for k in range(3):
            i = np.where(samples[:, k] > -0.3)
            samples = np.delete(samples, i, axis=0)

        # delete samples where alpha is greater than the maximum alpha found in approximation
        alpha_max = self.cluster[cluster]['max_alpha']
        alpha_too_fast = np.where(samples[:, 3] > alpha_max)
        samples = np.delete(samples, alpha_too_fast, axis=0)

        return samples

    def sample(self, n_samples: Union[int, List[int]], clusters: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Draw random samples from the stochastic channel model

        :param clusters: Clusters for which the PL should be calculated (from these clusters samples will be drawn).
                         If set to None all clusters will be considered. The notation is ['01', '02', '11', ...].
        :param n_samples: The total number of drawn samples from all clusters. This can also be a list giving
                          in each entry the number of samples for each cluster given in 'clusters'
        :return: a tuple of
                    1. an array where each row contains a drawn sample of the 6 parameters (a, b, c, alpha, beta, d)
                    2. a list containing the cluster label for each set of samples.
        """

        if clusters is None:
            clusters = self.all_cluster_indices

        if isinstance(n_samples, int):
            # n_samples gives the total number of samples
            # divide them evenly onto all clusters
            samples_per_cluster = int(np.around(n_samples/len(clusters)))
            n_samples_list = [samples_per_cluster for c in clusters]
            # in case the there are some number of samples missing add them
            for i in range(n_samples % len(clusters)):
                n_samples_list[i] += 1
            # if the number of samples is smaller than the number of clusters all entries in n_samples_list are by
            # one too large
            if n_samples < len(clusters):
                n_samples_list = list(np.array(n_samples_list) - 1)
        else:
            if len(n_samples) != len(clusters):
                raise ValueError("Length of 'n_samples' has to be the same as length of 'clusters'")
            n_samples_list = n_samples

        # generate empty matrix to store samples from each cluster
        sample_matrix = np.zeros((int(np.sum(n_samples)), 6))
        cluster_mapping = []
        start_index = 0

        # draw sample for each cluster
        for i, cl in enumerate(clusters):
            gmm_cl = self.cluster[cl]['GMM']
            n_observations = int(n_samples_list[i])
            if n_observations == 0:
                continue

            samples = gmm_cl.sample(n_observations)[0]

            # delete all samples that do not fulfill the plausibility test
            samples = self._delete_implausible_samples(cl, samples)

            while samples.shape[0] < n_observations:
                samples_add = gmm_cl.sample(n_observations - samples.shape[0])[0]
                samples = np.vstack((samples, samples_add))
                # delete all samples that do not fulfill the plausibility test
                samples = self._delete_implausible_samples(cl, samples)

            sample_matrix[start_index:start_index + n_observations, :] = samples
            cluster_mapping = cluster_mapping + [cl] * n_observations
            start_index += n_observations

        return sample_matrix, cluster_mapping

    def path_loss(self, n_samples: Union[int, List[int]],
                  clusters: List[str] = None,
                  f_start: float = 3.1e9,
                  f_end: float = 4.8e9,
                  n_frequency_samples: int = 2048,
                  parameter_matrix: np.array = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the path loss, assuming that the transmit power is distributed uniformly between f_start and f_end.

        :param parameter_matrix: A matrix already containing row-wise parameters for the transfer functions.
                                 If None then parameters will be drawn randomly
        :param clusters: clusters for which the PL should be calculated (from these clusters samples will be drawn).
                         If set to None all clusters will be considered.
        :param n_samples: The total number of drawn samples from all clusters. This can also be a list giving
                          in each entry the number of samples for each cluster given in 'clusters'
        :param f_start: lower frequency limit for path loss calculation
        :param f_end: upper frequency limit for path loss calculation
        :param n_frequency_samples: number of frequency samples used for calculation of the transfer function
        :return: a tuple with the path loss in the first entry and the corresponding distances (in meter) in the second.
        """

        if parameter_matrix is None:
            # get random samples
            sample_matrix = self.sample(clusters=clusters, n_samples=n_samples)[0]
        else:
            sample_matrix = parameter_matrix

        # matrix to store the path losses and the corresponding distances
        path_loss_result = np.zeros((np.shape(sample_matrix)[0],))
        distance = np.zeros((np.shape(sample_matrix)[0],))

        # frequency vector and bandwidth
        f = np.linspace(f_start, f_end, n_frequency_samples)
        bandwidth = f_end - f_start

        for row in range(np.shape(sample_matrix)[0]):
            tf_parameters = sample_matrix[row, 0:5]
            transfer_function = self.transfer_function(f, *tf_parameters)
            path_gain = np.trapz(abs(transfer_function) ** 2, x=f, axis=0) / bandwidth
            path_loss = 1 / path_gain

            path_loss_result[row] = path_loss
            # also store the distance
            distance[row] = sample_matrix[row, 5]

        return path_loss_result, distance

    @staticmethod
    def transfer_function(f, a, b, c, alpha, beta):
        """
        Calculate the value of the transfer function vs. frequency.
        :param f:  Frequency
        :param a:
        :param b:
        :param c:
        :param alpha:
        :param beta:
        :return:
        """
        return (np.exp(f * a * 1e-9) + np.exp(f * b * 1e-9) + np.exp(f * c * 1e-9)) \
               * np.exp(1j * (f * alpha * 1e-9 + beta))

    @staticmethod
    def impulse_response(transfer_function: np.array, threshold_dB, equivalent_baseband):
        """
        Calculate impulse response from transfer function. The impulse response is truncated at a value
        threshold_dB smaller than the maximum value.

        :param transfer_function: transfer function
        :param threshold_dB: threshold after which the impulse response is cut
        :param equivalent_baseband: dictionary containing the bandwidth 'B' and the center frequency 'fc' when
                calculating the PDP in the equivalent baseband; if None then the PDP will be computed in the passband
                from frequency 0 to 10GHz (sampling rate 20 GHz)
        :return:
        """
        transfer_function.shape = (-1, 1)

        if equivalent_baseband is not None:
            # shift to equivalent baseband
            tf_bb = np.fft.ifftshift(transfer_function)
            impulse_response = np.fft.ifft(tf_bb, axis=0)
        else:
            impulse_response = np.fft.irfft(transfer_function, axis=0)

        # determine the value of the maximum - threshold_dB (amplitudes -> factor 20!)
        threshold_value = np.max(np.abs(impulse_response)) * 10 ** (- threshold_dB / 20)
        # find the index where the threshold is first met
        threshold_index = np.argmax(np.abs(impulse_response) > threshold_value)
        if threshold_index == impulse_response.size - 1:
            start_index = 0
        else:
            start_index = threshold_index
        impulse_response = impulse_response[start_index:]
        return impulse_response

    def calculate_power_delay_profile(self, parameter_matrix: np.array = None, clusters: np.array = None,
                                      n_samples: np.array = None,
                                      n_frequency_samples: int = 2048,
                                      threshold_dB: float = 30,
                                      equivalent_baseband: Dict = None):
        """
        Calculate the power delay profile in the equivalent baseband

        :param parameter_matrix: a matrix already containing row-wise parameters for the transfer functions.
                                 If None then parameters will be drawn randomly
        :param clusters: clusters for which the PDP should be calculated (from these clusters samples will be drawn)
        :param n_samples: the number of samples drawn for all clusters
        :param n_frequency_samples: number of frequency samples used for calculation of transfer function
        :param threshold_dB: threshold after which the impulse response is cut and up to which the PDP is considered
        :param equivalent_baseband: dictionary containing the bandwidth 'B' and the center frequency 'fc' when
                calculating the PDP in the equivalent baseband; if None then the PDP will be computed in the passband
                from frequency 0 to 10GHz (sampling rate 20 GHz)
        :return:
        """

        # sampling rate cannot be chosen arbitrarily when converting to eq. baseband
        if equivalent_baseband is not None:
            f_sample = equivalent_baseband['B']
            f_start = equivalent_baseband['fc'] - equivalent_baseband['B']/2
            f_end = equivalent_baseband['fc'] + equivalent_baseband['B']/2
        else:
            f_sample = 20e9
            f_start = 0
            f_end = f_sample / 2

        # if no parameter matrix is given draw random samples
        if parameter_matrix is None:
            sample_matrix = self.sample(clusters=clusters, n_samples=n_samples)[0]
        else:
            sample_matrix = parameter_matrix

        # get frequency vector
        f = np.linspace(f_start, f_end, n_frequency_samples)

        for row in range(np.shape(sample_matrix)[0]):
            parameters_tf = sample_matrix[row, 0:5]
            transfer_function = self.transfer_function(f, *parameters_tf)

            # get impulse response
            ir = self.impulse_response(transfer_function, threshold_dB, equivalent_baseband=equivalent_baseband)
            if row == 0:
                power_delay_profile = abs(ir) ** 2
            else:
                diff_size = ir.size - power_delay_profile.size
                if diff_size > 0:
                    power_delay_profile = np.pad(power_delay_profile, [(0, diff_size), (0, 0)],
                                                 mode='constant', constant_values=0)
                elif diff_size < 0:
                    diff_size = -diff_size
                    ir = np.pad(ir, [(0, diff_size), (0, 0)],
                                mode='constant', constant_values=0)
                power_delay_profile += abs(ir) ** 2

        # calculate the average
        total_length = sample_matrix.shape[0]  # number of considered impulse responses
        power_delay_profile = power_delay_profile / total_length
        power_delay_profile.shape = (-1,)

        # ** cut the power delay profile after it has reached a decay of more than threshold_dB dB
        # find the maximum
        pdp_max_ind = np.argmax(power_delay_profile)
        pdp_max = power_delay_profile[pdp_max_ind]
        # determine the threshold value
        threshold_value = pdp_max * 10 ** (-threshold_dB / 10)
        # find the index where the threshold is first met after the PDP maximum
        threshold_index = np.argmax(power_delay_profile[pdp_max_ind::] < threshold_value)
        power_delay_profile = power_delay_profile[0:(threshold_index + pdp_max_ind)]

        tau = np.arange(0, power_delay_profile.size) / f_sample
        tau.shape = (-1,)

        # *** calculate the RMS delay spread
        tau_mean = np.trapz(power_delay_profile * tau, tau) / \
                   np.trapz(power_delay_profile, tau)
        tau_rms = np.sqrt(np.trapz((tau - tau_mean) ** 2 * power_delay_profile, tau) /
                          np.trapz(power_delay_profile, tau))

        # adjust the values such that the first entry of the PDP is 0dB
        power_delay_profile /= pdp_max

        return power_delay_profile, tau, tau_rms, tau_mean
