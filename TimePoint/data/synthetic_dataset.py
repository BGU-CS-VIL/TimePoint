import torch
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm.auto import tqdm


class SynthAlign(Dataset):
    def __init__(
        self,
        data_types,
        probs,
        data_length=1000,
        total_samples=100,
        cache_size=100,
        transform=None,
    ):
        """
        :param data_types: List of functions that generate different time series data types.
        :param probs: Probabilities with which each data type is chosen.
        :param data_length: The length of each time series data.
        :param total_samples: The total number of samples in the dataset.
        :param cache_size: Number of samples to pre-generate and cache.
        :param transform: Optional transform to be applied on a sample.
        """
        assert len(data_types) == len(
            probs
        ), "Each data type must have a corresponding probability."
        self.data_types = data_types
        self.probs = probs
        self.data_length = data_length
        self.total_samples = total_samples
        self.transform = transform
        self.cache_size = min(
            cache_size, total_samples
        )  # Ensure cache size does not exceed total samples
        if self.cache_size > 0:
            self.cache = self._generate_cache()
        self.y = np.zeros(total_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < self.cache_size:
            sample, keypoints = self.cache[idx]
        else:
            sample, keypoints = self._generate_sample()

        if self.transform:
            sample = self.transform(sample)

        outputs = {"signals": sample, "keypoints": keypoints}

        return outputs

    def _generate_sample(self):
        # Sample from 1 to len(self.data_types) data types
        num_data_types = random.randint(1, len(self.data_types))
        selected_data_types = np.random.choice(
            self.data_types, size=num_data_types, p=self.probs, replace=False
        )

        # Initialize an empty array for the final sample and keypoints
        sample = np.zeros(self.data_length)
        keypoints = np.zeros(self.data_length, dtype=int)

        # With a probability, replace selected data types with simple patterns
        if random.random() > 0.5:
            selected_data_types = [self.simple_patterns]

        # Compose selected data types
        for data_type in selected_data_types:
            # print(data_type, data_type(self.data_length).shape)
            data, kp = data_type(self.data_length)
            sample += data
            keypoints += kp

        # Add linear line with probability 0.5
        if random.random() > 0.5:
            data, kp = self.linear_line(self.data_length)
            sample += data + np.random.normal(0, 0.01, self.data_length)

        # Randomly flip a section of the signal
        if random.random() > 0.5:
            # Select a section of the signal and flip it
            start = random.randint(0, self.data_length // 2)
            end = random.randint(self.data_length // 2, self.data_length)
            sample[start:end] = -sample[start:end]
            keypoints[start] = 1
            keypoints[end - 1] = 1

        # Convert to torch tensors
        sample = torch.tensor(sample).unsqueeze(0).float()  # Shape: (1, data_length)
        keypoints = np.clip(keypoints, 0, 1)
        keypoints = torch.tensor(keypoints).float()  # Shape: (data_length, )
        keypoints[0], keypoints[-1] = 0, 0

        return sample, keypoints

    def _generate_cache(self):
        print(f"Generating cache of {self.cache_size} samples...")
        return [
            self._generate_sample()
            for _ in tqdm(range(self.cache_size), desc="Caching")
        ]

    @staticmethod
    def sine_wave_composition(data_length, min_waves=1, max_waves=10):
        num_waves = random.randint(min_waves, max_waves)
        sample = np.zeros(data_length)
        t = np.linspace(0, 1, data_length, endpoint=False)
        for _ in range(num_waves):
            frequency = random.uniform(0.1, 15)  #  frequency range 0.1-15
            amplitude = random.uniform(0.1, 1)  # amplitude range
            phase = random.uniform(0, 2 * np.pi)
            sample += amplitude * np.sin(2 * np.pi * frequency * t + phase)

        # Find keypoints (local maxima and minima)
        keypoints = np.zeros(data_length, dtype=int)
        # Compute numerical derivative
        derivative = np.gradient(sample)
        # Find zero crossings of the derivative
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
        keypoints[zero_crossings] = 1

        return sample, keypoints

    @staticmethod
    def block_wave(data_length):
        block_size = random.randint(10, 100)
        amplitude = random.uniform(-4, 4)
        data = np.zeros(data_length)
        keypoints = np.zeros(data_length, dtype=int)
        for start in range(0, data_length, block_size * 2):
            end = min(start + block_size, data_length)
            data[start:end] = amplitude
            # Mark start and end points as keypoints
            if end == data_length:
                end -= 1
            keypoints[start] = 1
            keypoints[end] = 1

        return data, keypoints

    @staticmethod
    def linear_line(data_length):
        slope = random.uniform(-1, 1)
        intercept = random.uniform(-1, 1)
        t = np.linspace(0, 1, data_length, endpoint=False)
        data = slope * t + intercept
        # Keypoints can be start and end points
        keypoints = np.zeros(data_length, dtype=int)
        keypoints[0] = 1
        keypoints[-1] = 1
        return data, keypoints.astype(int)

    @staticmethod
    def sawtooth_wave(data_length, frequency=None, amplitude=1):
        if frequency is None:
            frequency = random.uniform(0.1, 5)
        t = np.linspace(0, 1, data_length, endpoint=False)
        data = amplitude * (t * frequency - np.floor(t * frequency))
        # Keypoints at reset points
        keypoints = np.zeros(data_length, dtype=int)
        reset_points = (np.diff(np.mod(t * frequency, 1)) < 0).nonzero()[0]
        keypoints[reset_points] = 1
        return data, keypoints

    @staticmethod
    def radial_basis_function(data_length, max_num_gaussians=5):
        """
        Generates a mixture of Gaussian blobs (RBFs) along the time series.
        """
        t = np.linspace(0, 1, data_length)
        num_gaussians = random.randint(1, max_num_gaussians)
        sample = np.zeros(data_length)
        keypoints = np.zeros(data_length, dtype=int)
        for _ in range(num_gaussians):
            mean = random.uniform(0, 1)  # Center of the Gaussian
            std_dev = random.uniform(0.02, 0.2)
            amplitude = random.uniform(0.5, 2)
            gaussian = amplitude * np.exp(-((t - mean) ** 2) / (2 * std_dev**2))
            sample += gaussian
            # Keypoint at the peak
            peak_index = np.abs(t - mean).argmin()
            keypoints[peak_index] = 1
        if random.random() > 0.5:
            sample *= -1

        return sample, keypoints

    def simple_patterns(self, data_length, num_sections=5):
        """
        Generates time series data by partitioning the input into K sections and applying a pattern to each.
        """
        if num_sections <= 0:
            raise ValueError("num_sections must be greater than 0")

        section_length = data_length // num_sections
        remaining = data_length % num_sections
        patterns = [
            self.triangle_wave,
            self.gaussian_blob,
            self.block_wave,
            self.sine_wave_composition,
        ]

        sample = np.zeros(data_length)
        keypoints = np.zeros(data_length, dtype=int)

        start_idx = 0
        for section in range(num_sections):
            # Choose a random pattern for this section
            pattern_func = np.random.choice(patterns)
            # Adjust section length for the last section
            current_section_length = section_length + (
                remaining if section == num_sections - 1 else 0
            )
            # Generate the pattern for this section
            pattern, kp = pattern_func(current_section_length)
            # Place the pattern in the corresponding section of the sample
            sample[start_idx : start_idx + current_section_length] = pattern
            keypoints[start_idx : start_idx + current_section_length] = kp
            start_idx += current_section_length

        return sample, keypoints

    @staticmethod
    def triangle_wave(section_length, amplitude=1):
        """
        Generates a single triangle wave for a section.
        """
        amplitude *= 2 * np.random.rand() * amplitude
        x = np.linspace(0, 1, section_length)
        data = 2 * amplitude * np.abs(x - np.floor(x + 0.5))
        # Keypoints at peaks and troughs
        keypoints = np.zeros(section_length, dtype=int)
        # Peaks at x = 0, 0.5, 1
        peaks = [0, section_length // 2, section_length - 1]
        for idx in peaks:
            if 0 <= idx < section_length + 1:
                keypoints[idx] = 1
        if np.random.rand() > 0.5:
            data *= -1
        return data, keypoints

    @staticmethod
    def gaussian_blob(section_length, mean=None, std_dev=0.1, amplitude=1):
        """
        Generates a Gaussian blob for a section.
        """
        amplitude *= 2 * np.random.rand() * amplitude

        if mean is None:
            mean = section_length / 2
        x = np.arange(section_length)
        data = amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))
        # Keypoint at the peak
        keypoints = np.zeros(section_length, dtype=int)
        peak_index = int(mean)
        if 0 <= peak_index < section_length + 1:
            keypoints[peak_index] = 1
        if np.random.rand() > 0.5:
            data *= -1

        return data, keypoints

    def cache_to_numpy(self):
        N = len(self.cache)
        X = torch.stack([item[0] for item in self.cache]).numpy()
        X = X.transpose(0, 2, 1)
        keypoints = torch.stack([item[1] for item in self.cache]).int().numpy()

        return X, keypoints

    def set_data_length(self, new_length):
        self.data_length = new_length
