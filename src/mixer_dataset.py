import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import deque
from typing import List

class SampleIndexDealer:
    """
    A utility class that manages sampling of indices without replacement when the 
    sampled population is smaller than the original population. And, whenever the 
    sampled population is going to be larger than the original population, it ensures
    that replacement is minimized.
    
    This class maintains a shuffled queue of sample indices. When all indices
    have been sampled, it automatically reshuffles and starts over, ensuring
    all samples are used before any repetition occurs.
    
    Attributes:
        samples (list): List of sample indices from 0 to n_samples-1.
        _q (deque): Queue containing the shuffled indices ready to be sampled.
    
    Example:
        >>> dealer = SampleIndexDealer(n_samples=10)
        >>> idx = dealer.sample()  # Returns a random index 0-9
        >>> # After 10 calls, indices are reshuffled automatically
    """
    def __init__(self, samples : List[int]):
        self.samples = samples
        self._reset()

    def _reset(self):
        random.shuffle(self.samples)
        self._q = deque(self.samples)

    def sample(self):
        if not self._q:
            self._reset()

        item = self._q.popleft()
        return item

    def max_unique_samples(self):
        return len(self.samples)

class MixerDataset(Dataset):
    def __init__(self, data, targets, num_classes, mix_size, mix_combs_list, n_samples_per_mix, random_seed, transform=None):
        """
        Args:
            data (torch.Tensor): Tensor of shape (num_samples, num_channels, height, width).
            num_classes (int): Number of classes in the dataset.
            mix_size (int): Number of samples from different classes to mix.
            mix_combs_list (list): List of tuples of combinations of classes to mix.
            n_samples_per_mix (int): Number of samples for each mix.
            train and test should vary.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._raw_data = data
        self._num_classes = num_classes
        self.mix_size = mix_size
        self.mix_combs_list = mix_combs_list
        self.n_samples_per_mix = n_samples_per_mix

        self._raw_targets = targets
        self._random_seed = random_seed
        self.transform = transform

        self._target_to_index = {i : [] for i in range(num_classes)}
        for i, target in enumerate(targets):
            self._target_to_index[target].append(i)

        self._samples_log = {i : [] for i in range(num_classes)}

        self.data = None
        self.targets = None

        self._prep_mixes()

    def _prep_mixes(self):
        """
        Prepares the mixes for the dataset.
        """

        from helper_functions import mult_hot_encode

        random.seed(self._random_seed)

        self.data = []
        self.targets = []

        if self.mix_size == 1:
            max_samples = min(self.n_samples_per_mix * self._num_classes, len(self._raw_data))
            self.data = self._raw_data[:max_samples]
            # Apply mult_hot_encode to each target individually
            self.targets = [mult_hot_encode([target], self._num_classes) for target in self._raw_targets[:max_samples]]
            print('Mixed size is 1, so ignoring: [mix_size, classes_for_test].')
            return

        else:
            indexSamplers = []
            for c in range(self._num_classes):
                indexSamplers.append(SampleIndexDealer(self._target_to_index[c]))

            for i in range(self.n_samples_per_mix):
                for comb in self.mix_combs_list:
                    mixed_sample = np.zeros_like(self._raw_data[0])
                    mixed_target = mult_hot_encode(list(comb), self._num_classes)
                    
                    idx_log = []
                    for cls in comb:
                        #idx = random.choice(self._target_to_index[cls])
                        idx = indexSamplers[cls].sample()
                        mixed_sample += self._raw_data[idx]
                        idx_log.append((idx, self._raw_targets[idx]))

                    for cls in comb:
                        self._samples_log[cls].append((comb, idx_log))
                    self.data.append(mixed_sample)
                    self.targets.append(mixed_target)

        print('Prepared mixes.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        label = self.targets[idx]

        # Apply transformations if any
        if self.transform:
            # Convert numpy array (H, W, C) uint8 to PIL Image
            if isinstance(sample, np.ndarray):
                # Ensure values are in valid range [0, 255] and convert to uint8
                sample = np.clip(sample, 0, 255).astype(np.uint8)
                sample = Image.fromarray(sample)
            sample = self.transform(sample)

        return sample, label

# Example Usage (assuming you have some dummy data and labels):
# class_labels = [0,1,2,3,4,5,6,7,8,9]
# dummy_data = torch.randn(100, 3, 32, 32) # 100 samples of 3x32x32 images
# dummy_labels = torch.randint(0, 10, (100,)) # 100 random labels between 0 and 9

# custom_dataset = CustomDataset(dummy_data, dummy_labels)
# custom_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

# for batch_idx, (data, labels) in enumerate(custom_loader):
#     print(f"Batch {batch_idx}: Data shape {data.shape}, Labels shape {labels.shape}")
#     if batch_idx == 0:
#         break
