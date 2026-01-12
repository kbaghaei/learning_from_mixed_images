import torch
import numpy as np
from typing import List
import torchvision.transforms as transforms


# Define a transform to normalize the data for CIFAR-10
# CIFAR-10 mean and std deviation values
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(cifar10_mean, cifar10_std)])

def mult_hot_encode(integer_labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Converts a single integer label to a one-hot encoded PyTorch tensor.

    Args:
        integer_label (int): The integer label to convert (0-indexed).
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: A one-hot encoded tensor of shape (num_classes,).

    Raises:
        ValueError: If integer_label is out of bounds for num_classes.
    """

    # torch.eye(num_classes) creates an identity matrix
    # Indexing with integer_labels selects the corresponding row as the one-hot vector
    mult_hot_tensor = torch.eye(num_classes)[integer_labels].sum(dim=0)
    return mult_hot_tensor

from itertools import combinations

def get_combination_lists(mix_size : int, classes_list : List[int],
                          classes_for_test : List[int]) -> List[List[int]]:
    """
    Args:
    mix_size (int) : the number of classes to mix in each sample
    classes_list ( List[int] ): the list of all classes
    classes_for_test ( List[int] ) : the list of classes to use for testing


    Returns a pair of lists of combinations of classes for training and testing.
    The combinations used in test set will not show up in training set.

    Example:
    classes_list : [1,2,3,4]
    classes_for_test : [3,4]
    mix_size : 2

    Returns:
    combs_train : [[1,2], [1,3], [1,4], [2,3], [2,4]]
    combs_test: [[3,4]]
    """

    # All 3-combinations of 10 classes ( total : 120 )
    # All 2-combs ( total : 45)
    combs_all = list(combinations(classes_list, mix_size))
    combs_list = [tuple(comb) for comb in combs_all]

    # We set aside all combinations of last 5 classes for test ( total : 10 )
    # 2-combs of 5 classes: 10
    combs_test_classes = list(combinations(classes_for_test, mix_size))
    combs_test_classes = set([tuple(comb) for comb in combs_test_classes])

    combs_train = set(combs_list).difference(combs_test_classes)
    combs_test = combs_test_classes

    return list(combs_train), list(combs_test)


def load_mixed_datasets(train_dataset, test_dataset, classes_list, classes_for_test, mix_size, n_samples_per_mix, random_seed):
    """
    Load and return mixed datasets for training and testing.

    Args:
    train_dataset: training dataset of type torchvision.datasets.CIFAR10
    test_dataset: testing dataset of type torchvision.datasets.CIFAR10
    classes_list ( List[int] ): the list of all classes
    classes_for_test ( List[int] ) : the list of classes to use for testing
    mix_size (int) : the number of classes to mix in each sample
    n_samples_per_mix (int) : the number of mixed samples for each combination
    random_seed (int) : the seed for the random number generator
    """

    from mixer_dataset import MixerDataset

    combs_train, combs_test = get_combination_lists(mix_size, classes_list, classes_for_test)

    train_mx = MixerDataset(train_dataset.data, train_dataset.targets,
                num_classes=len(classes_list), mix_size=mix_size, mix_combs_list=combs_train,
                n_samples_per_mix=n_samples_per_mix, random_seed=random_seed, transform=train_transform)

    test_mx = MixerDataset(test_dataset.data, test_dataset.targets,
                num_classes=len(classes_list), mix_size=mix_size, mix_combs_list=combs_test,
                n_samples_per_mix=n_samples_per_mix, random_seed=random_seed, transform=test_transform)

    return train_mx, test_mx