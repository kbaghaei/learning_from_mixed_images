import torch
import tqdm
import numpy as np


@torch.no_grad()
def naive_baseline_all_zeros(test_dataloader, mix_size, num_classes):
    """
    Naive baseline: Predict no labels (all zeros).
    
    Args:
        test_dataloader: DataLoader for test dataset
        mix_size: int, number of classes mixed in each sample
        num_classes: int, total number of classes
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    all_targets = []
    all_preds = []
    
    for images, targets in tqdm.tqdm(test_dataloader, desc="Baseline: All Zeros"):
        targets = targets.float()
        # Predict all zeros
        preds = torch.zeros_like(targets).int()
        
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())
    
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    from eval import compute_metrics_from_predictions
    return compute_metrics_from_predictions(all_targets, all_preds, mix_size)


@torch.no_grad()
def naive_baseline_all_ones(test_dataloader, mix_size, num_classes):
    """
    Naive baseline: Predict all labels (all ones).
    
    Args:
        test_dataloader: DataLoader for test dataset
        mix_size: int, number of classes mixed in each sample
        num_classes: int, total number of classes
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    all_targets = []
    all_preds = []
    
    for images, targets in tqdm.tqdm(test_dataloader, desc="Baseline: All Ones"):
        targets = targets.float()
        # Predict all ones
        preds = torch.ones_like(targets).int()
        
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())
    
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    from eval import compute_metrics_from_predictions
    return compute_metrics_from_predictions(all_targets, all_preds, mix_size)


@torch.no_grad()
def naive_baseline_random(test_dataloader, mix_size, num_classes, label_prob=0.5, random_seed=42):
    """
    Naive baseline: Randomly predict labels with given probability.
    
    Args:
        test_dataloader: DataLoader for test dataset
        mix_size: int, number of classes mixed in each sample
        num_classes: int, total number of classes
        label_prob: float, probability of predicting each label (default: 0.5)
        random_seed: int, random seed for reproducibility
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    all_targets = []
    all_preds = []
    
    for images, targets in tqdm.tqdm(test_dataloader, desc=f"Baseline: Random (p={label_prob})"):
        targets = targets.float()
        # Random predictions with given probability
        preds = (torch.rand_like(targets) < label_prob).int()
        
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())
    
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    from eval import compute_metrics_from_predictions
    return compute_metrics_from_predictions(all_targets, all_preds, mix_size)


@torch.no_grad()
def naive_baseline_label_frequency(test_dataloader, train_dataloader, mix_size, num_classes, random_seed=42):
    """
    Naive baseline: Predict labels based on their frequency in training data.
    Each label is predicted with probability equal to its frequency in training set.
    
    Args:
        test_dataloader: DataLoader for test dataset
        train_dataloader: DataLoader for training dataset (to compute label frequencies)
        mix_size: int, number of classes mixed in each sample
        num_classes: int, total number of classes
        random_seed: int, random seed for reproducibility
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Compute label frequencies from training data
    print("Computing label frequencies from training data...")
    train_targets = []
    for images, targets in tqdm.tqdm(train_dataloader, desc="Computing frequencies"):
        train_targets.append(targets.float())
    
    train_targets = torch.cat(train_targets).numpy()
    label_frequencies = train_targets.mean(axis=0)  # (num_classes,)
    
    print(f"Label frequencies: {label_frequencies}")
    
    all_targets = []
    all_preds = []
    
    for images, targets in tqdm.tqdm(test_dataloader, desc="Baseline: Label Frequency"):
        targets = targets.float()
        batch_size = targets.shape[0]
        
        # Predict each label with probability equal to its frequency
        # Create random values for each sample and label
        random_vals = torch.rand(batch_size, num_classes)
        # Compare with label frequencies (broadcasted)
        label_probs = torch.tensor(label_frequencies).unsqueeze(0).expand(batch_size, -1)
        preds = (random_vals < label_probs).int()
        
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())
    
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    from eval import compute_metrics_from_predictions
    return compute_metrics_from_predictions(all_targets, all_preds, mix_size)


@torch.no_grad()
def naive_baseline_fixed_mix_size(test_dataloader, mix_size, num_classes, random_seed=42):
    """
    Naive baseline: Randomly predict exactly 'mix_size' labels per sample.
    
    Args:
        test_dataloader: DataLoader for test dataset
        mix_size: int, number of classes mixed in each sample (also used for predictions)
        num_classes: int, total number of classes
        random_seed: int, random seed for reproducibility
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    all_targets = []
    all_preds = []
    
    for images, targets in tqdm.tqdm(test_dataloader, desc=f"Baseline: Fixed Mix Size ({mix_size})"):
        targets = targets.float()
        batch_size = targets.shape[0]
        
        # For each sample, randomly select exactly mix_size labels
        preds = torch.zeros(batch_size, num_classes, dtype=torch.int)
        for i in range(batch_size):
            # Randomly select mix_size indices
            indices = torch.randperm(num_classes)[:mix_size]
            preds[i, indices] = 1
        
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())
    
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    from eval import compute_metrics_from_predictions
    return compute_metrics_from_predictions(all_targets, all_preds, mix_size)


def run_naive_baselines(
    train_dataloader,
    test_dataloader,
    mix_size,
    num_classes,
    random_seed=42
):
    """
    Run all naive baselines on the test dataset and return results.
    
    Args:
        train_dataloader: DataLoader for training dataset (needed for label frequency baseline)
        test_dataloader: DataLoader for test dataset
        mix_size: int, number of classes mixed in each sample
        num_classes: int, total number of classes
        random_seed: int, random seed for reproducibility
    
    Returns:
        dict: Dictionary containing results for each baseline
    """
    results = {}
    
    print("\n" + "="*60)
    print("Running Naive Baselines for Multilabel Classification")
    print("="*60)
    
    # All zeros baseline
    print("\n1. All Zeros Baseline")
    results['all_zeros'] = naive_baseline_all_zeros(test_dataloader, mix_size, num_classes)
    print(f"   Micro F1: {results['all_zeros']['micro_f1']:.4f}")
    print(f"   Macro F1: {results['all_zeros']['macro_f1']:.4f}")
    print(f"   Accuracy: {results['all_zeros']['accuracy']:.4f}")
    print(f"   Hamming: {results['all_zeros']['hamming']:.4f}")
    print(f"   Jaccard: {results['all_zeros']['jaccard']:.4f}")
    
    # All ones baseline
    print("\n2. All Ones Baseline")
    results['all_ones'] = naive_baseline_all_ones(test_dataloader, mix_size, num_classes)
    print(f"   Micro F1: {results['all_ones']['micro_f1']:.4f}")
    print(f"   Macro F1: {results['all_ones']['macro_f1']:.4f}")
    print(f"   Accuracy: {results['all_ones']['accuracy']:.4f}")
    print(f"   Hamming: {results['all_ones']['hamming']:.4f}")
    print(f"   Jaccard: {results['all_ones']['jaccard']:.4f}")
    
    # Random baseline (p=0.5)
    print("\n3. Random Baseline (p=0.5)")
    results['random_0.5'] = naive_baseline_random(test_dataloader, mix_size, num_classes, label_prob=0.5, random_seed=random_seed)
    print(f"   Micro F1: {results['random_0.5']['micro_f1']:.4f}")
    print(f"   Macro F1: {results['random_0.5']['macro_f1']:.4f}")
    print(f"   Accuracy: {results['random_0.5']['accuracy']:.4f}")
    print(f"   Hamming: {results['random_0.5']['hamming']:.4f}")
    print(f"   Jaccard: {results['random_0.5']['jaccard']:.4f}")
    
    # Random baseline (p=mix_size/num_classes)
    prob = mix_size / num_classes
    print(f"\n4. Random Baseline (p={prob:.3f}, mix_size/num_classes)")
    results[f'random_{prob:.3f}'] = naive_baseline_random(test_dataloader, mix_size, num_classes, label_prob=prob, random_seed=random_seed)
    print(f"   Micro F1: {results[f'random_{prob:.3f}']['micro_f1']:.4f}")
    print(f"   Macro F1: {results[f'random_{prob:.3f}']['macro_f1']:.4f}")
    print(f"   Accuracy: {results[f'random_{prob:.3f}']['accuracy']:.4f}")
    print(f"   Hamming: {results[f'random_{prob:.3f}']['hamming']:.4f}")
    print(f"   Jaccard: {results[f'random_{prob:.3f}']['jaccard']:.4f}")
    
    # Label frequency baseline
    print("\n5. Label Frequency Baseline")
    results['label_frequency'] = naive_baseline_label_frequency(
        test_dataloader, train_dataloader, mix_size, num_classes, random_seed=random_seed
    )
    print(f"   Micro F1: {results['label_frequency']['micro_f1']:.4f}")
    print(f"   Macro F1: {results['label_frequency']['macro_f1']:.4f}")
    print(f"   Accuracy: {results['label_frequency']['accuracy']:.4f}")
    print(f"   Hamming: {results['label_frequency']['hamming']:.4f}")
    print(f"   Jaccard: {results['label_frequency']['jaccard']:.4f}")
    
    # Fixed mix size baseline
    print(f"\n6. Fixed Mix Size Baseline (exactly {mix_size} labels)")
    results['fixed_mix_size'] = naive_baseline_fixed_mix_size(test_dataloader, mix_size, num_classes, random_seed=random_seed)
    print(f"   Micro F1: {results['fixed_mix_size']['micro_f1']:.4f}")
    print(f"   Macro F1: {results['fixed_mix_size']['macro_f1']:.4f}")
    print(f"   Accuracy: {results['fixed_mix_size']['accuracy']:.4f}")
    print(f"   Hamming: {results['fixed_mix_size']['hamming']:.4f}")
    print(f"   Jaccard: {results['fixed_mix_size']['jaccard']:.4f}")
    
    print("\n" + "="*60)
    print("Baseline Evaluation Complete")
    print("="*60 + "\n")
    
    return results