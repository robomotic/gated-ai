import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import json
import itertools
import pandas as pd
from typing import Dict, List, Tuple, Any

from models import MoEMNISTClassifier
from utils import get_normalization_stats, create_transforms
from train import train, validate

def train_with_hyperparams(hyperparams: Dict[str, Any], train_loader, val_loader, test_loader, device, epochs=5) -> Dict[str, Any]:
    """
    Train a model with the given hyperparameters and return metrics.
    
    Args:
        hyperparams: Dictionary containing hyperparameters 
                    (learning_rate, hidden_size, num_experts, k)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to train on
        epochs: Number of epochs to train for
        
    Returns:
        Dictionary with metrics (val_accuracy, train_time, etc.)
    """
    # Extract hyperparameters
    learning_rate = hyperparams['learning_rate']
    hidden_size = hyperparams['hidden_size']
    num_experts = hyperparams['num_experts']
    k = hyperparams['k']
    
    print(f"\n{'='*80}")
    print(f"Training with hyperparameters: learning_rate={learning_rate}, hidden_size={hidden_size}, num_experts={num_experts}, k={k}")
    print(f"{'='*80}\n")
    
    # Create model with these hyperparameters
    model = MoEMNISTClassifier(num_experts=num_experts, k=k, hidden_size=hidden_size).to(device)
    
    # Count parameters
    num_params = model.count_parameters()
    print(f"Model has {num_params} trainable parameters")
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_filename = f"model_checkpoints/moe_lr_{learning_rate}_hs_{hidden_size}_ne_{num_experts}_k_{k}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved at {model_filename} with accuracy: {best_acc:.2f}%")
    
    # Final evaluation on test set
    final_test_loss, final_test_acc = validate(model, test_loader, criterion, device)
    training_time = time.time() - start_time
    
    # Return metrics
    return {
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'num_experts': num_experts,
        'k': k,
        'num_params': num_params,
        'best_val_accuracy': best_acc,
        'final_test_accuracy': final_test_acc,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def run_grid_search(param_grid: Dict[str, List], train_loader, val_loader, test_loader, device, epochs=5) -> List[Dict[str, Any]]:
    """
    Run a grid search over hyperparameters.
    
    Args:
        param_grid: Dictionary mapping hyperparameter names to lists of values to try
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to train on
        epochs: Number of epochs to train for each hyperparameter combination
    
    Returns:
        List of dictionaries with results for each hyperparameter combination
    """
    # Generate all combinations of hyperparameters
    keys = param_grid.keys()
    values = param_grid.values()
    all_combinations = list(itertools.product(*values))
    
    # Create directory for results if it doesn't exist
    os.makedirs('hyperparameter_search_results', exist_ok=True)
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Run training for each combination
    results = []
    for i, combination in enumerate(all_combinations):
        hyperparams = dict(zip(keys, combination))
        print(f"\nCombination {i+1}/{len(all_combinations)}: {hyperparams}")
        
        result = train_with_hyperparams(hyperparams, train_loader, val_loader, test_loader, device, epochs)
        results.append(result)
        
        # Save interim results to avoid losing progress
        save_results(results, 'hyperparameter_search_results/interim_results.json')
    
    return results

def save_results(results: List[Dict[str, Any]], filename: str):
    """Save results to a file."""
    # Convert numpy arrays and tensors to lists
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                serializable_result[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.ndarray, torch.Tensor)):
                serializable_result[key] = [v.tolist() for v in value]
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def visualize_results(results: List[Dict[str, Any]]):
    """Create visualizations of hyperparameter search results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by validation accuracy
    df = df.sort_values('best_val_accuracy', ascending=False)
    
    # Create directory for visualizations
    os.makedirs('hyperparameter_search_results', exist_ok=True)
    
    # Plot hyperparameter vs. accuracy
    plt.figure(figsize=(15, 10))
    
    # Learning rate vs. accuracy subplot
    plt.subplot(2, 2, 1)
    for hidden_size in df['hidden_size'].unique():
        subset = df[df['hidden_size'] == hidden_size]
        plt.scatter(subset['learning_rate'], subset['best_val_accuracy'], 
                   label=f'Hidden Size={hidden_size}')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Learning Rate vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Hidden size vs. accuracy subplot
    plt.subplot(2, 2, 2)
    for lr in df['learning_rate'].unique():
        subset = df[df['learning_rate'] == lr]
        plt.scatter(subset['hidden_size'], subset['best_val_accuracy'],
                   label=f'Learning Rate={lr}')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Hidden Size vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Number of experts vs. accuracy subplot
    plt.subplot(2, 2, 3)
    for k in df['k'].unique():
        subset = df[df['k'] == k]
        plt.scatter(subset['num_experts'], subset['best_val_accuracy'],
                   label=f'k={k}')
    plt.xlabel('Number of Experts')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Number of Experts vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parameters vs. accuracy subplot
    plt.subplot(2, 2, 4)
    plt.scatter(df['num_params'], df['best_val_accuracy'])
    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Size vs. Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_results/hyperparameter_comparison.png')
    
    # Save detailed results table
    html_table = df[['learning_rate', 'hidden_size', 'num_experts', 'k', 
                     'num_params', 'best_val_accuracy', 'final_test_accuracy',
                     'training_time']].to_html()
    with open('hyperparameter_search_results/results_table.html', 'w') as f:
        f.write(html_table)
    
    # Plot learning curves for top models
    top_n = min(3, len(df))
    plt.figure(figsize=(15, 5 * top_n))
    for i in range(top_n):
        result = df.iloc[i]
        plt.subplot(top_n, 2, 2*i+1)
        plt.plot(result['train_losses'], label='Train Loss')
        plt.plot(result['val_losses'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Top {i+1} Model - LR={result['learning_rate']}, "
                 f"HS={result['hidden_size']}, "
                 f"NE={result['num_experts']}, k={result['k']}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(top_n, 2, 2*i+2)
        plt.plot(result['train_accs'], label='Train Acc')
        plt.plot(result['val_accs'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f"Top {i+1} Model - Acc={result['best_val_accuracy']:.2f}%")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_results/top_models_learning_curves.png')
    
    # Generate summary text
    with open('hyperparameter_search_results/summary.txt', 'w') as f:
        f.write("Hyperparameter Search Results Summary\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"Number of models trained: {len(df)}\n\n")
        
        f.write("Top 5 Models:\n")
        for i in range(min(5, len(df))):
            result = df.iloc[i]
            f.write(f"{i+1}. LR={result['learning_rate']}, HS={result['hidden_size']}, "
                   f"NE={result['num_experts']}, k={result['k']}\n")
            f.write(f"   Val Acc: {result['best_val_accuracy']:.2f}%, "
                   f"Test Acc: {result['final_test_accuracy']:.2f}%, "
                   f"Params: {result['num_params']}, "
                   f"Time: {result['training_time']:.2f}s\n\n")
        
        f.write("\nBest Parameters:\n")
        best = df.iloc[0]
        f.write(f"Learning Rate: {best['learning_rate']}\n")
        f.write(f"Hidden Size: {best['hidden_size']}\n")
        f.write(f"Number of Experts: {best['num_experts']}\n")
        f.write(f"k: {best['k']}\n")
        f.write(f"Parameters: {best['num_params']}\n")
        f.write(f"Validation Accuracy: {best['best_val_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {best['final_test_accuracy']:.2f}%\n")
        f.write(f"Training Time: {best['training_time']:.2f} seconds\n")
    
    print(f"Results visualizations saved to hyperparameter_search_results/")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('model_checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Define batch size
    batch_size = 128
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_size': [128, 256, 512],
        'num_experts': [4],  # Fixed for this search
        'k': [2]  # Fixed for this search
    }
    
    # Load raw dataset for computing statistics
    raw_train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Calculate normalization statistics dynamically
    mean, std = get_normalization_stats('mnist', raw_train_dataset)
    
    # Create transforms with calculated statistics
    train_transform = create_transforms(mean, std)
    test_transform = create_transforms(mean, std)
    
    # Load MNIST dataset with proper transforms
    train_dataset = datasets.MNIST(
        root='data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Split training data into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Run hyperparameter search with 5 epochs per model
    results = run_grid_search(param_grid, train_loader, val_loader, test_loader, device, epochs=5)
    
    # Save final results
    save_results(results, 'hyperparameter_search_results/final_results.json')
    
    # Visualize results
    visualize_results(results)
    
    # Print best hyperparameters
    df = pd.DataFrame(results)
    best_idx = df['best_val_accuracy'].idxmax()
    best_params = df.loc[best_idx]
    
    print("\n" + "="*80)
    print("Best Hyperparameters:")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Hidden Size: {best_params['hidden_size']}")
    print(f"Number of Experts: {best_params['num_experts']}")
    print(f"k: {best_params['k']}")
    print(f"Validation Accuracy: {best_params['best_val_accuracy']:.2f}%")
    print(f"Test Accuracy: {best_params['final_test_accuracy']:.2f}%")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
