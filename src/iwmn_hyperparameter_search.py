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

from models import IWMNMNISTClassifier
from utils import get_normalization_stats, create_transforms

def train_batch(model, data, target, optimizer, criterion, device):
    """Train a single batch with IWMN."""
    data, target = data.to(device), target.to(device)
    
    # Convert target to one-hot encoding for IWMN
    target_onehot = torch.zeros(target.size(0), 10, device=device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass - initial prediction without target
    initial_outputs, _, _ = model(data, target_onehot, return_modulations=True)
    
    # Calculate loss on initial outputs
    initial_loss = criterion(initial_outputs, target)
    
    # Backward and optimize
    initial_loss.backward()
    optimizer.step()
    
    # Track metrics
    _, predicted = initial_outputs.max(1)
    correct = predicted.eq(target).sum().item()
    
    return initial_loss.item(), correct

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Train batch
        loss, batch_correct = train_batch(
            model, data, target, optimizer, criterion, device
        )
        
        # Update metrics
        total_loss += loss
        total += target.size(0)
        correct += batch_correct
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Convert target to one-hot for IWMN
            target_onehot = torch.zeros(target.size(0), 10, device=device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            
            # Forward pass
            outputs, _, _ = model(data, target_onehot)
            
            # Calculate loss
            loss = criterion(outputs, target)
            
            # Accumulate statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def measure_inference_time(model, test_loader, device, num_samples=1000):
    """Measure inference time for the model."""
    model.eval()
    total_time = 0
    count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if count >= num_samples:
                break
                
            batch_size = data.size(0)
            data = data.to(device)
            
            # Convert target to one-hot for IWMN (needed for interface, but not used in inference)
            target_onehot = torch.zeros(batch_size, 10, device=device)
            
            # Measure inference time
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            
            total_time += end_time - start_time
            count += batch_size
    
    # Calculate average time per sample
    avg_time = (total_time / count) * 1000  # Convert to milliseconds
    return avg_time

def train_with_hyperparams(hyperparams: Dict[str, Any], train_loader, val_loader, test_loader, device, epochs=5) -> Dict[str, Any]:
    """
    Train a model with the given hyperparameters and return metrics.
    
    Args:
        hyperparams: Dictionary containing hyperparameters 
                    (num_iterations, modulation_strength, hidden_size, dropout_rate, learning_rate)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to train on
        epochs: Number of epochs to train for
        
    Returns:
        Dictionary with metrics (val_accuracy, train_time, etc.)
    """
    # Extract hyperparameters
    num_iterations = hyperparams['num_iterations']
    modulation_strength = hyperparams['modulation_strength']
    hidden_size = hyperparams['hidden_size']
    dropout_rate = hyperparams['dropout_rate']
    learning_rate = hyperparams['learning_rate']
    
    print(f"\n{'='*80}")
    print(f"Training IWMN with hyperparameters:")
    print(f"  num_iterations={num_iterations}, modulation_strength={modulation_strength}")
    print(f"  hidden_size={hidden_size}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
    print(f"{'='*80}\n")
    
    # Create model with these hyperparameters
    model = IWMNMNISTClassifier(
        num_iterations=num_iterations,
        modulation_strength=modulation_strength,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    ).to(device)
    
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
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
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
            model_filename = f"model_checkpoints/iwmn_iter_{num_iterations}_ms_{modulation_strength}_hs_{hidden_size}_dr_{dropout_rate}_lr_{learning_rate}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved at {model_filename} with accuracy: {best_acc:.2f}%")
    
    # Final evaluation on test set
    final_test_loss, final_test_acc = validate(model, test_loader, criterion, device)
    training_time = time.time() - start_time
    
    # Measure inference time
    inference_time = measure_inference_time(model, test_loader, device)
    
    # Return metrics
    return {
        'num_iterations': num_iterations,
        'modulation_strength': modulation_strength,
        'hidden_size': hidden_size,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'num_params': num_params,
        'best_val_accuracy': best_acc,
        'final_test_accuracy': final_test_acc,
        'training_time': training_time,
        'inference_time_ms': inference_time,
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
    os.makedirs('iwmn_hyperparameter_results', exist_ok=True)
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Run training for each combination
    results = []
    for i, combination in enumerate(all_combinations):
        hyperparams = dict(zip(keys, combination))
        print(f"\nCombination {i+1}/{len(all_combinations)}: {hyperparams}")
        
        result = train_with_hyperparams(hyperparams, train_loader, val_loader, test_loader, device, epochs)
        results.append(result)
        
        # Save interim results to avoid losing progress
        save_results(results, 'iwmn_hyperparameter_results/interim_results.json')
    
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
    os.makedirs('iwmn_hyperparameter_results', exist_ok=True)
    
    # Plot hyperparameter vs. accuracy
    plt.figure(figsize=(20, 15))
    
    # Iterations vs. accuracy subplot
    plt.subplot(3, 2, 1)
    for hidden_size in df['hidden_size'].unique():
        subset = df[df['hidden_size'] == hidden_size]
        plt.scatter(subset['num_iterations'], subset['best_val_accuracy'], 
                   label=f'Hidden Size={hidden_size}')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Number of Iterations vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Modulation strength vs. accuracy subplot
    plt.subplot(3, 2, 2)
    for hidden_size in df['hidden_size'].unique():
        subset = df[df['hidden_size'] == hidden_size]
        plt.scatter(subset['modulation_strength'], subset['best_val_accuracy'],
                   label=f'Hidden Size={hidden_size}')
    plt.xlabel('Modulation Strength')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Modulation Strength vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Hidden size vs. accuracy subplot
    plt.subplot(3, 2, 3)
    for iter_count in sorted(df['num_iterations'].unique()):
        subset = df[df['num_iterations'] == iter_count]
        plt.scatter(subset['hidden_size'], subset['best_val_accuracy'],
                   label=f'Iterations={iter_count}')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Hidden Size vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dropout rate vs. accuracy subplot
    plt.subplot(3, 2, 4)
    for hidden_size in df['hidden_size'].unique():
        subset = df[df['hidden_size'] == hidden_size]
        plt.scatter(subset['dropout_rate'], subset['best_val_accuracy'],
                   label=f'Hidden Size={hidden_size}')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Dropout Rate vs. Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parameters vs. accuracy subplot
    plt.subplot(3, 2, 5)
    scatter = plt.scatter(df['num_params'], df['best_val_accuracy'], 
                         c=df['num_iterations'], cmap='viridis')
    plt.colorbar(scatter, label='Number of Iterations')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Size vs. Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Inference time vs. accuracy subplot
    plt.subplot(3, 2, 6)
    scatter = plt.scatter(df['inference_time_ms'], df['best_val_accuracy'], 
                         c=df['num_iterations'], cmap='viridis')
    plt.colorbar(scatter, label='Number of Iterations')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Inference Time vs. Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iwmn_hyperparameter_results/hyperparameter_comparison.png')
    
    # Save detailed results table
    html_table = df[['num_iterations', 'modulation_strength', 'hidden_size', 'dropout_rate', 'learning_rate',
                     'num_params', 'best_val_accuracy', 'final_test_accuracy',
                     'training_time', 'inference_time_ms']].to_html()
    with open('iwmn_hyperparameter_results/results_table.html', 'w') as f:
        f.write(html_table)
    
    # Plot top models performance (iterations vs. accuracy)
    plt.figure(figsize=(10, 6))
    
    # Group by num_iterations and get mean accuracy
    iteration_perf = df.groupby('num_iterations')['best_val_accuracy'].mean().reset_index()
    plt.plot(iteration_perf['num_iterations'], iteration_perf['best_val_accuracy'], 'o-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Validation Accuracy (%)')
    plt.title('Number of Iterations vs. Mean Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('iwmn_hyperparameter_results/iterations_vs_accuracy.png')
    
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
        plt.title(f"Top {i+1} Model - Iterations={result['num_iterations']}, "
                 f"ModStr={result['modulation_strength']}, "
                 f"HS={result['hidden_size']}")
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
    plt.savefig('iwmn_hyperparameter_results/top_models_learning_curves.png')
    
    # Generate summary text
    with open('iwmn_hyperparameter_results/summary.txt', 'w') as f:
        f.write("IWMN Hyperparameter Search Results Summary\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Number of models trained: {len(df)}\n\n")
        
        f.write("Top 5 Models:\n")
        for i in range(min(5, len(df))):
            result = df.iloc[i]
            f.write(f"{i+1}. Iterations={result['num_iterations']}, "
                   f"ModStr={result['modulation_strength']}, "
                   f"HS={result['hidden_size']}, "
                   f"Dropout={result['dropout_rate']}\n")
            f.write(f"   Val Acc: {result['best_val_accuracy']:.2f}%, "
                   f"Test Acc: {result['final_test_accuracy']:.2f}%, "
                   f"Params: {result['num_params']}, "
                   f"Inference: {result['inference_time_ms']:.2f}ms\n\n")
        
        f.write("\nBest Parameters:\n")
        best = df.iloc[0]
        f.write(f"Number of Iterations: {best['num_iterations']}\n")
        f.write(f"Modulation Strength: {best['modulation_strength']}\n")
        f.write(f"Hidden Size: {best['hidden_size']}\n")
        f.write(f"Dropout Rate: {best['dropout_rate']}\n")
        f.write(f"Learning Rate: {best['learning_rate']}\n")
        f.write(f"Parameters: {best['num_params']}\n")
        f.write(f"Validation Accuracy: {best['best_val_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {best['final_test_accuracy']:.2f}%\n")
        f.write(f"Training Time: {best['training_time']:.2f} seconds\n")
        f.write(f"Inference Time: {best['inference_time_ms']:.2f} milliseconds\n")
    
    print(f"Results visualizations saved to iwmn_hyperparameter_results/")

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
        'num_iterations': [1, 2, 3, 4, 5],  # Number of iterative refinement steps
        'modulation_strength': [0.05, 0.1, 0.2],  # Strength of modulation signal
        'hidden_size': [64, 128, 256],  # Size of hidden layer
        'dropout_rate': [0.0, 0.2, 0.4],  # Dropout rate for regularization
        'learning_rate': [0.0005, 0.001, 0.002]  # Learning rate for optimizer
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
    save_results(results, 'iwmn_hyperparameter_results/final_results.json')
    
    # Visualize results
    visualize_results(results)
    
    # Print best hyperparameters
    df = pd.DataFrame(results)
    best_idx = df['best_val_accuracy'].idxmax()
    best_params = df.loc[best_idx]
    
    print("\n" + "="*80)
    print("Best Hyperparameters:")
    print(f"Number of Iterations: {best_params['num_iterations']}")
    print(f"Modulation Strength: {best_params['modulation_strength']}")
    print(f"Hidden Size: {best_params['hidden_size']}")
    print(f"Dropout Rate: {best_params['dropout_rate']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Validation Accuracy: {best_params['best_val_accuracy']:.2f}%")
    print(f"Test Accuracy: {best_params['final_test_accuracy']:.2f}%")
    print(f"Inference Time: {best_params['inference_time_ms']:.2f} milliseconds")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
