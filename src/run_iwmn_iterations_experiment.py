import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import json

from models import IWMNMNISTClassifier


def train_batch(model, data, target, optimizer, criterion, device, num_iterations):
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


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_iterations):
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
            model, data, target, optimizer, criterion, 
            device, num_iterations
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


def validate(model, val_loader, criterion, device, num_iterations):
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
            
            # Forward pass - get outputs with specified iterations
            model.iwmn.num_iterations = num_iterations
            outputs, _, _ = model(data, target_onehot)
            
            # Calculate loss
            loss = criterion(outputs, target)
            
            # Accumulate statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total


def measure_inference_time(model, test_loader, device, num_iterations, num_samples=1000):
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
            
            # Set iteration count
            model.iwmn.num_iterations = num_iterations
            
            # Measure inference time
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            
            total_time += end_time - start_time
            count += batch_size
    
    # Calculate average time per sample
    avg_time = total_time / count
    return avg_time


def train_and_evaluate(num_iterations, device, train_loader, test_loader, num_epochs=5):
    """Train and evaluate an IWMN model with specified iteration count."""
    print(f"\n{'='*50}")
    print(f"Training IWMN with {num_iterations} iterations")
    print(f"{'='*50}\n")
    
    # Create model, optimizer, criterion
    model = IWMNMNISTClassifier(num_iterations=num_iterations, modulation_strength=0.1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, num_epochs + 1):
        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_iterations
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device, num_iterations)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Final evaluation
    model.iwmn.num_iterations = num_iterations
    final_loss, final_acc = validate(model, test_loader, criterion, device, num_iterations)
    
    # Measure inference time
    inference_time = measure_inference_time(model, test_loader, device, num_iterations)
    
    # Count parameters
    num_params = model.count_parameters()
    
    # Save model
    torch.save(model.state_dict(), f'model_checkpoints/iwmn_iterations_{num_iterations}.pth')
    
    # Create training curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'IWMN ({num_iterations} iterations) Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'IWMN ({num_iterations} iterations) Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'iwmn_{num_iterations}_iterations_curves.png')
    plt.close()
    
    return {
        'iterations': num_iterations,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'inference_time': inference_time,
        'params': num_params
    }


def plot_comparison_results(results):
    """Plot comparison of different iteration counts."""
    # Extract data for plotting
    iterations = [result['iterations'] for result in results]
    accuracies = [result['final_acc'] for result in results]
    inference_times = [result['inference_time'] * 1000 for result in results]  # Convert to ms
    params = [result['params'] for result in results]
    
    # Ensure comparison_results directory exists
    os.makedirs('comparison_results', exist_ok=True)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame({
        'Iterations': iterations,
        'Accuracy (%)': accuracies,
        'Inference Time (ms)': inference_times,
        'Parameters': params
    })
    df.to_csv('comparison_results/iwmn_iterations_results.csv', index=False)
    
    # Plot accuracy vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy (%)')
    plt.title('IWMN: Accuracy vs Number of Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('comparison_results/iwmn_accuracy_vs_iterations.png')
    plt.close()
    
    # Plot inference time vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, inference_times, 'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Inference Time (ms)')
    plt.title('IWMN: Inference Time vs Number of Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('comparison_results/iwmn_inference_time_vs_iterations.png')
    plt.close()
    
    # Plot accuracy vs inference time
    plt.figure(figsize=(10, 6))
    plt.scatter(inference_times, accuracies, s=100, c=iterations, cmap='viridis')
    plt.colorbar(label='Iterations')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Accuracy (%)')
    plt.title('IWMN: Accuracy vs Inference Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('comparison_results/iwmn_accuracy_vs_inference_time.png')
    plt.close()
    
    # Print summary
    print("\nResults Summary:")
    print(df.to_string(index=False))
    print("\nResults saved to comparison_results/iwmn_iterations_results.csv")
    print("Plots saved to comparison_results/ directory")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('model_checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('comparison_results', exist_ok=True)
    
    # Define hyperparameters
    batch_size = 128
    num_epochs = 5
    iteration_counts = [4, 5, 6]  # Run experiments with these iteration counts
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Run experiments for each iteration count
    results = []
    for iterations in iteration_counts:
        result = train_and_evaluate(
            iterations, device, train_loader, test_loader, num_epochs
        )
        results.append(result)
    
    # Plot and save comparison results
    plot_comparison_results(results)
    
    # Save results as JSON
    with open('comparison_results/iwmn_iterations_results.json', 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {k: float(v) if isinstance(v, torch.Tensor) else v 
                                  for k, v in result.items()}
            serializable_results.append(serializable_result)
        json.dump(serializable_results, f, indent=2)


if __name__ == '__main__':
    main()
