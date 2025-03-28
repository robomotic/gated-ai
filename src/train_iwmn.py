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

from models import IWMNMNISTClassifier
from utils import get_normalization_stats, create_transforms


def train_batch(model, data, target, optimizer, criterion, device, num_iterations=3):
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


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_iterations=3):
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
            
            # Forward pass - just get the initial outputs
            outputs, _, _ = model(data, target_onehot)
            
            # Calculate loss
            loss = criterion(outputs, target)
            
            # Accumulate statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total


def visualize_weight_modulations(model, test_loader, device, num_samples=5):
    """Visualize how activations are modulated over iterations."""
    model.eval()
    
    # Get a batch of data
    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data[:num_samples].to(device), target[:num_samples].to(device)
    
    # Convert target to one-hot encoding
    target_onehot = torch.zeros(target.size(0), 10, device=device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    
    # Run forward pass with modulation history
    _, _, modulation_history = model(data, target_onehot, return_modulations=True)
    
    # Plot the first sample's image
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.imshow(data[0].cpu().squeeze(), cmap='gray')
    plt.title(f'Input Image - Target: {target[0].item()}')
    
    # Plot output probability changes over iterations
    plt.subplot(3, 1, 2)
    outputs_over_time = [torch.softmax(info['output'][0], dim=0).numpy() for info in modulation_history]
    outputs_over_time = np.vstack(outputs_over_time)
    
    plt.imshow(outputs_over_time, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('Class')
    plt.ylabel('Iteration')
    plt.title('Output Probabilities Over Iterations')
    
    # Plot the modulation magnitudes
    plt.subplot(3, 1, 3)
    hidden_mod_magnitude = [torch.norm(info['hidden_mod']).item() for info in modulation_history]
    output_mod_magnitude = [torch.norm(info['output_mod']).item() for info in modulation_history]
    
    plt.plot(hidden_mod_magnitude, label='Hidden Layer Modulation Magnitude')
    plt.plot(output_mod_magnitude, label='Output Layer Modulation Magnitude')
    plt.xlabel('Iteration')
    plt.ylabel('L2 Norm')
    plt.title('Activation Modulation Magnitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_modulation_visualization.png')
    plt.close()
    print("Activation modulation visualization saved to activation_modulation_visualization.png")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('model_checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Define hyperparameters
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001
    num_iterations = 3  # Number of weight modulation iterations
    modulation_strength = 0.1  # Strength of weight modulation
    
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
    
    # Create model
    model = IWMNMNISTClassifier(
        num_iterations=num_iterations,
        modulation_strength=modulation_strength
    ).to(device)
    
    print(f"Model created with {num_iterations} iterations and modulation strength {modulation_strength}")
    print(f"Total parameters: {model.count_parameters():,}")
    
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
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_iterations
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model_checkpoints/iwmn_best.pth')
            print(f"Model saved with accuracy: {best_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Visualize weight modulations
    visualize_weight_modulations(model, test_loader, device)
    
    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('iwmn_training_curves.png')
    print("Training curves saved to iwmn_training_curves.png")


if __name__ == '__main__':
    main()
