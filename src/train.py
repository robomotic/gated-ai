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

from models import MoEMNISTClassifier
from utils import get_normalization_stats, create_transforms


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, gates = model(data)
        
        # Calculate loss
        loss = criterion(outputs, target)
        
        # Add load balancing loss (optional)
        # This encourages equal use of experts
        expert_usage = gates.mean(0)
        load_balancing_loss = torch.sum(expert_usage * torch.log(expert_usage + 1e-10)) * 0.1
        loss += load_balancing_loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar description
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs, _ = model(data)
            
            # Calculate loss
            loss = criterion(outputs, target)
            
            # Accumulate statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total


def visualize_expert_usage(model, test_loader, device, num_samples=100):
    model.eval()
    expert_usages = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            _, gates = model(data)
            
            # Store gates and labels
            expert_usages.append(gates.cpu().numpy())
            labels.append(target.cpu().numpy())
            
            if len(expert_usages) * data.size(0) >= num_samples:
                break
    
    # Concatenate all samples
    expert_usages = np.concatenate(expert_usages, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Plot expert usage by digit class
    plt.figure(figsize=(12, 8))
    for digit in range(10):
        plt.subplot(2, 5, digit + 1)
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) > 0:
            usage = expert_usages[digit_indices].mean(axis=0)
            plt.bar(range(len(usage)), usage)
            plt.title(f'Digit {digit}')
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('expert_usage_by_digit.png')
    plt.close()


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
    num_experts = 4
    k = 2  # Top-k experts to use per sample
    
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
    model = MoEMNISTClassifier(num_experts=num_experts, k=k).to(device)
    print(f"Model created with {num_experts} experts, using top-{k} gating")
    
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
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
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
            torch.save(model.state_dict(), 'model_checkpoints/moe_best.pth')
            print(f"Model saved with accuracy: {best_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Visualize expert usage
    visualize_expert_usage(model, test_loader, device)
    
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
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")


if __name__ == '__main__':
    main()
