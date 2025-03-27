import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import argparse

from models import MoEMNISTClassifier, IWMNMNISTClassifier


def test_model(model, test_loader, device):
    """Test the model on the test dataset."""
    model.eval()
    correct = 0
    total = 0
    all_gates = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs, gates = model(data)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Update statistics
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store gates and labels for visualization
            all_gates.append(gates.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    
    # Accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, np.concatenate(all_gates), np.concatenate(all_labels)


def visualize_gates_tsne(gates, labels, num_experts):
    """Visualize gating decisions using t-SNE."""
    print("Generating t-SNE visualization of expert gates...")
    
    # Create a t-SNE embedding of the gates
    tsne = TSNE(n_components=2, random_state=42)
    gates_tsne = tsne.fit_transform(gates)
    
    # Plot the embedding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(gates_tsne[:, 0], gates_tsne[:, 1], 
                          c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("t-SNE Visualization of Expert Gates")
    plt.savefig('gates_tsne.png')
    plt.close()
    
    # Also visualize expert assignment by class
    plt.figure(figsize=(14, 8))
    for digit in range(10):
        plt.subplot(2, 5, digit + 1)
        digit_gates = gates[labels == digit]
        # Get the most active expert for each sample
        expert_indices = np.argmax(digit_gates, axis=1)
        # Count occurrences of each expert
        expert_counts = np.bincount(expert_indices, minlength=num_experts)
        expert_counts = expert_counts / expert_counts.sum()  # Normalize
        plt.bar(range(num_experts), expert_counts)
        plt.title(f'Digit {digit}')
        plt.xlabel('Expert Index')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('expert_distribution.png')
    plt.close()
    print("Visualizations saved to gates_tsne.png and expert_distribution.png")


def visualize_misclassifications(model, test_loader, device, is_iwmn=False):
    """Visualize some misclassified examples."""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if is_iwmn:
                outputs, _, _ = model(data)
            else:
                outputs, _ = model(data)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Find misclassified examples
            incorrect_mask = ~predicted.eq(target)
            misclassified_data = data[incorrect_mask]
            misclassified_targets = target[incorrect_mask]
            misclassified_preds = predicted[incorrect_mask]
            
            # Store up to 10 misclassified examples
            for i in range(min(len(misclassified_data), 10)):
                if len(misclassified) >= 25:  # Limit to 25 examples total
                    break
                misclassified.append((
                    misclassified_data[i].cpu(),
                    misclassified_targets[i].item(),
                    misclassified_preds[i].item()
                ))
            
            if len(misclassified) >= 25:
                break
    
    # Plot misclassified examples
    if misclassified:
        plt.figure(figsize=(12, 10))
        for i, (img, true_label, pred_label) in enumerate(misclassified):
            plt.subplot(5, 5, i + 1)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            plt.title(f'True: {true_label}, Pred: {pred_label}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified.png')
        plt.close()
        print("Misclassified examples saved to misclassified.png")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test MoE or IWMN model on MNIST')
    parser.add_argument('--model', type=str, default='moe', choices=['moe', 'iwmn'],
                        help='Model type to test (moe or iwmn)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for testing')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST test dataset
    test_dataset = datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Create model based on argument
    if args.model.lower() == 'moe':
        # MoE hyperparameters
        num_experts = 4
        k = 2  # Top-k experts to use per sample
        
        # Create MoE model
        model = MoEMNISTClassifier(num_experts=num_experts, k=k).to(device)
        print(f"Model created: MoE with {num_experts} experts, using top-{k} gating")
        model_path = 'model_checkpoints/moe_best.pth'
    else:  # IWMN model
        # IWMN hyperparameters
        num_iterations = 3
        modulation_strength = 0.1
        
        # Create IWMN model
        model = IWMNMNISTClassifier(
            num_iterations=num_iterations,
            modulation_strength=modulation_strength
        ).to(device)
        print(f"Model created: IWMN with {num_iterations} iterations and modulation strength {modulation_strength}")
        model_path = 'model_checkpoints/iwmn_best.pth'
    
    # Load saved model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
        
        # Test the model
        if args.model.lower() == 'moe':
            accuracy, gates, labels = test_model(model, test_loader, device)
            # Visualize gates using t-SNE
            visualize_gates_tsne(gates, labels, num_experts)
            # Visualize misclassifications
            visualize_misclassifications(model, test_loader, device)
        else:  # IWMN model
            # For IWMN, we don't collect gates in the same way, so just test accuracy
            with torch.no_grad():
                correct = 0
                total = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs, _, _ = model(data)
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                
                accuracy = 100 * correct / total
                print(f"Test Accuracy: {accuracy:.2f}%")
                
                # Just visualize misclassifications for IWMN
                visualize_misclassifications(model, test_loader, device, is_iwmn=True)
    else:
        print(f"Model file not found at {model_path}. Please train the model first.")


if __name__ == '__main__':
    main()
