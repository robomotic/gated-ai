# Comparative Study: Mixture of Experts (MoE) vs Iterative Weight Modulation Networks (IWMN)

This project is a comprehensive study comparing two neural network architectures:

1. **Mixture of Experts (MoE)**: A traditional approach where multiple expert networks are combined with a gating mechanism that dynamically routes inputs.

2. **Iterative Weight Modulation Network (IWMN)**: A novel approach that dynamically adjusts network activations through multiple inference passes.

Both architectures are implemented and evaluated on MNIST digit classification using PyTorch.

## Project Structure

```
.
├── Makefile           # Project automation
├── README.md          # This file
├── data/              # MNIST dataset (downloaded automatically)
├── logs/              # Training logs
├── model_checkpoints/ # Saved model checkpoints
├── requirements.txt   # Python dependencies
├── setup.sh           # Environment setup script
└── src/               # Source code
    ├── models.py      # MoE model implementation
    ├── test.py        # Model evaluation script
    └── train.py       # Model training script
```

## Features

- Mixture of Experts architecture with top-k gating mechanism
- Dynamic expert selection based on input content
- Load balancing to ensure even expert utilization
- MNIST digit classification demo
- Visualization of expert specialization

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Setup

1. Clone the repository
2. Run the setup script to create a virtual environment and install dependencies:

```bash
make setup
source venv/bin/activate
```

## Usage

### Recommended Workflow

For optimal results, we recommend the following workflow sequence:

1. Run hyperparameter search for both models
2. Train final models with optimal parameters
3. Run performance comparisons
4. Analyze results

### Hyperparameter Tuning

Before training the final models, you should run hyperparameter tuning to find optimal configurations for both architectures. The project includes comprehensive grid search implementations for both models.

#### MoE Hyperparameter Search

```bash
make hyperparameter-search
```

This will run a grid search over the following hyperparameters:
- **Learning Rate**: Controls the step size during optimization (typically 0.001, 0.0005, 0.0001)
- **Hidden Size**: Number of neurons in hidden layers (128, 256, 512)
- **Number of Experts**: Total expert networks in the ensemble (4, 8, 16)
- **K Value**: Number of experts activated per input (1, 2, 4)

The hyperparameter search process:
1. Creates a grid of all possible hyperparameter combinations
2. Trains a model for each combination for a fixed number of epochs
3. Evaluates each model on validation data
4. Tracks key metrics:
   - Validation accuracy
   - Test accuracy
   - Training time
   - Parameter count
   - Learning curves
5. Generates detailed visualizations:
   - Learning rate vs. accuracy plots
   - Hidden size vs. accuracy plots
   - Number of experts vs. accuracy plots
   - K value vs. accuracy plots
   - Parameter efficiency comparisons
6. Produces a summary report with the top-performing configurations
7. Saves all results to `hyperparameter_search_results/`

#### IWMN Hyperparameter Search

```bash
make iwmn-hyperparameter-search
```

This will run a grid search over the following IWMN-specific hyperparameters:
- **Learning Rate**: Controls the step size during optimization (typically 0.001, 0.0005, 0.0001)
- **Hidden Size**: Number of neurons in hidden layers (128, 256, 512)
- **Number of Iterations**: How many refinement passes to perform (2, 3, 4, 5)
- **Modulation Strength**: How strongly to adjust activations (0.05, 0.1, 0.2, 0.5)
- **Dropout Rate**: Regularization parameter to prevent overfitting (0.0, 0.2, 0.5)

The IWMN hyperparameter search process:
1. Creates a grid of all possible hyperparameter combinations
2. Trains a model for each combination for a fixed number of epochs
3. Evaluates each model on validation data
4. Tracks key metrics:
   - Validation accuracy
   - Test accuracy
   - Training time
   - Parameter count
   - Inference time per sample
   - Learning curves
5. Generates detailed visualizations:
   - Number of iterations vs. accuracy plots
   - Modulation strength vs. accuracy plots
   - Iteration count vs. inference time trade-offs
   - Accuracy vs. computational cost analysis
   - Learning curves for different configurations
6. Produces a summary report with the top-performing configurations
7. Saves all results to `iwmn_hyperparameter_results/`

#### Analyzing Hyperparameter Search Results

After running the hyperparameter searches, you can analyze the results:

1. **Best Configurations**: Check the summary reports in the respective results directories to find the best-performing hyperparameter combinations.

2. **Performance Visualization**: Review the generated plots to understand how each hyperparameter affects model performance.

3. **Trade-off Analysis**: 
   - For MoE: Analyze the trade-off between number of experts, k value, and accuracy
   - For IWMN: Analyze the trade-off between number of iterations, modulation strength, and inference time

4. **Selecting Final Parameters**: Use the analysis to select the optimal hyperparameters for your specific requirements (accuracy vs. speed vs. model size).

### Training

To train the MoE model on MNIST:

```bash
make train
```

To train the IWMN model on MNIST:

```bash
make train-iwmn
```

These commands will:
- Download the MNIST dataset (if not already downloaded)
- Train the model for multiple epochs
- Save the best model to `model_checkpoints/`
- Generate training curves and visualization

### Testing

To evaluate the trained MoE model:

```bash
make test
```

To evaluate the trained IWMN model:

```bash
make test-iwmn
```

These commands will:
- Load the previously trained model
- Evaluate it on the MNIST test set
- Generate visualizations of model behavior
- Display performance metrics

## Model Architecture

The implemented MoE model consists of:

1. **Expert Networks**: Multiple identical feed-forward networks, each specializing in different parts of the input space.
2. **Gating Network**: A network that determines which experts to use for each input.
3. **Top-k Gating**: For each input, only the top k experts with the highest gating values are used.

The implemented IWMN model consists of:

1. **Base Network**: A standard feed-forward network that processes inputs initially.
2. **Gating Controllers**: Meta-networks that observe the output and generate modulation signals.
3. **Activation Modulation**: A mechanism to adjust activations based on error feedback.
4. **Iterative Processing**: Multiple passes of the same input with progressive refinement.

### Activation-Based Modulation Approach

Our IWMN implementation uses an activation-based modulation approach rather than directly modifying weights. This design choice offers several advantages:

#### How It Works

1. **Initial Forward Pass**: The input is processed through the base network to get an initial output.

2. **Error Calculation**: The difference between the current output and the target is computed.

3. **Modulation Signal Generation**: Based on the input and error, modulation networks generate adjustment signals for each layer's activations.

4. **Activation Adjustment**: Instead of modifying the network weights (which would be memory-intensive), we adjust the activations:
   ```python
   hidden = F.relu(self.fc1(x_flat) + modulation_strength * hidden_mod)
   output = self.fc2(hidden) + modulation_strength * output_mod
   ```

5. **Iterative Refinement**: Steps 2-4 are repeated for a fixed number of iterations, with each pass refining the output.

#### Implementation Details

```python
class SimpleIWMN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_iterations=3, modulation_strength=0.1):
        # Base network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Modulation networks
        self.hidden_modulator = nn.Sequential(
            nn.Linear(input_size + output_size, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
            nn.Tanh()  # Output in [-1, 1] for activation modulation
        )
        
        self.output_modulator = nn.Sequential(
            nn.Linear(input_size + output_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Tanh()  # Output in [-1, 1] for activation modulation
        )
```

#### Advantages Over Weight Modulation

1. **Memory Efficiency**: Modulating activations requires much less memory than modulating weights, making it feasible for larger networks.

2. **Computational Efficiency**: Activation modulation is computationally cheaper than recalculating weights for each input.

3. **Stability**: Direct weight modulation can lead to instability, while activation modulation provides more controlled adjustments.

4. **Adaptability**: The approach allows the network to adapt its behavior for each input without changing its fundamental structure.

## Data Preprocessing

Instead of hardcoding normalization values, we should dynamically compute these statistics from the dataset. Here are recommended best practices for normalization:

### 1. Calculate Statistics from the Training Set

```python
def calculate_normalization_stats(dataset):
    loader = DataLoader(dataset, batch_size=1000, num_workers=4, shuffle=False)
    mean = 0.
    std = 0.
    total_samples = 0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std

# Usage example
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
mean, std = calculate_normalization_stats(train_dataset)
print(f"Dataset mean: {mean.item():.4f}, std: {std.item():.4f}")

# Apply the calculated statistics
transforms.Normalize((mean.item(),), (std.item(),))
```

### 2. Apply Consistent Normalization

Once calculated, apply the same statistics to all splits:

```python
# Create a transform pipeline with dynamic normalization
def create_transforms(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

# Apply to all dataset splits
train_transform = create_transforms(mean.item(), std.item())
val_transform = create_transforms(mean.item(), std.item())
test_transform = create_transforms(mean.item(), std.item())
```

### 3. Cache for Efficiency

For larger datasets, computing statistics can be time-consuming. Calculate once and cache the results:

```python
def get_normalization_stats(dataset_name, recalculate=False):
    stats_file = f"normalization_stats_{dataset_name}.json"
    
    if not recalculate and os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            return stats['mean'], stats['std']
    
    # Calculate stats as shown above
    mean, std = calculate_normalization_stats(dataset)
    
    # Save for future use
    with open(stats_file, 'w') as f:
        json.dump({'mean': mean.item(), 'std': std.item()}, f)
    
    return mean.item(), std.item()
```

### 4. Implementation Results

After implementing dynamic normalization in our project, we obtained the following statistics for the MNIST dataset:

```
Dataset mnist - mean: 0.1307, std: 0.3015
```

These values are very close to the previously hardcoded constants (mean=0.1307, std=0.3081), validating our approach. The slight difference in standard deviation had no negative impact on model performance.

Dynamically computing normalization parameters ensures adaptability to different datasets and better scientific reproducibility by removing hardcoded constants.

## Model Comparison Results

This project compares two neural network architectures:

1. **Mixture of Experts (MoE)**: A traditional approach where multiple expert networks are combined with a gating mechanism.
2. **Iterative Weight Modulation Network (IWMN)**: A novel approach that dynamically adjusts activations during inference.

### Detailed Performance Results

For a comprehensive analysis of model performance, including accuracy metrics, parameter counts, and inference times, please refer to the detailed comparison report:

[Comparison Report](comparison_results/comparison_report.md)

This report includes detailed tables and visualizations comparing both architectures across multiple configurations and metrics. The comparison report is generated automatically when running the `make compare` or `make run-iwmn-experiment` commands.

## Inspiration and Related Work

The IWMN architecture in this project was inspired by (though not a direct implementation of) the work of Danko Nikolić on gating neural networks. For more information on his approach, visit his official website: [https://gating.ai/](https://gating.ai/)

Gating is a result of several decades of scientific work. Here are some key scientific publications that led to Gating technology:

- Nikolić, D. (2023). Where is the mind within the brain? Transient selection of subnetworks by metabotropic receptors and G protein-gated ion channels. Computational Biology and Chemistry, 103, 107820.

- Nikolić, D. (2015). Practopoiesis: Or how life fosters a mind. Journal of Theoretical Biology, 373, 40-61.

- Nikolić, D. (2017). Why deep neural nets cannot ever match biological intelligence and what to do about it?. International Journal of Automation and Computing, 14(5), 532-541.

- Lazar, A., Lewis, C., Fries, P., Singer, W., & Nikolić, D. (2021). Visual exposure enhances stimulus encoding and persistence in primary cortex. Proceedings of the National Academy of Sciences, 118(43), e2105276118.

- Nikolić, D., Häusler, S., Singer, W., & Maass, W. (2009). Distributed fading memory for stimulus properties in the primary visual cortex. PLoS Biology, 7(12), e1000260.

- Nikolić, D. (2009). Is synaesthesia actually ideaestesia? An inquiry into the nature of the phenomenon. In Proceedings of the Third International Congress on Synaesthesia, Science & Art (pp. 26-29).

## Customization

You can modify the hyperparameters in `src/train.py` to experiment with:
- Number of experts
- Value of k in top-k gating
- Hidden layer sizes
- Learning rate and batch size

## Extending the Model

This simple implementation can be extended in several ways:
- Use more complex expert architectures (CNNs, RNNs, etc.)
- Apply to different datasets and tasks
- Implement more sophisticated load balancing
- Add expert dropout for better generalization

## License

This project is open source and available under the MIT License.
