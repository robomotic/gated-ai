import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph
from models import MoEMNISTClassifier, IWMNMNISTClassifier

def visualize_moe_architecture():
    """
    Create visualization of the MoE architecture using torchview
    """
    print("Creating MoE architecture visualization...")
    
    # Initialize the MoE model
    model = MoEMNISTClassifier(num_experts=4, k=2)
    
    # Create a sample input
    sample_input = torch.randn(1, 1, 28, 28)
    
    # Create the graph visualization
    graph = draw_graph(
        model, 
        input_data=sample_input,
        expand_nested=True,
        depth=6,
        save_graph=True,
        directory="./",
        filename="moe_architecture_torchview",
        format="png"
    )
    
    print(f"MoE model has {model.count_parameters():,} parameters")
    print("MoE visualization saved as moe_architecture_torchview.png")

def visualize_iwmn_architecture():
    """
    Create visualization of the IWMN architecture using torchview
    """
    print("Creating IWMN architecture visualization...")
    
    # Initialize the IWMN model
    model = IWMNMNISTClassifier(num_iterations=3, modulation_strength=0.1)
    
    # Create a sample input
    sample_input = torch.randn(1, 1, 28, 28)
    
    # Create the graph visualization
    graph = draw_graph(
        model, 
        input_data=sample_input,
        expand_nested=True,
        depth=6,
        save_graph=True,
        directory="./",
        filename="iwmn_architecture_torchview",
        format="png"
    )
    
    print(f"IWMN model has {model.count_parameters():,} parameters")
    print("IWMN visualization saved as iwmn_architecture_torchview.png")

def create_custom_comparison_doc():
    """
    Create a detailed explanation document with custom diagrams 
    showing the key differences between MoE and IWMN
    """
    with open("architecture_comparison.md", "w") as f:
        f.write("""# MoE vs IWMN Architecture Comparison

## Mixture of Experts (MoE)

![MoE Architecture](moe_architecture_torchview.png)

### Key Characteristics:
- **Parallel Processing**: Uses multiple expert networks that process inputs simultaneously
- **Single-Pass Inference**: Input is processed in a single forward pass through the network
- **Gating Mechanism**: A dedicated gating network determines which experts to use for each input
- **Expert Specialization**: Different experts specialize in different parts of the input space
- **Sparse Activation**: Typically only a subset of experts (top-k) are activated for any given input

### Processing Flow:
1. Input enters the system
2. Gating network evaluates input and assigns weights to each expert
3. Input is processed by all experts in parallel
4. Final output is a weighted combination of expert outputs based on gating weights

## Iterative Weight Modulation Network (IWMN)

![IWMN Architecture](iwmn_architecture_torchview.png)

### Key Characteristics:
- **Dynamic Weight Adjustment**: Network weights are modified between iterations
- **Multi-Pass Inference**: Input is processed multiple times with refined weights
- **Adaptive Feedback**: Outputs from previous passes inform weight adjustments
- **Progressive Refinement**: Each iteration improves the prediction

### Processing Flow:
1. Input enters the base network
2. First-pass output is produced
3. Gating controller evaluates the input and output to generate modulation signals
4. Network weights are adjusted based on modulation signals
5. Input is processed again with the adjusted weights
6. Steps 3-5 repeat for a fixed number of iterations
7. Final output comes from the last iteration

## Key Differences

| Feature | MoE | IWMN |
|---------|-----|------|
| Network Structure | Multiple parallel networks | Single network with dynamic weights |
| Inference Pattern | Single-pass | Multi-pass (iterative) |
| Adaptation Mechanism | Expert selection | Weight modulation |
| Parameter Efficiency | Lower (multiple experts) | Higher (weight sharing across iterations) |
| Computational Pattern | Parallel computation | Sequential refinement |

The fundamental conceptual difference is that MoE achieves adaptability through specialization and selection (choosing the right experts), while IWMN achieves adaptability through iteration and refinement (repeatedly adjusting a single network).
""")
    print("Created architecture_comparison.md with detailed explanations")

if __name__ == "__main__":
    try:
        visualize_moe_architecture()
        visualize_iwmn_architecture()
        create_custom_comparison_doc()
        print("Visualizations complete! Check the output files.")
    except Exception as e:
        print(f"Error during visualization: {e}")
