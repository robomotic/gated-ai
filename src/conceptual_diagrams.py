import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects

def create_moe_diagram():
    """Create a conceptual diagram of MoE architecture"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    title = ax.text(5, 6.5, "Mixture of Experts (MoE)", ha='center', va='center', fontsize=18, fontweight='bold')
    title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Draw input
    input_circle = Circle((1, 3.5), 0.5, fc='#8ecae6', ec='black', lw=2)
    ax.add_patch(input_circle)
    ax.text(1, 3.5, "Input", ha='center', va='center', fontweight='bold')
    
    # Draw gating network
    gate_rect = Rectangle((3, 5), 2, 1, fc='#219ebc', ec='black', lw=2)
    ax.add_patch(gate_rect)
    ax.text(4, 5.5, "Gating Network", ha='center', va='center', fontweight='bold')
    
    # Draw experts
    expert_positions = [(3, 4), (3, 3), (3, 2), (3, 1)]
    for i, pos in enumerate(expert_positions):
        expert = Rectangle((pos[0], pos[1]), 2, 0.7, fc='#ffb703', ec='black', lw=2)
        ax.add_patch(expert)
        ax.text(pos[0] + 1, pos[1] + 0.35, f"Expert {i+1}", ha='center', va='center', fontweight='bold')
    
    # Draw output
    output_circle = Circle((7, 3.5), 0.5, fc='#8ecae6', ec='black', lw=2)
    ax.add_patch(output_circle)
    ax.text(7, 3.5, "Output", ha='center', va='center', fontweight='bold')
    
    # Draw connections
    # Input to gating
    input_to_gate = FancyArrowPatch((1.5, 3.5), (3, 5.5), 
                                    connectionstyle="arc3,rad=0.3", 
                                    arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(input_to_gate)
    
    # Input to experts
    for i, pos in enumerate(expert_positions):
        input_to_expert = FancyArrowPatch((1.5, 3.5), (3, pos[1] + 0.35), 
                                         arrowstyle='->', lw=2, color='#023047')
        ax.add_patch(input_to_expert)
    
    # Gating to experts (control)
    for i, pos in enumerate(expert_positions):
        gate_to_expert = FancyArrowPatch((4, 5), (4, pos[1] + 0.7), 
                                        arrowstyle='->', lw=1.5, color='#023047', linestyle=':')
        ax.add_patch(gate_to_expert)
    
    # Experts to output
    for i, pos in enumerate(expert_positions):
        expert_to_output = FancyArrowPatch((5, pos[1] + 0.35), (6.5, 3.5),
                                         connectionstyle="arc3,rad=0.2", 
                                         arrowstyle='->', lw=2, color='#023047')
        ax.add_patch(expert_to_output)
    
    # Add key points
    explanation_text = (
        "• Single-pass architecture\n"
        "• Multiple expert networks in parallel\n"
        "• Gating network selects relevant experts\n"
        "• Each input processed once\n"
        "• Weighted combination of expert outputs"
    )
    ax.text(5, 0.4, explanation_text, ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)
    
    plt.tight_layout()
    plt.savefig("moe_conceptual.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created MoE conceptual diagram")

def create_iwmn_diagram():
    """Create a conceptual diagram of IWMN architecture"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    title = ax.text(5, 6.5, "Iterative Weight Modulation Network (IWMN)", ha='center', va='center', fontsize=18, fontweight='bold')
    title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Draw input
    input_circle = Circle((1, 3.5), 0.5, fc='#8ecae6', ec='black', lw=2)
    ax.add_patch(input_circle)
    ax.text(1, 3.5, "Input", ha='center', va='center', fontweight='bold')
    
    # Draw base network
    network_rect = Rectangle((3, 3), 2, 1, fc='#ffb703', ec='black', lw=2)
    ax.add_patch(network_rect)
    ax.text(4, 3.5, "Base Network", ha='center', va='center', fontweight='bold')
    
    # Draw controller
    controller_rect = Rectangle((5, 5), 2, 1, fc='#219ebc', ec='black', lw=2)
    ax.add_patch(controller_rect)
    ax.text(6, 5.5, "Gating Controller", ha='center', va='center', fontweight='bold')
    
    # Draw weight modulation
    modulation_rect = Rectangle((3, 1.5), 2, 1, fc='#fb8500', ec='black', lw=2)
    ax.add_patch(modulation_rect)
    ax.text(4, 2, "Weight Modulation", ha='center', va='center', fontweight='bold')
    
    # Draw output
    output_circle = Circle((7, 3.5), 0.5, fc='#8ecae6', ec='black', lw=2)
    ax.add_patch(output_circle)
    ax.text(7, 3.5, "Output", ha='center', va='center', fontweight='bold')
    
    # Draw iterative loop indicator
    loop_circle = Circle((8.5, 2.5), 0.4, fc='none', ec='#023047', lw=2, linestyle='--')
    ax.add_patch(loop_circle)
    
    # Draw arrow for iterative loop
    theta = np.linspace(-0.25*np.pi, 1.5*np.pi, 100)
    radius = 0.3
    x = 8.5 + radius * np.cos(theta)
    y = 2.5 + radius * np.sin(theta)
    ax.plot(x, y, color='#023047', linestyle='--', lw=2)
    
    # Add arrowhead to loop
    arrow_head = FancyArrowPatch((x[-2], y[-2]), (x[-1], y[-1]), 
                                arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(arrow_head)
    ax.text(8.5, 2, "Multiple\nIterations", ha='center', va='center', fontsize=9)
    
    # Draw connections
    # Input to network
    input_to_network = FancyArrowPatch((1.5, 3.5), (3, 3.5), 
                                      arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(input_to_network)
    
    # Network to output
    network_to_output = FancyArrowPatch((5, 3.5), (6.5, 3.5), 
                                       arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(network_to_output)
    
    # Network output to controller
    out_to_controller = FancyArrowPatch((5, 3.7), (6, 5), 
                                       connectionstyle="arc3,rad=0.3", 
                                       arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(out_to_controller)
    
    # Input to controller
    input_to_controller = FancyArrowPatch((1.5, 3.5), (5, 5.5), 
                                         connectionstyle="arc3,rad=0.3", 
                                         arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(input_to_controller)
    
    # Controller to modulation
    controller_to_mod = FancyArrowPatch((6, 5), (4, 2.5), 
                                       connectionstyle="arc3,rad=-0.3", 
                                       arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(controller_to_mod)
    
    # Modulation to network
    mod_to_network = FancyArrowPatch((4, 2.5), (4, 3), 
                                    arrowstyle='->', lw=2, color='#023047')
    ax.add_patch(mod_to_network)
    
    # Add key points
    explanation_text = (
        "• Multi-pass architecture\n"
        "• Single base network with dynamic weights\n"
        "• Controller observes outputs and adjusts weights\n"
        "• Each input processed multiple times\n"
        "• Progressive refinement through iterations"
    )
    ax.text(5, 0.4, explanation_text, ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)
    
    plt.tight_layout()
    plt.savefig("iwmn_conceptual.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created IWMN conceptual diagram")

def create_comparison_document():
    """Create a markdown document explaining the key differences"""
    with open("architecture_comparison.md", "w") as f:
        f.write("""# MoE vs IWMN Architecture Comparison

## Mixture of Experts (MoE)

![MoE Architecture](moe_conceptual.png)

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

![IWMN Architecture](iwmn_conceptual.png)

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
    create_moe_diagram()
    create_iwmn_diagram()
    create_comparison_document()
    print("All visualizations and documentation complete!")
