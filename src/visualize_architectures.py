import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as patches

def visualize_moe():
    """Create a visualization of Mixture of Experts (MoE) architecture"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Draw Input
    input_box = Rectangle((10, 45), 15, 10, fc='lightblue', ec='black')
    ax.add_patch(input_box)
    ax.text(17.5, 50, 'Input', ha='center', va='center')
    
    # Draw Gating Network
    gate_box = Rectangle((40, 70), 20, 15, fc='lightgreen', ec='black')
    ax.add_patch(gate_box)
    ax.text(50, 77.5, 'Gating\nNetwork', ha='center', va='center')
    
    # Draw Expert Networks
    expert_boxes = []
    for i in range(4):
        y_pos = 25 + i*15
        expert = Rectangle((40, y_pos), 20, 10, fc='orange', ec='black')
        ax.add_patch(expert)
        ax.text(50, y_pos+5, f'Expert {i+1}', ha='center', va='center')
        expert_boxes.append(expert)
    
    # Draw Output Combination
    output_box = Rectangle((75, 45), 15, 10, fc='lightblue', ec='black')
    ax.add_patch(output_box)
    ax.text(82.5, 50, 'Output', ha='center', va='center')
    
    # Draw Arrows
    # Input to Gating
    arrow1 = FancyArrowPatch((25, 50), (40, 77.5), 
                           connectionstyle="arc3,rad=0.3", 
                           arrowstyle='->', color='black')
    ax.add_patch(arrow1)
    
    # Input to Experts
    for i, expert in enumerate(expert_boxes):
        y_pos = 30 + i*15
        arrow = FancyArrowPatch((25, 50), (40, y_pos), 
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', color='black')
        ax.add_patch(arrow)
    
    # Gating to Experts (dotted lines showing control)
    for i, expert in enumerate(expert_boxes):
        y_pos = 30 + i*15
        arrow = FancyArrowPatch((50, 70), (50, y_pos+10), 
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', color='black', 
                               linestyle='dotted')
        ax.add_patch(arrow)
    
    # Experts to Output
    for i, expert in enumerate(expert_boxes):
        y_pos = 30 + i*15
        arrow = FancyArrowPatch((60, y_pos+5), (75, 50), 
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', color='black')
        ax.add_patch(arrow)
    
    # Title
    ax.set_title('Mixture of Experts (MoE) Architecture', fontsize=14)
    
    # Add explanation
    ax.text(50, 10, 'Single-pass architecture: Input goes through experts only once', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig('moe_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def visualize_iwmn():
    """Create a visualization of Iterative Weight Modulation Network (IWMN) architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Draw Base Network Components
    input_box = Rectangle((10, 45), 15, 10, fc='lightblue', ec='black')
    hidden_box = Rectangle((40, 45), 20, 10, fc='orange', ec='black')
    output_box = Rectangle((80, 45), 15, 10, fc='lightblue', ec='black')
    
    ax.add_patch(input_box)
    ax.add_patch(hidden_box)
    ax.add_patch(output_box)
    
    ax.text(17.5, 50, 'Input', ha='center', va='center')
    ax.text(50, 50, 'Hidden\nLayer', ha='center', va='center')
    ax.text(87.5, 50, 'Output', ha='center', va='center')
    
    # Draw Controller
    controller_box = Rectangle((45, 75), 30, 10, fc='lightgreen', ec='black')
    ax.add_patch(controller_box)
    ax.text(60, 80, 'Gating Controller', ha='center', va='center')
    
    # Draw Weight Modulation
    mod_box1 = Rectangle((40, 20), 20, 10, fc='lightcoral', ec='black')
    ax.add_patch(mod_box1)
    ax.text(50, 25, 'Weight\nModulation', ha='center', va='center')
    
    # Draw arrows for base network
    arrow1 = FancyArrowPatch((25, 50), (40, 50), arrowstyle='->', color='black')
    arrow2 = FancyArrowPatch((60, 50), (80, 50), arrowstyle='->', color='black')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    
    # Draw arrows for the feedback loop
    feedback_arrow1 = FancyArrowPatch((87.5, 45), (60, 75), 
                                      connectionstyle="arc3,rad=-0.3", 
                                      arrowstyle='->', color='black')
    ax.add_patch(feedback_arrow1)
    
    # Draw arrow from input to controller
    input_to_controller = FancyArrowPatch((17.5, 45), (45, 75), 
                                          connectionstyle="arc3,rad=0.3", 
                                          arrowstyle='->', color='black')
    ax.add_patch(input_to_controller)
    
    # Controller to weight modulation
    controller_to_mod = FancyArrowPatch((60, 75), (50, 30), 
                                       connectionstyle="arc3,rad=0", 
                                       arrowstyle='->', color='black')
    ax.add_patch(controller_to_mod)
    
    # Weight modulation to layers
    mod_to_layer = FancyArrowPatch((50, 30), (50, 45), 
                                   connectionstyle="arc3,rad=0", 
                                   arrowstyle='->', color='black')
    ax.add_patch(mod_to_layer)
    
    # Create a dashed circular arrow to indicate iteration
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 5
    x = 105 + radius * np.cos(theta)
    y = 50 + radius * np.sin(theta)
    
    verts = list(zip(x, y))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='black', linestyle='dashed')
    ax.add_patch(patch)
    
    # Add arrowhead to the circular arrow
    arrow_head = FancyArrowPatch((x[75], y[75]), (x[80], y[80]), 
                                arrowstyle='->', color='black')
    ax.add_patch(arrow_head)
    
    # Add iteration text
    ax.text(105, 60, 'Multiple\nIterations', ha='center', va='center')
    
    # Title
    ax.set_title('Iterative Weight Modulation Network (IWMN) Architecture', fontsize=14)
    
    # Add explanation
    ax.text(60, 10, 'Multi-pass architecture: Input goes through the network multiple times\n' +
                    'with weights adjusted between iterations based on previous outputs', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig('iwmn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_visualization():
    """Create a side-by-side comparison visualization of MoE vs IWMN"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # MoE visual in left subplot
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    
    # Draw Input for MoE
    input_box = Rectangle((10, 45), 15, 10, fc='lightblue', ec='black')
    ax1.add_patch(input_box)
    ax1.text(17.5, 50, 'Input', ha='center', va='center')
    
    # Draw Gating Network for MoE
    gate_box = Rectangle((40, 70), 20, 15, fc='lightgreen', ec='black')
    ax1.add_patch(gate_box)
    ax1.text(50, 77.5, 'Gating\nNetwork', ha='center', va='center')
    
    # Draw Expert Networks for MoE
    expert_boxes = []
    for i in range(4):
        y_pos = 25 + i*15
        expert = Rectangle((40, y_pos), 20, 10, fc='orange', ec='black')
        ax1.add_patch(expert)
        ax1.text(50, y_pos+5, f'Expert {i+1}', ha='center', va='center')
        expert_boxes.append(expert)
    
    # Draw Output Combination for MoE
    output_box = Rectangle((75, 45), 15, 10, fc='lightblue', ec='black')
    ax1.add_patch(output_box)
    ax1.text(82.5, 50, 'Output', ha='center', va='center')
    
    # Draw Arrows for MoE
    # Input to Gating
    arrow1 = FancyArrowPatch((25, 50), (40, 77.5), 
                           connectionstyle="arc3,rad=0.3", 
                           arrowstyle='->', color='black')
    ax1.add_patch(arrow1)
    
    # Input to Experts
    for i, expert in enumerate(expert_boxes):
        y_pos = 30 + i*15
        arrow = FancyArrowPatch((25, 50), (40, y_pos), 
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', color='black')
        ax1.add_patch(arrow)
    
    # Gating to Experts (dotted lines showing control)
    for i, expert in enumerate(expert_boxes):
        y_pos = 30 + i*15
        arrow = FancyArrowPatch((50, 70), (50, y_pos+10), 
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', color='black', 
                               linestyle='dotted')
        ax1.add_patch(arrow)
    
    # Experts to Output
    for i, expert in enumerate(expert_boxes):
        y_pos = 30 + i*15
        arrow = FancyArrowPatch((60, y_pos+5), (75, 50), 
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', color='black')
        ax1.add_patch(arrow)
    
    # Title for MoE
    ax1.set_title('Mixture of Experts (MoE)\nSingle-Pass Architecture', fontsize=14)
    
    # MoE key points
    ax1.text(50, 10, '• Parallel expert networks\n• One-time forward pass\n• Expert specialization\n• Selective activation via gating', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # IWMN visual in right subplot
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 100)
    ax2.axis('off')
    
    # Draw Base Network Components for IWMN
    input_box = Rectangle((10, 45), 15, 10, fc='lightblue', ec='black')
    hidden_box = Rectangle((40, 45), 20, 10, fc='orange', ec='black')
    output_box = Rectangle((80, 45), 15, 10, fc='lightblue', ec='black')
    
    ax2.add_patch(input_box)
    ax2.add_patch(hidden_box)
    ax2.add_patch(output_box)
    
    ax2.text(17.5, 50, 'Input', ha='center', va='center')
    ax2.text(50, 50, 'Hidden\nLayer', ha='center', va='center')
    ax2.text(87.5, 50, 'Output', ha='center', va='center')
    
    # Draw Controller for IWMN
    controller_box = Rectangle((45, 75), 30, 10, fc='lightgreen', ec='black')
    ax2.add_patch(controller_box)
    ax2.text(60, 80, 'Gating Controller', ha='center', va='center')
    
    # Draw Weight Modulation for IWMN
    mod_box1 = Rectangle((40, 20), 20, 10, fc='lightcoral', ec='black')
    ax2.add_patch(mod_box1)
    ax2.text(50, 25, 'Weight\nModulation', ha='center', va='center')
    
    # Draw arrows for base network in IWMN
    arrow1 = FancyArrowPatch((25, 50), (40, 50), arrowstyle='->', color='black')
    arrow2 = FancyArrowPatch((60, 50), (80, 50), arrowstyle='->', color='black')
    ax2.add_patch(arrow1)
    ax2.add_patch(arrow2)
    
    # Draw arrows for the feedback loop in IWMN
    feedback_arrow1 = FancyArrowPatch((87.5, 45), (60, 75), 
                                      connectionstyle="arc3,rad=-0.3", 
                                      arrowstyle='->', color='black')
    ax2.add_patch(feedback_arrow1)
    
    # Draw arrow from input to controller in IWMN
    input_to_controller = FancyArrowPatch((17.5, 45), (45, 75), 
                                          connectionstyle="arc3,rad=0.3", 
                                          arrowstyle='->', color='black')
    ax2.add_patch(input_to_controller)
    
    # Controller to weight modulation in IWMN
    controller_to_mod = FancyArrowPatch((60, 75), (50, 30), 
                                       connectionstyle="arc3,rad=0", 
                                       arrowstyle='->', color='black')
    ax2.add_patch(controller_to_mod)
    
    # Weight modulation to layers in IWMN
    mod_to_layer = FancyArrowPatch((50, 30), (50, 45), 
                                   connectionstyle="arc3,rad=0", 
                                   arrowstyle='->', color='black')
    ax2.add_patch(mod_to_layer)
    
    # Create a dashed circular arrow to indicate iteration in IWMN
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 5
    x = 105 + radius * np.cos(theta)
    y = 50 + radius * np.sin(theta)
    
    verts = list(zip(x, y))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='black', linestyle='dashed')
    ax2.add_patch(patch)
    
    # Add arrowhead to the circular arrow in IWMN
    arrow_head = FancyArrowPatch((x[75], y[75]), (x[80], y[80]), 
                                arrowstyle='->', color='black')
    ax2.add_patch(arrow_head)
    
    # Add iteration text for IWMN
    ax2.text(105, 55, 'Multiple\nIterations', ha='center', va='center')
    
    # Title for IWMN
    ax2.set_title('Iterative Weight Modulation Network (IWMN)\nMulti-Pass Architecture', fontsize=14)
    
    # IWMN key points
    ax2.text(60, 10, '• Single base network with weight modulation\n• Multiple forward passes\n• Progressive refinement\n• Dynamic adaptation via feedback', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('moe_vs_iwmn_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_moe()
    visualize_iwmn()
    create_comparison_visualization()
    print("Architecture visualizations created successfully!")
