import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================== MoE Implementation ==========================
class Expert(nn.Module):
    """
    Single expert network for the MoE model.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """
    Gating network that decides which experts to use for each input.
    """
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model with a gating network and multiple expert networks.
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts, k=2):
        super(MixtureOfExperts, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.k = min(k, num_experts)  # Top-k gating: using only k experts per sample
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size) for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gating_network = GatingNetwork(input_size, num_experts)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input for gating network
        reshaped_x = x.view(batch_size, -1)
        
        # Get gates (expert weights)
        gates = self.gating_network(reshaped_x)
        
        # Get outputs from all experts
        expert_outputs = torch.zeros(batch_size, self.num_experts, 10, device=x.device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, i] = expert(reshaped_x)
        
        # Implement top-k gating: only keep the k experts with highest gating values
        if self.k < self.num_experts:
            # Find the top-k experts
            _, indices = torch.topk(gates, self.k, dim=1)
            
            # Create a mask to zero out non-top-k expert outputs
            mask = torch.zeros_like(gates)
            mask.scatter_(1, indices, 1.0)
            
            # Normalize the gates for the top-k experts
            gates = gates * mask
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-12)
        
        # Combine expert outputs using the gating weights
        # gates shape: [batch_size, num_experts]
        # expert_outputs shape: [batch_size, num_experts, output_size]
        # weighted_outputs: [batch_size, output_size]
        gates = gates.unsqueeze(2)  # Add output dimension: [batch_size, num_experts, 1]
        weighted_outputs = (expert_outputs * gates).sum(dim=1)
        
        return weighted_outputs, gates.squeeze(2)


class MoEMNISTClassifier(nn.Module):
    """
    MoE model for MNIST classification.
    """
    def __init__(self, num_experts=4, k=2, hidden_size=256):
        super(MoEMNISTClassifier, self).__init__()
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = hidden_size  # Configurable hidden size
        self.output_size = 10  # 10 digit classes
        
        self.moe = MixtureOfExperts(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_experts=num_experts,
            k=k
        )
        
    def forward(self, x):
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Forward through MoE
        outputs, gates = self.moe(x)
        
        return outputs, gates
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========================== IWMN Implementation ==========================
class WeightModulationLayer(nn.Module):
    """
    Layer that modulates weights of a target layer based on gating signals.
    Memory-efficient implementation using dimensionality reduction.
    """
    def __init__(self, target_size, max_hidden_dim=512):
        super(WeightModulationLayer, self).__init__()
        # Limit the maximum hidden dimension to prevent memory issues
        hidden_dim = min(target_size // 4, max_hidden_dim)  # Significantly reduced hidden dimension
        projection_dim = min(target_size, 1024)  # Cap the projection dimension
        
        # Use a bottleneck architecture to reduce parameters
        if target_size > projection_dim:
            # For very large target sizes, use a low-rank approximation approach
            self.compress = nn.Linear(target_size, projection_dim)
            self.expand = nn.Linear(projection_dim, target_size)
            self.modulation = nn.Sequential(
                nn.Linear(projection_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, projection_dim),
                nn.Tanh()  # Output in [-1, 1] for weight adjustments
            )
        else:
            # For smaller target sizes, use a standard bottleneck
            self.compress = nn.Identity()
            self.expand = nn.Identity()
            self.modulation = nn.Sequential(
                nn.Linear(target_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_size),
                nn.Tanh()  # Output in [-1, 1] for weight adjustments
            )
        
    def forward(self, weights):
        # Generate modulation values for the weights using the bottleneck architecture
        compressed = self.compress(weights)
        modulated = self.modulation(compressed)
        return self.expand(modulated)


class GatingController(nn.Module):
    """
    Meta-network that observes outputs and generates weight modulation signals.
    Memory-efficient implementation using factorized architecture.
    """
    def __init__(self, input_size, output_size, target_size, max_output_dim=1024):
        super(GatingController, self).__init__()
        
        # Limit the output dimension to prevent memory issues
        self.target_size = target_size
        hidden_dim = 256  # Fixed hidden dimension for the controller
        
        # Use a more memory-efficient factorized approach for large target sizes
        if target_size > max_output_dim:
            # Instead of directly producing a large output vector, produce factors that can be
            # combined to reconstruct a large output through outer product
            factor_size = int(math.sqrt(max_output_dim)) 
            self.factorized = True
            self.factor_size = factor_size
            
            # Produce two factors whose outer product will approximate the large output
            self.controller = nn.Sequential(
                nn.Linear(input_size + output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.factor1_head = nn.Linear(hidden_dim, factor_size)
            self.factor2_head = nn.Linear(hidden_dim, factor_size)
        else:
            # For smaller target sizes, use a standard approach
            self.factorized = False
            self.controller = nn.Sequential(
                nn.Linear(input_size + output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, target_size)
            )
        
    def forward(self, x, y):
        # Combine input and output
        combined = torch.cat([x, y], dim=1)
        
        if self.factorized:
            # Generate factors and combine them to approximate a large output
            features = self.controller(combined)
            factor1 = self.factor1_head(features)
            factor2 = self.factor2_head(features)
            
            # Create outer product and reshape to target size
            # This creates a low-rank approximation of the full matrix
            batch_size = x.size(0)
            outer_product = torch.bmm(
                factor1.unsqueeze(2),  # [batch_size, factor_size, 1]
                factor2.unsqueeze(1)   # [batch_size, 1, factor_size]
            )  # Result: [batch_size, factor_size, factor_size]
            
            # Reshape to match as much of the target size as possible
            flattened = outer_product.view(batch_size, -1)  # [batch_size, factor_size*factor_size]
            
            # Pad or truncate to match target size
            if flattened.size(1) >= self.target_size:
                return flattened[:, :self.target_size]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(batch_size, self.target_size - flattened.size(1), device=x.device)
                return torch.cat([flattened, padding], dim=1)
        else:
            # Standard forward pass for smaller target sizes
            return self.controller(combined)


import math  # Add this import at the top of the file

class IterativeWeightModulationNetwork(nn.Module):
    """
    Iterative Weight Modulation Network (IWMN) with gating controllers and modulation layers.
    Memory-efficient implementation with size limits.
    """
    def __init__(self, input_size, hidden_size, output_size, num_iterations=3, modulation_strength=0.1, max_layer_size=1000000):
        super(IterativeWeightModulationNetwork, self).__init__()
        self.input_size = input_size
        self.num_iterations = num_iterations
        self.modulation_strength = modulation_strength
        
        # Apply size limits to prevent memory issues
        self.hidden_size = min(hidden_size, 512)  # Cap the hidden size
        self.output_size = output_size
        
        # Base network
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output_size)
        
        # Calculate target sizes (number of parameters in each layer)
        self.fc1_size = input_size * self.hidden_size + self.hidden_size  # weights + biases
        self.fc2_size = self.hidden_size * output_size + output_size  # weights + biases
        
        # Enforce maximum layer size
        if self.fc1_size > max_layer_size or self.fc2_size > max_layer_size:
            raise ValueError(f"Layer size exceeds maximum allowed ({max_layer_size}). "
                             f"FC1: {self.fc1_size}, FC2: {self.fc2_size}. "
                             f"Reduce input_size or hidden_size.")
        
        # Gating controllers for each layer - use memory-efficient implementation
        self.fc1_controller = GatingController(input_size, output_size, self.fc1_size, max_output_dim=1024)
        self.fc2_controller = GatingController(input_size, output_size, self.fc2_size, max_output_dim=1024)
        
        # Weight modulation layers - use memory-efficient implementation
        self.fc1_modulation = WeightModulationLayer(self.fc1_size, max_hidden_dim=512)
        self.fc2_modulation = WeightModulationLayer(self.fc2_size, max_hidden_dim=512)
        
    def _get_flat_params(self, layer):
        """Get flattened parameters (weights and biases) from a layer."""
        params = []
        for p in layer.parameters():
            params.append(p.view(-1))
        return torch.cat(params)
    
    def _set_params(self, layer, flat_params):
        """Set the parameters of a layer from flattened parameters."""
        offset = 0
        for param in layer.parameters():
            param_size = param.numel()
            param.data = flat_params[offset:offset + param_size].view(param.size())
            offset += param_size
    
    def forward(self, x, target=None, return_modulations=False):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Initial forward pass
        h = F.relu(self.fc1(x_flat))
        output = self.fc2(h)
        
        # If no target is provided, just return the initial output
        if target is None:
            return output, None if not return_modulations else (output, None, [])
        
        modulation_history = [] if return_modulations else None
        
        # Iterative refinement
        for iteration in range(self.num_iterations):
            # Get current parameters
            fc1_params = self._get_flat_params(self.fc1)
            fc2_params = self._get_flat_params(self.fc2)
            
            # Generate control signals based on current output and target
            error = target - output
            fc1_control = self.fc1_controller(x_flat, error)
            fc2_control = self.fc2_controller(x_flat, error)
            
            # Generate weight modulations
            fc1_modulation = self.fc1_modulation(fc1_control)
            fc2_modulation = self.fc2_modulation(fc2_control)
            
            # Apply modulations to weights
            fc1_new_params = fc1_params + self.modulation_strength * fc1_modulation
            fc2_new_params = fc2_params + self.modulation_strength * fc2_modulation
            
            # Set new parameters
            self._set_params(self.fc1, fc1_new_params)
            self._set_params(self.fc2, fc2_new_params)
            
            # Forward pass with new weights
            h = F.relu(self.fc1(x_flat))
            output = self.fc2(h)
            
            # Store modulation info if needed
            if return_modulations:
                modulation_info = {
                    'iteration': iteration,
                    'fc1_modulation': fc1_modulation.detach().cpu(),
                    'fc2_modulation': fc2_modulation.detach().cpu(),
                    'output': output.detach().cpu()
                }
                modulation_history.append(modulation_info)
        
        return output, error, modulation_history if return_modulations else (output, error)
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Optimized IWMN implementation for memory efficiency and accuracy
class SimpleIWMN(nn.Module):
    """
    An optimized and memory-efficient Iterative Weight Modulation Network.
    Instead of directly modulating weights, this version uses additive modulation
    at the activation level, which is much more memory efficient while maintaining
    high accuracy.
    """
    def __init__(self, input_size, hidden_size, output_size, num_iterations=3, modulation_strength=0.1, dropout_rate=0.2):
        super(SimpleIWMN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_iterations = num_iterations
        self.modulation_strength = modulation_strength
        
        # Base network with batch normalization for better training stability
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Enhanced modulation networks with deeper architecture
        modulator_hidden = 128  # Increased capacity for better modulation
        
        self.hidden_modulator = nn.Sequential(
            nn.Linear(input_size + output_size, modulator_hidden),
            nn.BatchNorm1d(modulator_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  # Lighter dropout in modulators
            nn.Linear(modulator_hidden, modulator_hidden // 2),
            nn.ReLU(),
            nn.Linear(modulator_hidden // 2, hidden_size),
            nn.Tanh()  # Output in [-1, 1] for activation modulation
        )
        
        self.output_modulator = nn.Sequential(
            nn.Linear(input_size + output_size, modulator_hidden),
            nn.BatchNorm1d(modulator_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(modulator_hidden, modulator_hidden // 2),
            nn.ReLU(),
            nn.Linear(modulator_hidden // 2, output_size),
            nn.Tanh()  # Output in [-1, 1] for activation modulation
        )
    
    def forward(self, x, target=None, return_modulations=False):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Initial forward pass with improved base network
        hidden = self.fc1(x_flat)
        hidden = self.bn1(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        
        # If no target is provided, just return the initial output
        if target is None:
            return output, None if not return_modulations else (output, None, [])
        
        modulation_history = [] if return_modulations else None
        
        # Iterative refinement
        for iteration in range(self.num_iterations):
            # Calculate error
            error = target - output
            
            # Create input for modulators
            modulator_input = torch.cat([x_flat, error], dim=1)
            
            # Generate modulation signals
            hidden_mod = self.hidden_modulator(modulator_input)
            output_mod = self.output_modulator(modulator_input)
            
            # Apply modulation to activations (not weights)
            # This is much more memory efficient
            hidden = self.fc1(x_flat) + self.modulation_strength * hidden_mod
            hidden = self.bn1(hidden)  # Apply batch norm for stability
            hidden = F.relu(hidden)
            # No dropout during iterative refinement to maintain stability
            output = self.fc2(hidden) + self.modulation_strength * output_mod
            
            # Store modulation info if needed
            if return_modulations:
                modulation_info = {
                    'iteration': iteration,
                    'hidden_mod': hidden_mod.detach().cpu(),
                    'output_mod': output_mod.detach().cpu(),
                    'output': output.detach().cpu()
                }
                modulation_history.append(modulation_info)
        
        return output, error, modulation_history if return_modulations else (output, error)
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IWMNMNISTClassifier(nn.Module):
    """
    IWMN model for MNIST classification.
    Memory-efficient implementation.
    """
    def __init__(self, num_iterations=3, modulation_strength=0.1, hidden_size=128, dropout_rate=0.2):
        super(IWMNMNISTClassifier, self).__init__()
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = hidden_size  # Configurable hidden size
        self.output_size = 10      # 10 digit classes
        
        # Create simplified IWMN that modulates activations instead of weights
        self.iwmn = SimpleIWMN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_iterations=num_iterations,
            modulation_strength=modulation_strength,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x, target=None, return_modulations=False):
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Forward through IWMN
        return self.iwmn(x, target, return_modulations)
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return self.iwmn.count_parameters()
