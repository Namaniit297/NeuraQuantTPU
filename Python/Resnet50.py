
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import copy
import numpy as np
import re
from torch.optim.lr_scheduler import CosineAnnealingLR

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
TOTAL_EPOCHS = 100
PRUNING_EPOCHS = 70  # First 70 epochs for pruning
QUANT_EPOCHS = 30    # Last 30 epochs for quantization-aware fine-tuning
TARGET_SPARSITY = 0.7
TARGET_BOPS_REDUCTION = 0.5  # 50% reduction in BOPs

# ----------------------------
# Enhanced ResNet-50 Model
# ----------------------------

def create_enhanced_model():
    """Create enhanced ResNet-50 for CIFAR-10"""
    model = models.resnet50(weights=None)
    
    # Adjust for CIFAR-10's 32x32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Wider final layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 10))
    
    return model

# ----------------------------
# Data Loading with Augmentation
# ----------------------------

def get_data_loaders():
    """Prepare CIFAR-10 data loaders with augmentation"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

# ----------------------------
# Layer-wise Movement Pruner
# ----------------------------

class LayerwiseMovementPruner:
    def __init__(self, model, layer_target_sparsity, warmup_epochs=10):
        self.model = model
        self.layer_target_sparsity = self._process_layer_patterns(layer_target_sparsity)
        self.warmup_epochs = warmup_epochs
        self.masks = {}
        self.stats = {'pruning_progress': []}
        
        # Initialize masks
        for name, param in model.named_parameters():
            if 'weight' in name and self._get_layer_sparsity(name) is not None:
                self.masks[name] = torch.ones_like(param.data)
    
    def _process_layer_patterns(self, config):
        """Convert regex patterns to concrete layer names"""
        concrete_config = {}
        all_layer_names = [name for name, _ in self.model.named_parameters() if 'weight' in name]
        
        for pattern, sparsity in config.items():
            regex = re.compile(pattern.replace('*', '.*'))
            for layer in all_layer_names:
                if regex.fullmatch(layer):
                    concrete_config[layer] = sparsity
        return concrete_config
    
    def _get_layer_sparsity(self, layer_name):
        return self.layer_target_sparsity.get(layer_name, None)
    
    def update_masks(self, epoch, total_epochs):
        current_progress = min((epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs), 1.0)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    target_sparsity = self._get_layer_sparsity(name)
                    current_sparsity = target_sparsity * (1 - (1 - current_progress)**3)
                    
                    # Movement pruning
                    scores = torch.abs(param.grad * param.data)
                    flat_scores = scores.flatten()
                    k = int(current_sparsity * flat_scores.numel())
                    if k > 0:
                        threshold = flat_scores.kthvalue(k).values
                        self.masks[name] = (scores > threshold).float()
        
        # Record stats
        self.stats['pruning_progress'].append({
            'epoch': epoch,
            'sparsity': self.calculate_sparsity()
        })
    
    def calculate_sparsity(self):
        total_params = 0 
        zero_params = 0
        for name, param in self.model.named_parameters():
            if name in self.masks:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0
    
    def apply_masks(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name])

# ----------------------------
# Mixed Precision Quantization
# ----------------------------

def quantize_tensor(tensor, bitwidth):
    """Quantize tensor to specified bitwidth"""
    if bitwidth == 32:
        return tensor
        
    max_val = tensor.abs().max()
    scale = (2**(bitwidth-1)-1) / max_val
    return torch.clamp(torch.round(tensor * scale), -2**(bitwidth-1), 2**(bitwidth-1)-1) / scale

def quantize_model(model, quant_config):
    """Apply mixed-precision quantization"""
    quant_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in quant_model.named_modules():
            if name in quant_config and isinstance(module, (nn.Conv2d, nn.Linear)):
                bitwidth = quant_config[name]
                module.weight.data = quantize_tensor(module.weight.data, bitwidth)
                if module.bias is not None:
                    module.bias.data = quantize_tensor(module.bias.data, bitwidth)
    return quant_model

def calculate_bops(layer, bitwidth, input_shape=(32,32)):
    if isinstance(layer, nn.Conv2d):
        H_out = (input_shape[0] + 2*layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0]-1)-1)//layer.stride[0] + 1
        W_out = (input_shape[1] + 2*layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1]-1)-1)//layer.stride[1] + 1
        return H_out * W_out * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * (bitwidth**2), (H_out, W_out)
    elif isinstance(layer, nn.Linear):
        return layer.in_features * layer.out_features * (bitwidth**2), None
    return 0, input_shape

def analyze_sensitivity(model, test_loader, bitwidths=[8,4,2]):
    """Analyze layer sensitivity to quantization"""
    baseline_acc = evaluate_model(model, test_loader)
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_results = {}
            for bitwidth in bitwidths:
                quant_model = copy.deepcopy(model)
                quant_module = next(m for n,m in quant_model.named_modules() if n == name)
                quant_module.weight.data = quantize_tensor(quant_module.weight.data, bitwidth)
                if quant_module.bias is not None:
                    quant_module.bias.data = quantize_tensor(quant_module.bias.data, bitwidth)
                layer_results[bitwidth] = evaluate_model(quant_model, test_loader)
            
            # Calculate BOPs
            bops = {}
            current_shape = (32,32)
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    b, current_shape = calculate_bops(m, 8, current_shape)
            
            module_bops = {}
            for b in bitwidths:
                bops, _ = calculate_bops(module, b, current_shape)
                module_bops[b] = bops
            
            results[name] = {
                'accuracies': layer_results,
                'delta_acc': {b: baseline_acc - layer_results[b] for b in bitwidths},
                'bops': module_bops
            }
    
    return results

def greedy_mixed_precision(sensitivity_results, target_bops_reduction):
    """Greedy mixed-precision allocation"""
    total_bops = sum(data['bops'][8] for data in sensitivity_results.values())
    target_bops = total_bops * target_bops_reduction
    
    layers = []
    for name, data in sensitivity_results.items():
        delta_bops = data['bops'][8] - data['bops'][4]
        delta_acc = data['delta_acc'][4]
        w = delta_bops / delta_acc if delta_acc != 0 else float('inf')
        layers.append({
            'name': name,
            'w': w,
            'bops_8': data['bops'][8],
            'bops_4': data['bops'][4],
            'delta_bops': delta_bops
        })
    
    layers.sort(key=lambda x: -x['w'])
    quant_config = {layer['name']: 8 for layer in layers}
    current_bops = total_bops
    
    for layer in layers:
        if current_bops <= target_bops:
            break
        quant_config[layer['name']] = 4
        current_bops -= layer['delta_bops']
    
    return quant_config

# ----------------------------
# Training Utilities
# ----------------------------

def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def train_model(model, train_loader, test_loader, epochs, pruner=None):
    """Training loop with optional pruning"""
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if pruner and epoch >= pruner.warmup_epochs:
                pruner.update_masks(epoch, epochs)
                pruner.apply_masks()
            
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation
        acc = evaluate_model(model, test_loader)
        print(f'Epoch {epoch+1}/{epochs}: Accuracy {acc:.2f}%')
        
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
    
    return best_model

# ----------------------------
# Main Training Pipeline
# ----------------------------

def main():
    # Initialize
    train_loader, test_loader = get_data_loaders()
    model = create_enhanced_model().to(device)
    
    # Phase 1: Base Training (30 epochs)
    print("=== Phase 1: Base Training (30 epochs) ===")
    model = train_model(model, train_loader, test_loader, 30)
    base_acc = evaluate_model(model, test_loader)
    print(f"Base model accuracy: {base_acc:.2f}%")
    
    # Phase 2: Pruning (40 epochs)
    print("\n=== Phase 2: Layer-wise Pruning (40 epochs) ===")
    pruner = LayerwiseMovementPruner(
        model,
        layer_target_sparsity={
            'conv1.weight': TARGET_SPARSITY * 0.9,
            'layer1.*.conv*.weight': TARGET_SPARSITY * 0.8,
            'layer2.*.conv*.weight': TARGET_SPARSITY * 0.7,
            'layer3.*.conv*.weight': TARGET_SPARSITY * 0.6,
            'fc.weight': TARGET_SPARSITY * 0.3
        },
        warmup_epochs=10
    )
    model = train_model(model, train_loader, test_loader, 40, pruner)
    pruned_acc = evaluate_model(model, test_loader)
    print(f"Pruned model accuracy: {pruned_acc:.2f}%")
    print(f"Achieved sparsity: {pruner.calculate_sparsity():.1%}")
    
    # Phase 3: Quantization (30 epochs)
    print("\n=== Phase 3: Mixed-Precision Quantization (30 epochs) ===")
    print("Analyzing layer sensitivity...")
    sensitivity = analyze_sensitivity(model, test_loader)
    quant_config = greedy_mixed_precision(sensitivity, TARGET_BOPS_REDUCTION)
    print("Quantization config:", quant_config)
    
    # Quantization-aware fine-tuning
    quant_model = quantize_model(model, quant_config)
    quant_model = train_model(quant_model, train_loader, test_loader, 30)
    final_acc = evaluate_model(quant_model, test_loader)
    print(f"Final quantized accuracy: {final_acc:.2f}%")
    
    # Save results
    torch.save({
        'model_state': quant_model.state_dict(),
        'quant_config': quant_config,
        'accuracy': {
            'base': base_acc,
            'pruned': pruned_acc,
            'final': final_acc
        }
    }, 'compressed_resnet50_cifar10.pth')
    print("Model saved as 'compressed_resnet50_cifar10.pth'")

if __name__ == "__main__":
    main()
