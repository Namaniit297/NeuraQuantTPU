import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
TOTAL_EPOCHS = 100
PRUNING_EPOCHS = 70
QUANT_EPOCHS = 30
TARGET_SPARSITY = 0.7
TARGET_BOPS_REDUCTION = 0.5  # 50% reduction in BOPs

# Data Loading Function
def get_data_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    
    return trainloader, testloader

# VGG-16 Model Definition
class VGG16_CIFAR(nn.Module):
    def __init__(self):
        super(VGG16_CIFAR, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Pruning Class
class LayerwiseMovementPruner:
    def __init__(self, model, layer_target_sparsity, warmup_epochs=10):
        self.model = model
        self.layer_target_sparsity = layer_target_sparsity
        self.warmup_epochs = warmup_epochs
        self.masks = {}
        self.stats = {'pruning_progress': []}
        
        for name, param in model.named_parameters():
            if 'weight' in name and name in layer_target_sparsity:
                self.masks[name] = torch.ones_like(param.data)
    
    def update_masks(self, epoch, total_epochs):
        current_progress = min((epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs), 1.0)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    target_sparsity = self.layer_target_sparsity[name]
                    current_sparsity = target_sparsity * (1 - (1 - current_progress)**3)
                    
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
    
    def apply_masks(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name])
    
    def calculate_sparsity(self):
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            if name in self.masks:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0

# Quantization Functions
def quantize_tensor(tensor, bitwidth):
    if bitwidth == 32:
        return tensor
        
    max_val = tensor.abs().max()
    scale = (2**(bitwidth-1)-1) / max_val
    return torch.clamp(torch.round(tensor * scale), -2**(bitwidth-1), 2**(bitwidth-1)-1) / scale

def quantize_model(model, quant_config):
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

def analyze_sensitivity(model, test_loader, bitwidths=[8,4]):
    """Analyze layer sensitivity to different bitwidths"""
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
                acc = evaluate_model(quant_model, test_loader)
                layer_results[bitwidth] = acc
            
            # Calculate BOPs
            current_shape = (32,32)
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
    """Greedy algorithm for mixed-precision allocation"""
    total_bops = sum(data['bops'][8] for data in sensitivity_results.values())
    target_bops = total_bops * (1 - target_bops_reduction)
    
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
    
    # Sort layers by their benefit-to-cost ratio (descending)
    layers.sort(key=lambda x: -x['w'])
    
    # Start with all layers at 8 bits
    quant_config = {layer['name']: 8 for layer in layers}
    current_bops = total_bops
    
    # Greedily reduce bitwidth where it gives most BOPs reduction for least accuracy loss
    for layer in layers:
        if current_bops <= target_bops:
            break
        quant_config[layer['name']] = 4
        current_bops -= layer['delta_bops']
    
    return quant_config

# Training and Evaluation Functions
def evaluate_model(model, test_loader):
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
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    
    history = {'acc': [], 'loss': []}
    
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
        history['acc'].append(acc)
        history['loss'].append(loss.item())
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
    
    return model, history

# Visualization Functions
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_pruning_progress(pruner_stats):
    pruning_df = pd.DataFrame(pruner_stats['pruning_progress'])
    
    plt.figure(figsize=(8, 5))
    plt.plot(pruning_df['epoch'], pruning_df['sparsity'])
    plt.title('Pruning Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity')
    plt.grid(True)
    plt.savefig('pruning_progress.png')
    plt.close()

# Main Training Pipeline
def main():
    train_loader, test_loader = get_data_loaders()
    model = VGG16_CIFAR().to(device)
    
    # Phase 1: Base Training
    print("=== Phase 1: Base Training (30 epochs) ===")
    model, base_history = train_model(model, train_loader, test_loader, 30)
    base_acc = evaluate_model(model, test_loader)
    print(f"Base model accuracy: {base_acc:.2f}%")
    
    # Phase 2: Pruning
    print("\n=== Phase 2: Pruning (40 epochs) ===")
    pruner = LayerwiseMovementPruner(
        model,
        layer_target_sparsity={
            'features.0.weight': 0.8,
            'features.2.weight': 0.8,
            'features.5.weight': 0.7,
            'features.7.weight': 0.7,
            'features.10.weight': 0.6,
            'features.12.weight': 0.6,
            'features.14.weight': 0.6,
            'features.17.weight': 0.5,
            'features.19.weight': 0.5,
            'features.21.weight': 0.5,
            'features.24.weight': 0.4,
            'features.26.weight': 0.4,
            'features.28.weight': 0.4,
            'classifier.0.weight': 0.3,
            'classifier.3.weight': 0.2,
            'classifier.6.weight': 0.1,
        },
        warmup_epochs=10
    )
    model, pruning_history = train_model(model, train_loader, test_loader, 40, pruner)
    pruned_acc = evaluate_model(model, test_loader)
    print(f"Pruned model accuracy: {pruned_acc:.2f}%")
    print(f"Achieved sparsity: {pruner.calculate_sparsity():.1%}")
    
    # Phase 3: Mixed-Precision Quantization
    print("\n=== Phase 3: Mixed-Precision Quantization (30 epochs) ===")
    print("Analyzing layer sensitivity...")
    sensitivity = analyze_sensitivity(model, test_loader)
    quant_config = greedy_mixed_precision(sensitivity, TARGET_BOPS_REDUCTION)
    print("Quantization config:", quant_config)
    
    # Quantization-aware fine-tuning
    quant_model = quantize_model(model, quant_config)
    quant_model, quant_history = train_model(quant_model, train_loader, test_loader, 30)
    final_acc = evaluate_model(quant_model, test_loader)
    print(f"Final quantized accuracy: {final_acc:.2f}%")
    
    # Generate plots
    full_history = {
        'loss': base_history['loss'] + pruning_history['loss'] + quant_history['loss'],
        'acc': base_history['acc'] + pruning_history['acc'] + quant_history['acc']
    }
    plot_training_history(full_history)
    plot_pruning_progress(pruner.stats)
    
    # Save results
    save_path = 'compressed_vgg16_cifar10.pth'
    torch.save({
        'model_state': quant_model.state_dict(),
        'quant_config': quant_config,
        'accuracy': {
            'base': base_acc,
            'pruned': pruned_acc,
            'final': final_acc
        },
        'history': full_history,
        'pruning_stats': pruner.stats
    }, save_path)
    
    # Verify the file was saved
    import os
    if os.path.exists(save_path):
        print(f"Model successfully saved as '{save_path}'")
        print(f"File size: {os.path.getsize(save_path)/1024/1024:.2f} MB")
    else:
        print("Error: Model file was not saved!")
if __name__ == '__main__':
    main()
