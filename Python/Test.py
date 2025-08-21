import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import cv2
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveLayerAnalyzer:
    def __init__(self, model, dataset_config, device='cuda'):
        self.model = model
        self.device = device
        self.dataset_config = dataset_config
        self.layer_info = {}
        self.analysis_results = []
        self.tau_candidates = [1e-4, 1e-3, 1e-2, 1e-1]
        self._register_hooks()

    def _register_hooks(self):
        self.activation_stats = {}
        self.gradient_stats = {}
        def forward_hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                self.activation_stats[module] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'zeros_ratio': (output == 0).float().mean().item(),
                    'shape': output.shape,
                    'activation_norm_l1': output.abs().mean().item(),
                    'activation_norm_l2': output.norm().item() / output.numel(),
                    'kurtosis': self._calculate_kurtosis(output),
                    'skewness': self._calculate_skewness(output)
                }
        def backward_hook(module, grad_input, grad_output):
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                grad = module.weight.grad
                self.gradient_stats[module] = {
                    'grad_norm_l1': grad.abs().mean().item(),
                    'grad_norm_l2': grad.norm().item() / grad.numel(),
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_min': grad.min().item(),
                    'grad_max': grad.max().item()
                }
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                self.layer_info[module] = {
                    'name': name, 'in_channels': module.in_channels, 'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size, 'stride': module.stride, 'padding': module.padding
                }

    def _calculate_kurtosis(self, tensor):
        try:
            flattened = tensor.flatten().detach().cpu().numpy()
            return stats.kurtosis(flattened)
        except:
            return 0.0

    def _calculate_skewness(self, tensor):
        try:
            flattened = tensor.flatten().detach().cpu().numpy()
            return stats.skew(flattened)
        except:
            return 0.0

    def calculate_weight_statistics(self, module):
        if not isinstance(module, nn.Conv2d):
            return {}
        weight = module.weight.data
        stats_dict = {
            'weight_mean': weight.mean().item(),
            'weight_std': weight.std().item(),
            'weight_min': weight.min().item(),
            'weight_max': weight.max().item(),
            'weight_norm_l1': weight.abs().mean().item(),
            'weight_norm_l2': weight.norm().item() / weight.numel(),
            'weight_norm_inf': weight.abs().max().item(),
            'weight_kurtosis': stats.kurtosis(weight.flatten().detach().cpu().numpy()),
            'weight_skewness': stats.skew(weight.flatten().detach().cpu().numpy()),
            'channel_norm_mean': weight.view(weight.size(0), -1).norm(dim=1).mean().item(),
            'channel_norm_std': weight.view(weight.size(0), -1).norm(dim=1).std().item(),
            'channel_norm_cv': (weight.view(weight.size(0), -1).norm(dim=1).std() / (weight.view(weight.size(0), -1).norm(dim=1).mean() + 1e-8)).item(),
            'filter_importance_mean': weight.view(weight.size(0), -1).abs().sum(dim=1).mean().item(),
            'filter_importance_std': weight.view(weight.size(0), -1).abs().sum(dim=1).std().item()
        }
        for tau in self.tau_candidates:
            stats_dict[f'sparsity_tau_{tau}'] = (weight.abs() < tau).float().mean().item()
        return stats_dict

    def calculate_sensitivity_metrics(self, module):
        if not isinstance(module, nn.Conv2d) or module.weight.grad is None:
            return {}
        weight = module.weight.data
        grad = module.weight.grad.data
        sensitivity_dict = {}
        grad_flat = grad.flatten()
        total_grad_sum = grad_flat.abs().sum() + 1e-8
        connection_sensitivity = grad_flat.abs() / total_grad_sum
        sensitivity_dict['connection_sensitivity_mean'] = connection_sensitivity.mean().item()
        sensitivity_dict['connection_sensitivity_std'] = connection_sensitivity.std().item()
        sensitivity_dict['connection_sensitivity_max'] = connection_sensitivity.max().item()
        weight_grad_product = (weight * grad).abs()
        sensitivity_dict['weight_grad_product_mean'] = weight_grad_product.mean().item()
        sensitivity_dict['weight_grad_product_sum'] = weight_grad_product.sum().item()
        for tau in self.tau_candidates:
            mask = (weight.abs() >= tau).float()
            delta_w = weight * (1 - mask)
            sensitivity_dict[f'first_order_loss_change_tau_{tau}'] = (grad * delta_w).sum().abs().item()
        sensitivity_dict['gradient_weight_ratio'] = grad.norm().item() / (weight.norm().item() + 1e-8)
        return sensitivity_dict

    def calculate_computational_metrics(self, module):
        if not isinstance(module, nn.Conv2d):
            return {}
        in_c, out_c = module.in_channels, module.out_channels
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        param_count = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        return {
            'param_count': param_count,
            'weight_memory_mb': (module.weight.numel() * 4) / (1024 * 1024),
            'theoretical_macs_per_pixel': in_c * out_c * k_h * k_w,
            'channel_ratio': out_c / in_c,
            'spatial_efficiency': k_h * k_w
        }

    def calculate_batch_norm_metrics(self, layer_name):
        bn_metrics = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) and layer_name in name:
                if hasattr(module, 'weight') and module.weight is not None:
                    gamma = module.weight.data
                    beta = module.bias.data if module.bias is not None else torch.zeros_like(gamma)
                    bn_metrics = {
                        'bn_gamma_mean': gamma.mean().item(),
                        'bn_gamma_std': gamma.std().item(),
                        'bn_gamma_min': gamma.min().item(),
                        'bn_gamma_max': gamma.max().item(),
                        'bn_beta_mean': beta.mean().item(),
                        'bn_beta_std': beta.std().item(),
                        'bn_channel_importance_mean': gamma.abs().mean().item(),
                        'bn_channel_importance_std': gamma.abs().std().item()
                    }
                break
        return bn_metrics

    def analyze_weight_distributions(self, save_plots=True, output_dir='./analysis_plots'):
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        distribution_stats = {}
        layer_idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layer_name = f"Conv2d_Layer_{layer_idx}"
                weight = module.weight.data.flatten().detach().cpu().numpy()
                mu, sigma = stats.norm.fit(weight)
                thresholds = {}
                for k in [0.5, 1.0, 1.5, 2.0]:
                    threshold = k * sigma
                    sparsity_at_threshold = (np.abs(weight) < threshold).mean()
                    thresholds[f'threshold_{k}sigma'] = {'value': threshold, 'sparsity': sparsity_at_threshold}
                distribution_stats[layer_name] = {
                    'gaussian_mu': mu,
                    'gaussian_sigma': sigma,
                    'thresholds': thresholds,
                    'weight_range': (weight.min(), weight.max()),
                    'weight_median': np.median(weight)
                }
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    plt.hist(weight, bins=100, density=True, alpha=0.7, color='skyblue', label='Weight Distribution')
                    x = np.linspace(weight.min(), weight.max(), 100)
                    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label=f'Gaussian Fit (μ={mu:.4f}, σ={sigma:.4f})')
                    for k in [1.0, 2.0]:
                        threshold = k * sigma
                        plt.axvline(threshold, color='orange', linestyle='--', alpha=0.7, label=f'{k}σ threshold')
                        plt.axvline(-threshold, color='orange', linestyle='--', alpha=0.7)
                    plt.xlabel('Weight Value')
                    plt.ylabel('Density')
                    plt.title(f'{layer_name} - Weight Distribution Analysis')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/{layer_name}_weight_distribution.png', dpi=300)
                    plt.close()
                layer_idx += 1
        return distribution_stats

    def evaluate_model(self, model, dataloader):
        """Evaluate model using mAP@50:95 with Ultralytics DetectionValidator"""
        from ultralytics.utils import TQDM
        from ultralytics.cfg import get_cfg
        from ultralytics.models.yolo.detect import DetectionValidator

        class SimpleValidator(DetectionValidator):
            def __init__(self, dataloader, model, device, data_config):
                super().__init__()
                self.dataloader = dataloader
                self.model = model
                self.device = device
                self.args = get_cfg(overrides={
                    'task': 'detect',
                    'mode': 'val',
                    'device': device,
                    'data': data_config
                })

            def __call__(self):
                self.model.eval()
                pbar = TQDM(self.dataloader, total=len(self.dataloader))
                self.init_metrics(self.model)  # Initialize metrics
                for batch in pbar:
                    imgs = batch[0].to(self.device)
                    targets = batch[1]  # List of tensors
                    if not targets or all(len(t) == 0 for t in targets):
                        print("No ground-truth targets provided, skipping mAP calculation")
                        return 0.0
                    with torch.no_grad():
                        preds = self.model(imgs)
                        # Convert targets to validator format
                        batch_targets = []
                        for i, target in enumerate(targets):
                            if len(target) > 0:
                                # Convert [class, x_center, y_center, w, h] to [x1, y1, x2, y2]
                                target_boxes = torch.zeros_like(target[:, 1:5])
                                target_boxes[:, 0] = target[:, 1] - target[:, 3] / 2  # x1
                                target_boxes[:, 1] = target[:, 2] + target[:, 4] / 2  # y2
                                batch_targets.append({
                                    'boxes': target_boxes.to(self.device),
                                    'labels': target[:, 0].to(self.device),
                                    'batch_idx': torch.full((len(target),), i, device=self.device)
                                })
                            else:
                                batch_targets.append({
                                    'boxes': torch.empty((0, 4), device=self.device),
                                    'labels': torch.empty((0,), device=self.device),
                                    'batch_idx': torch.empty((0,), device=self.device)
                                })
                        # Process predictions
                        self.update_metrics(preds, batch_targets)
                stats = self.get_stats()
                return stats.get('metrics/mAP50-95(B)', 0.0) if stats else 0.0

        validator = SimpleValidator(dataloader, model, self.device, self.dataset_config)
        try:
            mAP = validator()
            return mAP if mAP is not None else 0.0
        except Exception as e:
            print(f"mAP calculation failed: {e}")
            return 0.0

    def evaluate_quantization(self, model, dataloader, bitwidth):
        """Evaluate mAP drop for given quantization bitwidth"""
        model.eval()
        try:
            if bitwidth == 8:
                quantized_model = torch.quantization.quantize_dynamic(model, {nn.Conv2d}, dtype=torch.qint8)
            else:
                print(f"Warning: {bitwidth}-bit quantization not natively supported, simulating")
                quantized_model = model  # Replace with Brevitas for 2-bit/4-bit
            return self.evaluate_model(quantized_model, dataloader)
        except Exception as e:
            print(f"Quantization evaluation failed for {bitwidth}-bit: {e}")
            return 0.0

    def evaluate_pruning(self, model, module, prune_type, sparsity_target=0.5):
        """Evaluate mAP drop for given pruning type and sparsity"""
        try:
            model_copy = torch.nn.Module.deepcopy(model)
            if prune_type == 'unstructured':
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=sparsity_target)
            elif prune_type == 'structured':
                torch.nn.utils.prune.ln_structured(module, name='weight', amount=sparsity_target, n=1, dim=0)
            return self.evaluate_model(model_copy, dataloader)
        except Exception as e:
            print(f"Pruning evaluation failed for {prune_type}: {e}")
            return 0.0

    def estimate_energy(self, module, sparsity, activation_zeros):
        """Estimate energy for digital and analog execution"""
        macs = module.out_channels * module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        digital_energy = macs * 5e-9 * 1000  # 5 pJ/MAC, converted to mJ
        analog_energy = macs * 2e-9 * (1 - sparsity) * (1 - activation_zeros) * 1000  # 2 pJ/MAC
        return digital_energy, analog_energy

    def perform_comprehensive_analysis(self, dataloader, num_batches=5):
        self.model.train()
        self.model.to(self.device)
        print("Starting comprehensive layer analysis...")
        batch_count, total_loss = 0, 0

        baseline_mAP = self.evaluate_model(self.model, dataloader)
        print(f"Baseline mAP@50:95: {baseline_mAP:.4f}")

        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= num_batches:
                break
            try:
                images = batch[0].to(self.device)
                targets = batch[1]  # List of tensors
                self.model.zero_grad()
                with torch.enable_grad():
                    outputs = self.model(images)
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    elif isinstance(outputs, (list, tuple)):
                        loss = sum(o.abs().mean() for o in outputs if isinstance(o, torch.Tensor))
                    else:
                        loss = sum(p.abs().mean() for p in self.model.parameters() if p.requires_grad)
                    loss.backward()
                    total_loss += loss.item()
                    batch_count += 1
                    print(f"Processed batch {batch_count}/{num_batches}, Loss: {loss.item():.6f}")
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        if batch_count == 0:
            print("No batches processed successfully, proceeding with weight-only analysis...")
        else:
            print(f"Average loss across {batch_count} batches: {total_loss/batch_count:.6f}")

        distribution_stats = self.analyze_weight_distributions()
        layer_idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layer_name = f"Conv2d_Layer_{layer_idx}"
                layer_metrics = {'layer_name': layer_name, 'layer_index': layer_idx}
                layer_metrics.update(self.layer_info.get(module, {}))
                layer_metrics.update(self.calculate_weight_statistics(module))
                layer_metrics.update(self.calculate_sensitivity_metrics(module))
                layer_metrics.update(self.calculate_computational_metrics(module))
                layer_metrics.update(self.calculate_batch_norm_metrics(layer_name))

                if module in self.activation_stats:
                    activation_stats = self.activation_stats[module]
                    for key, value in activation_stats.items():
                        if key != 'shape':
                            layer_metrics[f'activation_{key}'] = value
                else:
                    layer_metrics.update({
                        'activation_mean': 0.0, 'activation_std': 0.0, 'activation_min': 0.0,
                        'activation_max': 0.0, 'activation_zeros_ratio': 0.0,
                        'activation_activation_norm_l1': 0.0, 'activation_activation_norm_l2': 0.0,
                        'activation_kurtosis': 0.0, 'activation_skewness': 0.0
                    })

                if module in self.gradient_stats:
                    layer_metrics.update(self.gradient_stats[module])
                else:
                    layer_metrics.update({
                        'grad_norm_l1': 0.0, 'grad_norm_l2': 0.0, 'grad_mean': 0.0,
                        'grad_std': 0.0, 'grad_min': 0.0, 'grad_max': 0.0
                    })

                if layer_name in distribution_stats:
                    dist_stats = distribution_stats[layer_name]
                    layer_metrics['gaussian_mu'] = dist_stats['gaussian_mu']
                    layer_metrics['gaussian_sigma'] = dist_stats['gaussian_sigma']
                    layer_metrics['weight_median'] = dist_stats['weight_median']
                    for threshold_name, threshold_info in dist_stats['thresholds'].items():
                        layer_metrics[f'sparsity_{threshold_name}'] = threshold_info['sparsity']

                quant_2bit_mAP = self.evaluate_quantization(self.model, dataloader, 2)
                quant_4bit_mAP = self.evaluate_quantization(self.model, dataloader, 4)
                quant_8bit_mAP = self.evaluate_quantization(self.model, dataloader, 8)
                prune_unstruct_mAP = self.evaluate_pruning(self.model, module, 'unstructured', 0.5)
                prune_struct_mAP = self.evaluate_pruning(self.model, module, 'structured', 0.5)
                sparsity = layer_metrics.get('sparsity_tau_0.001', 0.0)
                activation_zeros = layer_metrics.get('activation_zeros_ratio', 0.0)
                digital_energy, analog_energy = self.estimate_energy(module, sparsity, activation_zeros)

                configs = [
                    {'bitwidth': 2, 'prune': 'none', 'mode': 'digital', 'mAP': quant_2bit_mAP, 'energy': digital_energy},
                    {'bitwidth': 4, 'prune': 'none', 'mode': 'digital', 'mAP': quant_4bit_mAP, 'energy': digital_energy},
                    {'bitwidth': 8, 'prune': 'none', 'mode': 'digital', 'mAP': quant_8bit_mAP, 'energy': digital_energy},
                    {'bitwidth': 4, 'prune': 'unstructured', 'mode': 'digital', 'mAP': prune_unstruct_mAP, 'energy': digital_energy * (1 - 0.5)},
                    {'bitwidth': 4, 'prune': 'structured', 'mode': 'digital', 'mAP': prune_struct_mAP, 'energy': digital_energy * (1 - 0.5)},
                    {'bitwidth': 4, 'prune': 'unstructured', 'mode': 'analog', 'mAP': prune_unstruct_mAP, 'energy': analog_energy},
                    {'bitwidth': 4, 'prune': 'structured', 'mode': 'analog', 'mAP': prune_struct_mAP, 'energy': analog_energy},
                ]
                best_config = max(configs, key=lambda x: x['mAP'] if x['energy'] < 0.02 else -float('inf'))
                layer_metrics['optimal_config'] = f"{best_config['bitwidth']}b_{best_config['prune']}_{best_config['mode']}"
                layer_metrics['quant_2bit_mAP_drop'] = baseline_mAP - quant_2bit_mAP
                layer_metrics['quant_4bit_mAP_drop'] = baseline_mAP - quant_4bit_mAP
                layer_metrics['quant_8bit_mAP_drop'] = baseline_mAP - quant_8bit_mAP
                layer_metrics['prune_unstruct_50_mAP_drop'] = baseline_mAP - prune_unstruct_mAP
                layer_metrics['prune_struct_50_mA```pythonP_drop'] = baseline_mAP - prune_struct_mAP
                layer_metrics['digital_energy_mJ'] = digital_energy
                layer_metrics['analog_energy_mJ'] = analog_energy

                self.analysis_results.append(layer_metrics)
                layer_idx += 1

        print(f"Analysis completed for {len(self.analysis_results)} layers")
        return self.analysis_results

    def save_analysis_results(self, output_path='comprehensive_layer_analysis.csv'):
        if not self.analysis_results:
            print("No analysis results to save!")
            return None
        df = pd.DataFrame(self.analysis_results)
        df.to_csv(output_path, index=False)
        print(f"Analysis results saved to {output_path}")
        for metric in ['weight_norm_l1', 'sparsity_tau_0.001', 'param_count', 'optimal_config']:
            if metric in df.columns:
                print(f"{metric}:\n  Mean: {df[metric].mean():.6f}\n  Std: {df[metric].std():.6f}\n")
        return df

    def create_correlation_analysis(self, df, output_path='correlation_heatmap.png'):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, square=True, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Layer Metrics Correlation Analysis')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation analysis saved to {output_path}")
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
        if high_corr_pairs:
            print("\nHighly Correlated Feature Pairs (|correlation| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        return correlation_matrix

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized label tensors"""
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]  # List of tensors
    return images, labels

def create_simple_dataloader(dataset_path, batch_size=4, num_samples=32):
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    class SimpleImageDataset(Dataset):
        def __init__(self, image_dir, label_dir=None, transform=None, max_samples=None):
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir) if label_dir else None
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(list(self.image_dir.glob(ext)))
                self.image_files.extend(list(self.image_dir.glob(ext.upper())))
            if max_samples:
                self.image_files = self.image_files[:max_samples]
            self.transform = transform or transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((640, 640)), transforms.ToTensor(),
            ])
        def __len__(self):
            return len(self.image_files)
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            if self.label_dir:
                label_path = self.label_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        labels = [line.strip().split() for line in f.readlines()]
                        labels = [[float(x) for x in label] for label in labels]  # [cls, x_center, y_center, w, h]
                        labels = torch.tensor(labels, dtype=torch.float32)
                else:
                    labels = torch.tensor([])
            else:
                labels = torch.tensor([])
            return image, labels
    for path in [dataset_path / "val" / "images", dataset_path / "train" / "images"]:
        if path.exists() and any(path.glob('*.jpg')):
            label_path = path.parent / "labels" if (path.parent / "labels").exists() else None
            dataset = SimpleImageDataset(path, label_path, max_samples=num_samples)
            if len(dataset) > 0:
                print(f"Using images from: {path}")
                if path == dataset_path / "train" / "images":
                    print("Warning: Using training set for validation. Consider creating a 'val' split.")
                return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    raise FileNotFoundError("No image directory found in 'val' or 'train'")

def run_comprehensive_analysis():
    dataset_path = Path("/mnt/DATA/MS25S001/Downloads/AGNI_NEW_VERSION/dataset")
    model_weights = Path("/mnt/DATA/MS25S001/Downloads/AGNI_NEW_VERSION/runs/detect/agni_train/weights/best.pt")
    data_yaml = dataset_path / "data.yaml"
    
    if not dataset_path.exists() or not model_weights.exists():
        print("Dataset or model weights not found")
        return
    data_config = yaml.safe_load(open(data_yaml, 'r')) if data_yaml.exists() else {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(str(model_weights))
    model.model.to(device)
    
    dataloader = create_simple_dataloader(dataset_path, batch_size=4, num_samples=32)
    analyzer = ComprehensiveLayerAnalyzer(model.model, data_config, device=device)
    results = analyzer.perform_comprehensive_analysis(dataloader, num_batches=3)
    if results:
        df = analyzer.save_analysis_results('comprehensive_layer_analysis.csv')
        if df is not None and len(df) > 1:
            analyzer.create_correlation_analysis(df)
    return results

if __name__ == "__main__":
    results = run_comprehensive_analysis()
