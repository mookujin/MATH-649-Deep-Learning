# Mansur Alimbayev  / 17.10.2025
# MATH 649 HW3 (CNN Hyperparameter Optimization)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# -------------------- Experimental Configuration --------------------
SEED = 42
BATCH_SIZE = 128
DROPOUT_RATE = 0.3

# Balanced hyperparameter search space
DEPTHS = [2, 3, 4]
LEARNING_RATES = [0.01, 0.001, 0.0001]
KERNEL_SIZES = [3, 5]
USE_PADDINGS = [True]
OPTIMIZERS = ['Adam', 'SGD']

# Model architectures
CHANNELS_MNIST = {2: [32, 64], 3: [32, 64, 128], 4: [32, 64, 128, 256]}
CHANNELS_CIFAR = {2: [64, 128], 3: [64, 128, 256], 4: [64, 128, 256, 512]}

# Training parameters
EPOCHS_MNIST = 8
EPOCHS_CIFAR = 12
NUM_CLASSES = 10
FC_HIDDEN = 128
TRAIN_SIZE = 8000
VAL_RATIO = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computational device: {device}')

# -------------------- Data Preparation --------------------
def prepare_datasets():
    """Prepare MNIST and CIFAR-10 datasets with proper splits"""

    # MNIST transforms
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # CIFAR-10 transforms with augmentation
    transform_cifar_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    trainset_mnist_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

    trainset_cifar_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar_train)
    testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar_test)

    # Create train/validation splits
    def create_splits(trainset_full, testset, train_size, val_ratio):
        train_indices = np.random.choice(len(trainset_full), train_size, replace=False)
        trainset_sub = Subset(trainset_full, train_indices)

        val_size = int(train_size * val_ratio)
        train_size_final = train_size - val_size
        trainset, valset = random_split(trainset_sub, [train_size_final, val_size])

        return trainset, valset, testset

    trainset_mnist, valset_mnist, testset_mnist = create_splits(trainset_mnist_full, testset_mnist, TRAIN_SIZE, VAL_RATIO)
    trainset_cifar, valset_cifar, testset_cifar = create_splits(trainset_cifar_full, testset_cifar, TRAIN_SIZE, VAL_RATIO)

    print(f'MNIST - Training: {len(trainset_mnist)}, Validation: {len(valset_mnist)}, Test: {len(testset_mnist)}')
    print(f'CIFAR-10 - Training: {len(trainset_cifar)}, Validation: {len(valset_cifar)}, Test: {len(testset_cifar)}')

    return (trainset_mnist, valset_mnist, testset_mnist,
            trainset_cifar, valset_cifar, testset_cifar)

# -------------------- Neural Network Architecture --------------------
class ConvolutionalNeuralNetwork(nn.Module):
    """
    Flexible CNN architecture with configurable depth, kernel sizes, and channels
    """
    def __init__(self, num_classes=10, in_channels=1, conv_configs=None,
                 dropout_rate=0.3, use_batchnorm=True):
        super(ConvolutionalNeuralNetwork, self).__init__()

        if conv_configs is None:
            conv_configs = [(32, 3)]

        layers = []
        prev_channels = in_channels

        # Build convolutional layers
        for i, (out_channels, kernel_size) in enumerate(conv_configs):
            padding = kernel_size // 2  # Same padding
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2) if i < len(conv_configs) - 1 else nn.Identity()
            ])
            prev_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(prev_channels, FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(FC_HIDDEN, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------- Training Dynamics Tracking --------------------
class TrainingMetrics:
    """Comprehensive tracking of training dynamics and metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.gradient_norms = []
        self.epoch_times = []

    def update_epoch(self, train_loss, val_loss, train_acc, val_acc, lr, grad_norm, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.gradient_norms.append(grad_norm)
        self.epoch_times.append(epoch_time)

    def get_summary(self):
        return {
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else None,
            'final_val_acc': self.val_accuracies[-1] if self.val_accuracies else None,
            'best_val_acc': max(self.val_accuracies) if self.val_accuracies else None,
            'total_training_time': sum(self.epoch_times) if self.epoch_times else None
        }

# -------------------- Training Utilities --------------------
def create_optimizer(model_parameters, optimizer_type, learning_rate):
    """Create optimizer with specified type and parameters"""
    if optimizer_type == 'SGD':
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type == 'Adam':
        return optim.Adam(model_parameters, lr=learning_rate, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def compute_gradient_norm(model):
    """Compute L2 norm of all model gradients"""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def evaluate_model(model, dataloader, criterion, device):
    """Model evaluation on validation or test set"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def train_with_metrics(model, train_loader, val_loader, criterion, optimizer,
                      epochs, device, experiment_name):
    """Complete training procedure with comprehensive metrics tracking"""
    metrics = TrainingMetrics()
    best_val_accuracy = 0.0
    best_model_state = None

    print(f"Training {experiment_name} for {epochs} epochs")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        gradient_norm = compute_gradient_norm(model)

        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        # Update metrics
        metrics.update_epoch(train_loss, val_loss, train_accuracy, val_accuracy,
                           current_lr, gradient_norm, epoch_time)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

        # Progress reporting
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
              f'LR: {current_lr:.2e}, Grad Norm: {gradient_norm:.2f}')

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, metrics, best_val_accuracy

# -------------------- Experimental Framework --------------------
def run_experiments(dataset_name, trainset, valset, testset, input_size,
                   in_channels, channel_config, epochs):
    """Run comprehensive hyperparameter optimization experiments"""
    results = []
    training_dynamics = {}

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

    total_experiments = len(DEPTHS) * len(LEARNING_RATES) * len(KERNEL_SIZES) * len(USE_PADDINGS) * len(OPTIMIZERS)
    experiment_count = 0

    print(f"Commencing {dataset_name} experiments: {total_experiments} configurations")

    for depth in DEPTHS:
        conv_channels = channel_config[depth]
        for learning_rate in LEARNING_RATES:
            for kernel_size in KERNEL_SIZES:
                for use_padding in USE_PADDINGS:
                    for optimizer_type in OPTIMIZERS:
                        experiment_count += 1

                        # Configuration
                        conv_configs = [(ch, kernel_size) for ch in conv_channels]
                        config_id = f"{dataset_name}_d{depth}_lr{learning_rate}_k{kernel_size}_o{optimizer_type}"

                        print(f"\nExperiment {experiment_count}/{total_experiments}: {config_id}")

                        try:
                            # Model initialization
                            model = ConvolutionalNeuralNetwork(
                                num_classes=NUM_CLASSES,
                                in_channels=in_channels,
                                conv_configs=conv_configs,
                                dropout_rate=DROPOUT_RATE
                            ).to(device)

                            criterion = nn.CrossEntropyLoss()
                            optimizer = create_optimizer(model.parameters(), optimizer_type, learning_rate)

                            # Training
                            trained_model, metrics, best_val_accuracy = train_with_metrics(
                                model, train_loader, val_loader, criterion, optimizer,
                                epochs, device, config_id
                            )

                            # Final test evaluation
                            test_loss, test_accuracy = evaluate_model(trained_model, test_loader, criterion, device)
                            training_summary = metrics.get_summary()

                            # Store results
                            result = {
                                'dataset': dataset_name,
                                'depth': depth,
                                'learning_rate': learning_rate,
                                'kernel_size': kernel_size,
                                'padding': use_padding,
                                'optimizer': optimizer_type,
                                'test_accuracy': test_accuracy,
                                'best_val_accuracy': best_val_accuracy,
                                'final_train_accuracy': training_summary['final_train_acc'],
                                'total_training_time': training_summary['total_training_time'],
                                'config_id': config_id
                            }
                            results.append(result)

                            # Store training dynamics
                            training_dynamics[config_id] = metrics

                            print(f"Completed: Test Accuracy = {test_accuracy:.2f}%, "
                                  f"Best Val Accuracy = {best_val_accuracy:.2f}%")

                        except Exception as error:
                            print(f"Experiment failed: {error}")
                            results.append({
                                'dataset': dataset_name,
                                'depth': depth,
                                'learning_rate': learning_rate,
                                'kernel_size': kernel_size,
                                'padding': use_padding,
                                'optimizer': optimizer_type,
                                'test_accuracy': None,
                                'error': str(error)
                            })

    return pd.DataFrame(results), training_dynamics

# -------------------- Analysis and Visualization --------------------
def analyze_results(results_df, dataset_name):
    """Comprehensive analysis of experimental results"""
    valid_results = results_df.dropna(subset=['test_accuracy'])

    if valid_results.empty:
        print(f"No valid results for {dataset_name}")
        return None

    best_result = valid_results.loc[valid_results['test_accuracy'].idxmax()]

    print(f"\n{'='*60}")
    print(f"{dataset_name} RESULTS ANALYSIS")
    print(f"{'='*60}")
    print(f"Best Configuration:")
    print(f"  Test Accuracy: {best_result['test_accuracy']:.2f}%")
    print(f"  Architecture: Depth={best_result['depth']}, "
          f"Kernel={best_result['kernel_size']}, Padding={best_result['padding']}")
    print(f"  Optimization: LR={best_result['learning_rate']}, "
          f"Optimizer={best_result['optimizer']}")

    # Optimizer comparison
    print(f"\nOptimizer Performance:")
    optimizer_stats = valid_results.groupby('optimizer')['test_accuracy'].agg(['mean', 'std'])
    for optimizer, stats in optimizer_stats.iterrows():
        print(f"  {optimizer}: {stats['mean']:.2f}% ± {stats['std']:.2f}%")

    # Depth analysis
    print(f"\nDepth Analysis:")
    depth_stats = valid_results.groupby('depth')['test_accuracy'].agg(['mean', 'std'])
    for depth, stats in depth_stats.iterrows():
        print(f"  Depth {depth}: {stats['mean']:.2f}% ± {stats['std']:.2f}%")

    return best_result

def plot_training_dynamics(training_dynamics, top_configs, dataset_name):
    """Visualize training dynamics for top configurations"""
    if not top_configs:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name} - Training Dynamics Analysis', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_configs)))

    for idx, (config_id, color) in enumerate(zip(top_configs, colors)):
        metrics = training_dynamics[config_id]
        epochs = range(1, len(metrics.train_losses) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, metrics.train_losses, color=color, linewidth=2, label=config_id)
        axes[0, 0].plot(epochs, metrics.val_losses, color=color, linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(epochs, metrics.train_accuracies, color=color, linewidth=2, label=config_id)
        axes[0, 1].plot(epochs, metrics.val_accuracies, color=color, linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[0, 2].plot(epochs, metrics.learning_rates, color=color, linewidth=2, label=config_id)
        axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

        # Gradient norms
        if metrics.gradient_norms:
            axes[1, 0].plot(epochs, metrics.gradient_norms, color=color, linewidth=2, label=config_id)
            axes[1, 0].set_title('Gradient Norms', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)

        # Epoch times
        if metrics.epoch_times:
            axes[1, 1].plot(epochs, metrics.epoch_times, color=color, linewidth=2, label=config_id)
            axes[1, 1].set_title('Epoch Duration', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)

    # Legend
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_comprehensive_analysis(results_df, dataset_name):
    """Create comprehensive analysis plots"""
    valid_results = results_df.dropna(subset=['test_accuracy'])

    if valid_results.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} - Hyperparameter Analysis', fontsize=16, fontweight='bold')

    # Optimizer comparison
    sns.boxplot(data=valid_results, x='optimizer', y='test_accuracy', ax=axes[0, 0])
    axes[0, 0].set_title('Optimizer Performance Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Test Accuracy (%)')

    # Depth analysis
    sns.boxplot(data=valid_results, x='depth', y='test_accuracy', ax=axes[0, 1])
    axes[0, 1].set_title('Network Depth Impact', fontweight='bold')
    axes[0, 1].set_ylabel('Test Accuracy (%)')

    # Learning rate analysis
    sns.scatterplot(data=valid_results, x='learning_rate', y='test_accuracy',
                   hue='optimizer', size='depth', ax=axes[1, 0])
    axes[1, 0].set_title('Learning Rate Sensitivity', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel('Test Accuracy (%)')
    axes[1, 0].set_xlabel('Learning Rate (log scale)')

    # Kernel size analysis
    sns.violinplot(data=valid_results, x='kernel_size', y='test_accuracy', ax=axes[1, 1])
    axes[1, 1].set_title('Kernel Size Impact', fontweight='bold')
    axes[1, 1].set_ylabel('Test Accuracy (%)')

    plt.tight_layout()
    plt.show()

# -------------------- Main Execution --------------------
def main():
    """Main execution function"""
    print("Convolutional Neural Network Hyperparameter Optimization Study")
    print("=" * 70)

    start_time = time.time()

    # Prepare datasets
    (trainset_mnist, valset_mnist, testset_mnist,
     trainset_cifar, valset_cifar, testset_cifar) = prepare_datasets()

    # Run MNIST experiments
    print("\n" + "="*70)
    print("CONDUCTING MNIST EXPERIMENTS")
    print("="*70)
    mnist_results, mnist_dynamics = run_experiments(
        'MNIST', trainset_mnist, valset_mnist, testset_mnist,
        input_size=28, in_channels=1, channel_config=CHANNELS_MNIST, epochs=EPOCHS_MNIST
    )

    # Run CIFAR-10 experiments
    print("\n" + "="*70)
    print("CONDUCTING CIFAR-10 EXPERIMENTS")
    print("="*70)
    cifar_results, cifar_dynamics = run_experiments(
        'CIFAR-10', trainset_cifar, valset_cifar, testset_cifar,
        input_size=32, in_channels=3, channel_config=CHANNELS_CIFAR, epochs=EPOCHS_CIFAR
    )

    # Analysis
    total_duration = time.time() - start_time
    print(f"\nExperimental completion time: {total_duration/60:.1f} minutes")

    # Results analysis
    best_mnist = analyze_results(mnist_results, 'MNIST')
    best_cifar = analyze_results(cifar_results, 'CIFAR-10')

    # Combined results
    all_results = pd.concat([mnist_results, cifar_results])
    valid_results = all_results.dropna(subset=['test_accuracy'])

    print(f"\n{'='*70}")
    print("OVERALL EXPERIMENTAL FINDINGS")
    print(f"{'='*70}")

    if not valid_results.empty:
        top_configurations = valid_results.nlargest(5, 'test_accuracy')[
            ['dataset', 'depth', 'learning_rate', 'kernel_size', 'optimizer', 'test_accuracy']
        ]
        print("\nTop 5 Configurations Across All Experiments:")
        print(top_configurations.to_string(index=False, float_format='%.2f'))

    # Visualizations
    if best_mnist is not None:
        top_mnist_configs = mnist_results.nlargest(3, 'test_accuracy')['config_id'].tolist()
        plot_training_dynamics(mnist_dynamics, top_mnist_configs, 'MNIST')
        plot_comprehensive_analysis(mnist_results, 'MNIST')

    if best_cifar is not None:
        top_cifar_configs = cifar_results.nlargest(3, 'test_accuracy')['config_id'].tolist()
        plot_training_dynamics(cifar_dynamics, top_cifar_configs, 'CIFAR-10')
        plot_comprehensive_analysis(cifar_results, 'CIFAR-10')

    # Save results
    all_results.to_csv('cnn_hyperparameter_optimization_results.csv', index=False)
    print(f"\nResults saved to: cnn_hyperparameter_optimization_results.csv")

    # Summary of key findings
    print(f"\n{'='*70}")
    print("KEY EXPERIMENTAL INSIGHTS")
    print(f"{'='*70}")
    if best_mnist is not None:
        print(f"MNIST Optimal: Depth {best_mnist['depth']}, "
              f"LR {best_mnist['learning_rate']}, {best_mnist['optimizer']} "
              f"→ {best_mnist['test_accuracy']:.2f}%")
    if best_cifar is not None:
        print(f"CIFAR-10 Optimal: Depth {best_cifar['depth']}, "
              f"LR {best_cifar['learning_rate']}, {best_cifar['optimizer']} "
              f"→ {best_cifar['test_accuracy']:.2f}%")

if __name__ == "__main__":
    main()
