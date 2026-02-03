# VeloGrad Optimizer Implementation

This repository contains the implementation of **VeloGrad**, a novel momentum-based optimizer with dynamic scaling and adaptive decay, as described in the research paper "VeloGrad: A Momentum-Based Optimizer with Dynamic Scaling and Adaptive Decay" by Jayant Biradar and Harsh Wasnik.

## Overview

VeloGrad is designed to enhance deep neural network training by addressing limitations of traditional optimizers like SGD and Adam. It integrates:

- **Gradient norm-based scaling**: Amplifies small gradients and dampens large ones
- **Directional momentum via cosine similarity**: Boosts aligned updates and reduces oscillations
- **Loss-aware learning rate adjustments**: Adapts learning rate based on current loss
- **Adaptive weight decay**: Dynamically adjusts regularization
- **Lookahead mechanism**: Smooths optimization trajectory for better generalization

## Experimental Results (Expected)

Based on the paper, VeloGrad achieves:

| Optimizer | Val Accuracy | Training Loss | F1 Score | Training Time |
|-----------|--------------|---------------|----------|---------------|
| **VeloGrad** | **79.12%** | **0.49** | **0.808** | 415.23s |
| Adam | 77.05% | 0.58 | 0.789 | 410.87s |
| SGD | 72.40% | 0.66 | 0.752 | 408.12s |

**Key Improvements:**
- 2.07% higher accuracy than Adam
- 6.72% higher accuracy than SGD
- 15.5% lower loss than Adam
- 25.8% lower loss than SGD

## Repository Structure

```
optimizers/
├── optimization_report (1).pdf    # Research paper
├── velograd_implementation.ipynb  # Main implementation notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### 1. Clone or navigate to the repository

```bash
cd /Users/jayantbiradar/Desktop/optimizers
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### GPU Support (Recommended)

For faster training, ensure you have:
- CUDA-compatible GPU
- CUDA toolkit installed
- PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Running the Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `velograd_implementation.ipynb`

3. Run all cells sequentially (Cell → Run All)

The notebook will:
- Download CIFAR-10 dataset automatically (first run only)
- Train ResNet-18 with VeloGrad, Adam, and SGD
- Generate comparison plots
- Display performance metrics

### Training Configuration

The notebook uses the following settings (as per the paper):

**VeloGrad:**
- Learning rate: 0.0015
- Betas: (0.9, 0.99)
- Weight decay: 1e-4
- Lookahead interval: 5
- Alpha slow: 0.5
- Alpha interpolation: 0.2

**Adam:**
- Learning rate: 0.001
- Weight decay: 1e-4

**SGD:**
- Learning rate: 0.01
- Momentum: 0.9
- Weight decay: 1e-4

**Common Settings:**
- Epochs: 20
- Batch size: 128
- Mixed-precision training: Enabled
- Gradient accumulation steps: 2

## VeloGrad Algorithm

### Key Components

1. **Selective Gradient Scaling**
```python
if ||g|| <= 1:
    scale = 1 + 0.5(1 - ||g||)
else:
    scale = 1 / ||g||
```

2. **Directional Momentum**
```python
cos_sim = (g_t · g_{t-1}) / (||g_t|| ||g_{t-1}||)
momentum_scale = 1 + 0.1 * cos_sim
```

3. **Hybrid Learning Rate**
```python
loss_scale = min(1, 1/(loss_avg + ε))
norm_scale = min(1, 1/(||g|| + ε))
adaptive_lr = lr * loss_scale * norm_scale
```

4. **Adaptive Weight Decay**
```python
λ_t = λ * min(1, 1/(loss_avg + ε)) * min(1, 1/(||Δθ|| + ε))
```

5. **Lookahead Mechanism** (every k=5 iterations)
```python
θ_slow = θ_slow + α_slow(θ - θ_slow)
θ = (1 - α_interp)θ + α_interp * θ_slow
```

## Expected Output

The notebook will generate:

1. **Training logs** showing progress for each optimizer
2. **Comparison table** with final metrics
3. **Visualization plots**:
   - Training loss curves
   - Validation accuracy curves
   - Loss variance across epochs
   - F1 score progression
4. **Statistical analysis** showing VeloGrad's improvements

## Customization

### Changing Hyperparameters

Edit the optimizer initialization cells:

```python
optimizer_velograd = VeloGrad(
    model.parameters(),
    lr=0.0015,           # Adjust learning rate
    betas=(0.9, 0.99),   # Adjust momentum terms
    weight_decay=1e-4,   # Adjust regularization
    lookahead_k=5,       # Adjust lookahead frequency
    alpha_slow=0.5,      # Adjust slow weights update
    alpha_interp=0.2     # Adjust interpolation
)
```

### Training for More Epochs

Change the `NUM_EPOCHS` variable:

```python
NUM_EPOCHS = 50  # Train for 50 epochs instead of 20
```

### Using Different Dataset

The VeloGrad optimizer can be used with any PyTorch model and dataset. Just replace the data loader and model:

```python
# Your custom dataset
train_loader = DataLoader(your_dataset, batch_size=128, shuffle=True)

# Your custom model
model = YourModel().to(device)

# Use VeloGrad
optimizer = VeloGrad(model.parameters(), lr=0.0015)
```

## Hardware Requirements

**Minimum:**
- CPU: Any modern multi-core processor
- RAM: 8GB
- Storage: 2GB for dataset and checkpoints

**Recommended:**
- GPU: NVIDIA GPU with 6GB+ VRAM (e.g., RTX 3060, V100)
- RAM: 16GB
- Storage: 5GB

**Training Time (20 epochs on CIFAR-10):**
- GPU (RTX 3090): ~7 minutes per optimizer (~21 minutes total)
- CPU: ~2-3 hours per optimizer (~6-9 hours total)

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```python
train_loader, test_loader = get_cifar10_dataloaders(batch_size=64)
```

Or disable mixed-precision training:
```python
USE_AMP = False
```

### Slow Training on CPU

The code will automatically use CPU if GPU is not available. To speed up:
- Use fewer epochs for testing
- Reduce model size
- Use a smaller dataset subset

### CUDA Not Available

Install PyTorch with CUDA support or train on CPU (slower).

## Citation

If you use VeloGrad in your research, please cite:

```bibtex
@article{biradar2025velograd,
  title={VeloGrad: A Momentum-Based Optimizer with Dynamic Scaling and Adaptive Decay},
  author={Biradar, Jayant and Disha},
  journal={arXiv preprint},
  year={2025}
}
```

## References

The VeloGrad optimizer builds upon and is inspired by:

1. **Adam**: Kingma & Ba (2015) - Adaptive moment estimation
2. **Lookahead**: Zhang et al. (2019) - k steps forward, 1 step back
3. **SGD with Momentum**: Polyak (1964) - Classical momentum
4. **RMSProp**: Tieleman & Hinton (2012) - Adaptive learning rates

## License

This implementation is for educational and research purposes. Please refer to the original paper for detailed methodology and theoretical foundations.

## Authors

- Jayant Biradar (jayantbiradar@arizona.edu)
- Disha (disha@arizona.edu)

College of Information, University of Arizona, Tucson, USA

## Acknowledgments

This work demonstrates the potential of adaptive optimization techniques in deep learning. Special thanks to the open-source community for PyTorch and related libraries.
