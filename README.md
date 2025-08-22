# Physics-Informed Neural Networks for Heat Equation

A comprehensive implementation of Physics-Informed Neural Networks (PINNs) for solving both forward and inverse heat equation problems. This project demonstrates the power of scientific machine learning by incorporating physical laws directly into neural network training, enabling parameter recovery from sparse measurements.

## Overview

This repository implements PINNs to solve the heat equation:

```
∂u/∂t = ∂/∂x(κ(x) ∂u/∂x)
```

where:
- `u(t,x)` is the temperature field
- `κ(x)` is the thermal conductivity
- Initial condition: `u(0,x) = sin(πx)`
- Boundary conditions: `u(t,0) = u(t,1) = 0`

## Key Features

- **Forward Problem Solving**: Compute temperature evolution with known diffusion coefficient
- **Inverse Problem Capability**: Recover unknown diffusion coefficient from sparse measurements
- **Physics-Informed Training**: Enforce PDEs, initial conditions, and boundary conditions directly in loss function
- **Comparison with FEM**: Benchmark against traditional finite element methods
- **Continuous Solutions**: Mesh-free approach providing differentiable solutions across entire domain

## Requirements

```bash
torch>=1.9.0
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.7.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Physics-Informed-Neural-Networks-for-Heat-Equation.git
cd Physics-Informed-Neural-Networks-for-Heat-Equation
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib scipy
```

## Usage

### Forward Problem

Solve the heat equation with known constant diffusion coefficient (κ = 1):

```python
# Run forward problem
python forward_problem.py
```

This will:
- Train a PINN to solve the forward heat equation
- Generate visualizations of the temperature evolution
- Compare results with analytical solution
- Save sampled data for the inverse problem

### Inverse Problem

Recover the diffusion coefficient from sparse temperature measurements:

```python
# Run inverse problem (requires forward_problem.py to be run first)
python inverse_problem.py
```

This will:
- Use 100 sparse measurements from the forward solution
- Simultaneously learn temperature field and diffusion coefficient
- Visualize the recovered parameters

### Comparison with FEM

Compare PINN performance against traditional finite element methods:

```python
# Run comparison analysis
python fem_comparison.py
```

## Results

### Forward Problem Performance
- **PINN Error vs Analytical**: Max = 0.026427, Mean = 0.006583
- **Training Time**: ~82 seconds (5000 epochs)
- **Parameters**: 1,341 neural network parameters

### Inverse Problem Performance
- **Parameter Recovery**: Successfully recovered κ(x) = 1.0
- **Mean Absolute Error**: 0.066057
- **Training from 100 sparse points**: Demonstrates PINN's power for data-scarce scenarios

### PINN vs FEM Comparison

| Method | Max Error | Mean Error | Parameters |
|--------|-----------|------------|------------|
| PINN   | 0.026427  | 0.006583   | 1,341      |
| FEM    | 0.017638  | 0.003102   | 100        |

## Architecture

### Neural Network Architecture
- **Temperature Network**: 4 hidden layers, 20 neurons per layer, tanh activation
- **Diffusion Coefficient Network**: 2 hidden layers, 10 neurons per layer, softplus output

### Loss Function Components
The total loss combines multiple physics-informed terms:

```
L_total = λ_IC × L_IC + λ_BC × L_BC + λ_PDE × L_PDE + λ_data × L_data
```

Where:
- `L_IC`: Initial condition loss
- `L_BC`: Boundary condition loss  
- `L_PDE`: PDE residual loss
- `L_data`: Data fitting loss (inverse problem)

## Project Structure

```
├── forward_problem.py          # Forward heat equation solver
├── inverse_problem.py          # Inverse problem implementation
├── fem_comparison.py          # FEM vs PINN comparison
├── sampled_data.npz           # Generated training data
├── inverse_results.npz        # Inverse problem results
├── report/                    # Detailed technical report
│   └── pinn_heat_equation_report.pdf
└── README.md
```

## Scientific Background

### Physics-Informed Neural Networks

PINNs leverage automatic differentiation to enforce physical laws during training. The key innovation is incorporating PDE residuals directly into the loss function, ensuring solutions satisfy governing equations.

### Advantages of PINNs

1. **Mesh-free approach**: No discretization required
2. **Inverse problem capability**: Direct parameter recovery
3. **Sparse data handling**: Effective with limited measurements
4. **Continuous solutions**: Differentiable across entire domain
5. **Complex geometries**: Easy adaptation to irregular domains

### Applications

- Parameter estimation in engineering systems
- Material property identification
- Uncertainty quantification
- Multi-physics problems
- Real-time monitoring and control

## Performance Considerations

### Computational Efficiency
- Forward problems: FEM is faster and more accurate for simple cases
- Inverse problems: PINNs provide unique advantages
- Training time: GPU acceleration recommended for larger problems

### Hyperparameter Sensitivity
- Loss function weighting crucial for convergence
- Network architecture affects solution quality
- Learning rate scheduling improves stability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please open an issue or contact [your-email@domain.com].

---

If you find this project useful, please consider giving it a star!
