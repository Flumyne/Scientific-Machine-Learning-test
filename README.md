## 🔬 Physics-Informed Neural Network (PINN) for 2D Poisson Equation

### 🎯 Project Goal: High-Fidelity Field Reconstruction from Sparse Data

This project demonstrates the application of a Physics-Informed Neural Network (PINN) to solve a classic forward and inverse problem: the 2D Poisson equation with a known source term. The core objective is to showcase the network's ability to achieve super-resolution by accurately reconstructing the full solution field from an extremely sparse set of measured data points.

---

### 🧠 Technical Summary 

Developed a PINN (Physics-Informed Neural Network) in PyTorch to solve the 2D Poisson equation and reconstruct full fields from sparse measurements. Implemented automatic differentiation to calculate the PDE residual, ensuring physical consistency alongside data fitting. Successfully enforced Dirichlet boundary conditions. Demonstrated super-resolution capabilities by reconstructing a $64 \times 64$ field from $<5\%$ sampled points (100 measurements in the domain $[0, 1]^2$), achieving an average reconstruction error of $<4\%$ against the ground truth.

---

### 🛠️ Methodology and Implementation

#### 1. Governing Equation & Domain
The model solves the 2D Poisson equation on the unit square $\Omega = [0, 1] \times [0, 1]$:
$$\nabla^2 u(x,y) = f(x,y)$$
where the exact solution is $u(x,y) = \sin(\pi x)\sin(\pi y)$, leading to the source term $f(x,y) = -2\pi^2 u(x,y)$. **Dirichlet boundary conditions** are enforced ($u=0$ on $\partial\Omega$).

#### 2. PINN Architecture
* **Model:** A standard Multi-Layer Perceptron (MLP).
* **Structure:** Input (2) $\to$ [50] $\to$ [50] $\to$ [50] $\to$ Output (1).
* **Activation:** **Tanh** (selected for its smooth, infinitely differentiable property, critical for higher-order derivative calculations).

#### 3. Loss Function Formulation
The network is trained by minimizing a composite loss function, balancing adherence to physics, boundary constraints, and data fidelity:

$$Loss = Loss_{PDE} + Loss_{BC} + \lambda_{data} Loss_{data}$$

| Component | Purpose | Calculation Method |
| :--- | :--- | :--- |
| **$Loss_{PDE}$** | Enforces the physical law (Poisson Equation) at **collocation points**. | **Autograd** (Automatic Differentiation) used to compute the second-order derivatives ($\nabla^2 u$) of the network output with respect to the input coordinates. |
| **$Loss_{BC}$** | Enforces the boundary condition ($u=0$) on the domain edges. | Mean Squared Error (MSE) between $u_{pred}$ and $u_{BC}$ at boundary points. |
| **$Loss_{data}$** | Enforces fidelity to the available **sparse measurements**. | MSE between $u_{pred}$ and the 100 known data points. ($\lambda_{data}=2.0$ used for weighting). |

---

### 📊 Results

The PINN successfully learned the underlying analytical solution by minimizing the physics residual and fitting the sparse data simultaneously.

| Metric | Value |
| :--- | :--- |
| **Number of Sparse Samples** | 100 points ($\approx 0.25\%$ of the $64 \times 64$ grid points) |
| **Final Loss** | $\sim 10^{-4}$ |
| **Mean Absolute Error (MAE)** | $<0.04$ |
| **Visual Fidelity** | High (see Heatmaps below) |

#### Field Reconstruction 

The heatmap comparison clearly illustrates the network's ability to **generalize** and **interpolate** across the unobserved regions, yielding a near-perfect reconstruction of the full field.
