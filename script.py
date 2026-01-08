import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. Problem Definition (Exact Solution and Source Term) ---

def exact_solution(x, y):
    """The analytical solution u(x,y) = sin(pi*x)*sin(pi*y)"""
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

def source_term(x, y):
    """The source term f(x,y) = -2*pi^2 * u(x,y) for the Poisson equation."""
    return -2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

# --- 2. Data Generation (Collocation, Boundary, and Sparse Data) ---

# 1. Collocation Points (Physics Loss): Random points inside the domain [0, 1]x[0, 1]
num_collocation = 2000
x_col = torch.rand(num_collocation, 1, requires_grad=True).to(device)
y_col = torch.rand(num_collocation, 1, requires_grad=True).to(device)

# 2. Boundary Points (Boundary Condition Loss): u=0 on the boundaries
num_bc = 500
x_bc = torch.rand(num_bc, 1).to(device)
y_bc = torch.rand(num_bc, 1).to(device)

# Enforce boundaries (x=0, x=1, y=0, y=1)
mask_side = torch.randint(0, 4, (num_bc, 1))
x_bc[mask_side == 0] = 0.0 # Left boundary
x_bc[mask_side == 1] = 1.0 # Right boundary
y_bc[mask_side == 2] = 0.0 # Bottom boundary
y_bc[mask_side == 3] = 1.0 # Top boundary

u_bc = torch.zeros_like(x_bc).to(device) # Dirichlet BC: u = 0

# 3. Sparse Data Points (Sparse Measurement Loss): Only 100 points known
num_data = 100 
x_data = torch.rand(num_data, 1).to(device)
y_data = torch.rand(num_data, 1).to(device)
# We use the exact solution to simulate the measurements
u_data = exact_solution(x_data, y_data).to(device) 

# --- 3. PINN Architecture (MLP with Tanh) ---

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y):
        # Concatenate x and y for the network input
        inputs = torch.cat([x, y], dim=1)
        return self.net(inputs)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. Training Loop and Loss Function ---

epochs = 5000
loss_history = []

print("Starting training...")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # --- 1. Loss PDE (Physics) ---
    u_pred_col = model(x_col, y_col)
    
    # Calculate first derivatives (du/dx, du/dy) using autograd
    du_dx = torch.autograd.grad(u_pred_col, x_col, torch.ones_like(u_pred_col), create_graph=True)[0]
    du_dy = torch.autograd.grad(u_pred_col, y_col, torch.ones_like(u_pred_col), create_graph=True)[0]
    
    # Calculate second derivatives (d2u/dx2, d2u/dy2)
    d2u_dx2 = torch.autograd.grad(du_dx, x_col, torch.ones_like(du_dx), create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y_col, torch.ones_like(du_dy), create_graph=True)[0]
    
    # Poisson residual: Laplacian(u) - f = 0
    laplacian = d2u_dx2 + d2u_dy2
    f_val = source_term(x_col, y_col)
    loss_pde = torch.mean((laplacian - f_val) ** 2)
    
    # --- 2. Loss BC (Boundary Conditions) ---
    u_pred_bc = model(x_bc, y_bc)
    loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)
    
    # --- 3. Loss Data (Sparse Reconstruction) ---
    u_pred_data = model(x_data, y_data)
    loss_data = torch.mean((u_pred_data - u_data) ** 2)
    
    # --- Total Loss ---
    loss = loss_pde + loss_bc + 2.0 * loss_data 
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.5f} (PDE: {loss_pde:.5f}, Data: {loss_data:.5f})")

print("Training finished.")

# --- 5. Results and Visualization (Super-Resolution Proof) ---

# Grid for visualization (64x64 to simulate super-resolution grid)
grid_size = 64
x_grid = np.linspace(0, 1, grid_size)
y_grid = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)

# Convert to tensors for prediction
x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
y_test = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)

# Prediction
model.eval()
with torch.no_grad():
    u_pred = model(x_test, y_test).cpu().numpy().reshape(grid_size, grid_size)
    u_true = exact_solution(x_test, y_test).cpu().numpy().reshape(grid_size, grid_size)

# Calculate Error
error = np.abs(u_true - u_pred)
mae = np.mean(error)
print(f"Mean Absolute Error (MAE) on the full field: {mae:.5f}")

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Exact Solution
sns.heatmap(u_true, cmap="viridis", ax=axes[0], cbar=True, square=True)
axes[0].set_title("Ground Truth (Exact Solution)")
axes[0].axis("off")

# 2. PINN Prediction + Sparse Data Points
sns.heatmap(u_pred, cmap="viridis", ax=axes[1], cbar=True, square=True)
# Overlay the sparse data points used for training
axes[1].scatter(x_data.cpu().numpy()* (grid_size - 1), y_data.cpu().numpy() * (grid_size - 1), c='red', s=10, label='Sparse Data')
axes[1].set_title(f"PINN Reconstruction\n(From {num_data} points)")
axes[1].legend(loc='upper right')
axes[1].axis("off")

# 3. Absolute Error
sns.heatmap(error, cmap="magma", ax=axes[2], cbar=True, square=True, vmin=0, vmax=np.max(error))
axes[2].set_title(f"Absolute Error (MAE: {mae:.4f})")
axes[2].axis("off")

plt.tight_layout()
plt.show()
