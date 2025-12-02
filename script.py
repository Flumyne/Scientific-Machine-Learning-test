import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour la reproductibilité
torch.manual_seed(42)
np.random.seed(42)

# Utilisation du GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement sur : {device}")

#################################################################################################################

#################################################################################################################

def exact_solution(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

def source_term(x, y):
    # f(x,y) = -2 * pi^2 * sin(pi*x) * sin(pi*y)
    return -2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

# --- Génération des données ---

# 1. Points de Collocation (Physics Loss) : Points aléatoires dans le domaine pour vérifier l'EDP
num_collocation = 2000
x_col = torch.rand(num_collocation, 1, requires_grad=True).to(device)
y_col = torch.rand(num_collocation, 1, requires_grad=True).to(device)

# 2. Points de Bord (Boundary Condition Loss) : u=0 sur les bords
# On génère des points aléatoires sur les 4 côtés
num_bc = 500
x_bc = torch.rand(num_bc, 1).to(device)
y_bc = torch.rand(num_bc, 1).to(device)

# Force certains points à être 0 ou 1 pour simuler les bords
mask_side = torch.randint(0, 4, (num_bc, 1))
x_bc[mask_side == 0] = 0.0 # Bord gauche
x_bc[mask_side == 1] = 1.0 # Bord droit
y_bc[mask_side == 2] = 0.0 # Bord bas
y_bc[mask_side == 3] = 1.0 # Bord haut

u_bc = torch.zeros_like(x_bc).to(device) # Condition de Dirichlet u=0

# 3. Données Parcellaires (Sparse Data Loss) : C'est la "Super-Résolution"
# On prend seulement 100 points aléatoires (très peu !) pour reconstruire le champ
num_data = 100 
x_data = torch.rand(num_data, 1).to(device)
y_data = torch.rand(num_data, 1).to(device)
u_data = exact_solution(x_data, y_data).to(device) # On connait la vérité seulement ici

#############################################################################################

#############################################################################################
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Entrée : (x, y) -> 2 neurones
        # Sortie : u(x, y) -> 1 neurone
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
        # Concaténer x et y pour l'entrée du réseau
        inputs = torch.cat([x, y], dim=1)
        return self.net(inputs)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#############################################################################################

#############################################################################################
epochs = 5000
loss_history = []

print("Début de l'entraînement...")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # --- 1. Loss PDE (Physique) ---
    # Prédiction aux points de collocation
    u_pred_col = model(x_col, y_col)
    
    # Calcul des dérivées premières (du/dx, du/dy)
    grad_u = torch.autograd.grad(u_pred_col, [x_col, y_col], 
                                 grad_outputs=torch.ones_like(u_pred_col), 
                                 create_graph=True)[0] # [du/dx, du/dy] (mal géré par autograd combiné parfois, on sépare souvent)
    
    # Méthode plus explicite pour les dérivées :
    du_dx = torch.autograd.grad(u_pred_col, x_col, torch.ones_like(u_pred_col), create_graph=True)[0]
    du_dy = torch.autograd.grad(u_pred_col, y_col, torch.ones_like(u_pred_col), create_graph=True)[0]
    
    # Calcul des dérivées secondes (d2u/dx2, d2u/dy2)
    d2u_dx2 = torch.autograd.grad(du_dx, x_col, torch.ones_like(du_dx), create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y_col, torch.ones_like(du_dy), create_graph=True)[0]
    
    # Résidu de l'équation de Poisson : Laplacian(u) - f = 0
    laplacian = d2u_dx2 + d2u_dy2
    f_val = source_term(x_col, y_col)
    loss_pde = torch.mean((laplacian - f_val) ** 2)
    
    # --- 2. Loss BC (Conditions aux limites) ---
    u_pred_bc = model(x_bc, y_bc)
    loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)
    
    # --- 3. Loss Data (Reconstruction Sparse) ---
    u_pred_data = model(x_data, y_data)
    loss_data = torch.mean((u_pred_data - u_data) ** 2)
    
    # --- Loss Totale ---
    # On peut pondérer les termes (ex: donner plus de poids aux données)
    loss = loss_pde + loss_bc + 2.0 * loss_data
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.5f} (PDE: {loss_pde:.5f}, Data: {loss_data:.5f})")

print("Entraînement terminé.")

########################################################################################################################

########################################################################################################################
# Grille pour la visualisation (64x64 pour simuler la super-résolution)
x_grid = np.linspace(0, 1, 64)
y_grid = np.linspace(0, 1, 64)
X, Y = np.meshgrid(x_grid, y_grid)

# Conversion en tenseurs
x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
y_test = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)

# Prédiction
model.eval()
with torch.no_grad():
    u_pred = model(x_test, y_test).cpu().numpy().reshape(64, 64)
    u_true = exact_solution(x_test, y_test).cpu().numpy().reshape(64, 64)

# Calcul de l'erreur
error = np.abs(u_true - u_pred)
mae = np.mean(error)
print(f"Erreur Moyenne Absolue (MAE) sur tout le champ : {mae:.5f}")

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Solution Exacte
sns.heatmap(u_true, cmap="viridis", ax=axes[0], cbar=True)
axes[0].set_title("Ground Truth (Exact Solution)")
axes[0].axis("off")

# 2. Prédiction PINN + Points de données
sns.heatmap(u_pred, cmap="viridis", ax=axes[1], cbar=True)
# Superposition des points de données utilisés (les 100 points)
# Note: il faut mapper les coordonnées [0,1] vers les indices [0,63] pour l'affichage heatmap
axes[1].scatter(x_data.cpu().numpy()*63, y_data.cpu().numpy()*63, c='red', s=10, label='Sparse Data')
axes[1].set_title(f"PINN Reconstruction\n(From {num_data} points)")
axes[1].legend(loc='upper right')
axes[1].axis("off")

# 3. Erreur absolue
sns.heatmap(error, cmap="magma", ax=axes[2], cbar=True)
axes[2].set_title(f"Absolute Error (MAE: {mae:.4f})")
axes[2].axis("off")

plt.tight_layout()
plt.show()
