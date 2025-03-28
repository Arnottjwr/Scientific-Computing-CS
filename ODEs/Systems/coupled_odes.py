
# import libraries
from torch import nn as nn
import torch
import torch.autograd as ag
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

seed = 140
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def analytical(x, y0):
    # y0 = np.array([1,0])
    # y  = odeint(DE, y0, x)
    # return y[:,0]
    return odeint(system, y0, x)



class ODENet(nn.Module):
    def __init__(self, input_size: int = 1, output_size: int = 2, hidden_layers: int = 5, hidden_nodes: int = 50) -> None:
        """
            Builds an neural network to approximate the value of the differential equation
            input_size must be set to number of parameters in the function f for ODE this is 1
            output_size number of values output by the function for ODE this is 1
        """
        super().__init__()
        
        self.inputs  = nn.Linear(input_size, hidden_nodes)
        self.model   = nn.ModuleList([nn.Linear(hidden_nodes, hidden_nodes)] * hidden_layers)
        self.outputs = nn.Linear(hidden_nodes, output_size) 
        
        self.activation = nn.Sigmoid() # Can be either tanh or sigmoid, ReLU is not differentiable everywhere, I think
    
    def _apply_model(self, x: torch.Tensor) -> torch.Tensor:
        """
            Runs the input through the hidden layers
        """
        for layer in self.model:
            x = self.activation(layer(x))
        return x

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self.activation(self.inputs(t))
        t = self._apply_model(t)
        output = self.outputs(t)
        y1, y2 = output[:, 0], output[:, 1]  # Ensure ordering is correct
        return [y1, y2]  # Ensures correct shape
    

def analytical_sol(x, x_bc, y1_bc, y2_bc):
    y1 = 1 - x*(1-x)
    # y1 = x*(1-x) + 1
    y2 = x*x*(1-x) + x
    # y2 = x*x*(1-x)

    return y1,y2

# x = x.reshape(-1,1)
# y = y.reshape(-1,1)

# x_data = x[0:200:20]
# y_data = y[0:200:20]
x_bc = np.array([[0], [1]])
y1_bc= np.array([1, 1])
y2_bc= np.array([0, 1])

x = np.linspace(0,1,100)
# Initial conditions
y1_0 = [1, 1]
y2_0 = [0, 1]
y1_analytical, y2_analytical = analytical_sol(x, x_bc, y1_bc, y2_bc)
# Extract solutions
# y1_analytical = y[:, 0]
# y2_analytical = y[:, 1]

# print(x_bc.shape, y_bc.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_data_t = torch.tensor(x_bc, dtype =torch.float32)
y1_data_t = torch.tensor(y1_bc, dtype =torch.float32)
y2_data_t = torch.tensor(y2_bc, dtype =torch.float32)

x_physics = torch.linspace(0, 1, 100, requires_grad=True,  dtype=torch.float32).reshape(-1,1)

# Hyperparameters
num_epochs = 20_000
LR = 1e-4
gamma=8
l1_weighting=1

model = ODENet(hidden_layers=2, hidden_nodes=30).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=.01)

store_loss=[]
store_loss_y1=[]
store_loss_y2=[]
for epoch in range(num_epochs):
    optim.zero_grad()
    
    # Compute the BC loss
    y_pred = model(x_data_t)
    y1, y2 = y_pred[0], y_pred[1]
    y1_bc_loss = y1_data_t - y1
    y2_bc_loss = y2_data_t - y2
    loss_bc_y1 = torch.sum( y1_bc_loss**2) 
    loss_bc_y2 = torch.sum( y2_bc_loss**2 )
    
    # Compute the Derivates of the model WRT inputs
    y_pred   = model(x_physics)
    y1, y2 = y_pred[0], y_pred[1]
    dy1  = ag.grad(y1, x_physics, torch.ones_like(y1), create_graph=True)[0]
    ddy1  = ag.grad(dy1, x_physics, torch.ones_like(dy1), create_graph=True)[0]
    dy2  = ag.grad(y2, x_physics, torch.ones_like(y2), create_graph=True)[0]
    ddy2  = ag.grad(dy2, x_physics, torch.ones_like(dy2), create_graph=True)[0]

    # Compute the internal loss
    y1_loss = torch.mean((ddy1 - x_physics * y1 - 2 +2*x_physics - y2)**2) + gamma * loss_bc_y1
    y2_loss = torch.mean((ddy2 - y1 -1 + (5+x_physics)*x_physics)**2) + gamma * loss_bc_y2
    loss_y = l1_weighting*y1_loss +y2_loss
    
    # backpropagate joint loss
    loss = y1_loss + y2_loss# add two loss terms together
    store_loss.append(loss.item())
    store_loss_y1.append(y1_loss.item())
    store_loss_y2.append(y2_loss.item())

    loss.backward()
    optim.step()
    # scheduler.step()
    
    if epoch % 1000 == 0:
        print(f"Epochs = {epoch} of {num_epochs}, Loss = {float(loss):.7f}")
    
model.eval()
y_pred = model(torch.tensor(x, dtype = torch.float32, device=device).reshape(-1,1))
y1_pred = y_pred[0].detach().cpu().numpy().reshape(-1,)
y2_pred = y_pred[1].detach().cpu().numpy().reshape(-1,)

title_font = {
    "fontsize" : 18,
    "fontweight": "bold"
} 



fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
# ax1.xticks(np.arange(0, 5.5, step=0.5))
ax1.set_title("Coupled ODEs")
ax1.plot(x, y1_analytical, label="Analytical Solution y1", color = 'lightblue')
ax1.plot(x, y2_analytical, label="Analytical Solution y2", color = 'thistle')
ax1.set_xlabel("x")
ax1.plot(x, y1_pred, label="NN Approximation y1", color = 'navy')
ax1.plot(x, y2_pred, label="NN Approximation y2", color = 'purple')
ax1.set_ylabel("y")
ax1.scatter(x_bc.reshape(-1,), y1_bc.reshape(-1,) , label="Boundary Conditions", color='pink')
ax1.scatter(x_bc.reshape(-1,), y2_bc.reshape(-1,) , color='pink')
ax1.legend()


fig.savefig("coupled_odes.png")
