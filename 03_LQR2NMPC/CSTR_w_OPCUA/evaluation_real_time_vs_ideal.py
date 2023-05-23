#%% [markdown]
## OPC UA operated MPC testing vs. offline synchronized MPC testing

#%%
import numpy as np
import os
import sys
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

plot_path = os.path.join('..','..','00_plotting')
sys.path.append(plot_path)
import mplconfig
mplconfig.config_mpl(os.path.join('..','..','00_plotting','notation.tex'))

#%% [markdown]
# Load the results and a dict containing the constraints:
#%% 
# Paths to constraints-dict and simulator.data
example_path = os.path.join('..','..','01_Example_Systems','CSTR')
data_path = os.path.join('..','..','03_LQR2NMPC','CSTR_w_OPCUA','results')
sys.path.append(example_path)

# Load the data
with open(os.path.join(data_path, 'cstr_res_closed_loop_real_time.pkl'), 'rb') as f:
    data_opc = pickle.load(f)

with open(os.path.join(data_path, 'cstr_res_closed_loop_ideal.pkl'), 'rb') as f:
    data = pickle.load(f)

# Load dict containing the constraints information
bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))

#%% [markdown]
# Compute closed loop cost and check for constraint violations:
#%%

# A function to check if constraints are violated
def cons_viol(res, bound_dict):

    x_lb = np.array([val for val in bound_dict['states']['lower'].values()])
    x_ub = np.array([val for val in bound_dict['states']['upper'].values()])

    lb_viol = np.maximum(x_lb - res['_x'], 0)
    ub_viol = np.maximum(res['_x'] - x_ub, 0)

    cons_viol = lb_viol + ub_viol

    return cons_viol

# Compute closed loop cost for the operation time
cost_opc = sum(data_opc['_aux','closed_loop_cost'][:len(data_opc['_x'])])
cost_wo_opc = sum(data['_aux','closed_loop_cost'][:len(data_opc['_x'])])

print(f'Closed-loop cost real time simulation with OPC UA: {float(cost_opc)}, Closed-loop cost without time delay: {float(cost_wo_opc)}')
print(f'OPCUA leads to an increase of {float(100 - cost_wo_opc/cost_opc*100)}%')
print(f'Constraint-viol. X1-X4:{max(cons_viol(data_opc,bound_dict)[:,0]),max(cons_viol(data_opc,bound_dict)[:,1]),max(cons_viol(data_opc,bound_dict)[:,2]**2),max(cons_viol(data_opc,bound_dict)[:,3]**2)}')


#%% [markdown]
### Visualization
# Visualize the closed loop costs along with the concentrations

#%%

fig, ax = plt.subplots(2,1, sharex=True)

t_opc = data_opc['_time']
t = data['_time'][:len(data_opc['_x'])]
ax[0].plot(t_opc, data_opc['_aux','closed_loop_cost'][:len(data_opc['_x'])], label='cost$_{opcua}$')
ax[0].plot(t, data['_aux', 'closed_loop_cost'][:len(data_opc['_x'])], label='cost')
ax[0].legend()
ax[0].set_ylabel('Cost')

ax[1].plot(t_opc, data_opc['_x'][:,0],label='c$_{a,opcua}$')
ax[1].plot(t_opc, data_opc['_x'][:,1],label='c$_{b,opcua}$')
ax[1].plot(t, data['_x'][:len(data_opc['_x']),0],':',label='c$_{a}$')
ax[1].plot(t, data['_x'][:len(data_opc['_x']),1],':',label='c$_{b}$')
ax[1].legend()
ax[1].set_ylabel('Concentration [mol/l]')
ax[1].set_xlabel('Time')
plt.show(block=True)
#%%