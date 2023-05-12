#%% [markdown]
## OPC UA operated MPC testing vs. offline synchronized MPC testing
# Import necessary librarys and change some mpl values:
#%%
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'lines.linewidth':2,
    'font.size': 14,
    'axes.labelsize': 'large',
    'figure.figsize': (15, 7),
    'axes.grid': True,
    'lines.markersize': 10,
    'axes.unicode_minus': False,
    'ps.fonttype': 42,            
    'pdf.fonttype': 42,           
})
color = mpl.rcParams['axes.prop_cycle'].by_key()['color']

#%% [markdown]
# Load the results and a dict containing the constraints:
#%% 
# Paths to constraints-dict and simulator.data
# example_path = os.path.join('01_Example_Systems','CSTR')
# data_path = os.path.join('03_LQR2NMPC','CSTR_w_OPCUA','results')
example_path = os.path.join('..','..','01_Example_Systems','CSTR')
data_path = os.path.join('..','..','03_LQR2NMPC','CSTR_w_OPCUA','results')
sys.path.append(example_path)

# Load the data
data_opc = pd.read_pickle(data_path +'/cstr_res_closed_loop_real_time.pkl')
data = pd.read_pickle(data_path +'/cstr_res_closed_loop_ideal.pkl')

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
ax[0].plot(data_opc['_aux'][:len(data_opc['_x']),-1], label='$cost_{opc}$')
ax[0].plot(data['_aux'][:len(data_opc['_x']),-1], label='cost')
ax[0].legend()
ax[0].set_ylabel('Cost')

ax[1].plot(data_opc['_x'][:,0],label='$c_{a,opc}$')
ax[1].plot(data_opc['_x'][:,1],label='$c_{b,opc}$')
ax[1].plot(data['_x'][:len(data_opc['_x']),0],':',label='$c_{a}$')
ax[1].plot(data['_x'][:len(data_opc['_x']),1],':',label='$c_{b}$')
ax[1].legend()
ax[1].set_ylabel('Concentration [mol/l]')
ax[1].set_xlabel('Time')
plt.show(block=True)
#%%