#%% Import librarys
import numpy as np
import pandas as pd
import time as time
import sys 
import os
import json
import do_mpc
from multiprocessing import Process, Queue
import pickle
from casadi import *
from casadi.tools import *

# Import CSTR files
# example_path = os.path.join('..','..',01_Example_Systems','CSTR')
example_path = os.path.join('01_Example_Systems','CSTR')
sys.path.append(example_path)
import cstr_model
import cstr_controller
import cstr_simulator
                 
#%% Setup server and dummy clients to get namespaces automatically

# Initiate do-mpc classes
model = cstr_model.get_model()
bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))
mpc, mpc_tvp_template  = cstr_controller.get_mpc(model, bound_dict)
simulator = cstr_simulator.get_simulator(model)

# Server options
server_opts = do_mpc.opcua.ServerOpts("CSTR-OPCUA","opc.tcp://localhost:4840/freeopcua/server/",  4840)                      
client_opts_mpc = do_mpc.opcua.ClientOpts("CSTR-OPCUA Client_02","opc.tcp://localhost:4840/freeopcua/server/",4840)
client_opts_sim = do_mpc.opcua.ClientOpts("CSTR-OPCUA Client_01","opc.tcp://localhost:4840/freeopcua/server/",4840)
Server = do_mpc.opcua.RTServer(server_opts)
rt_mpc = do_mpc.opcua.RTBase(mpc, client_opts_mpc)
rt_sim = do_mpc.opcua.RTBase(simulator, client_opts_sim)

# Setup server namespaces
Server.namespace_from_client(rt_mpc)
Server.namespace_from_client(rt_sim)

# Finally get the namespace tags
mpc_write_tags = rt_mpc.namespace['u'].copy()
sim_write_tags  = rt_sim.namespace['x'].copy()
mpc_read_tags = rt_sim.namespace['x'].copy()
sim_read_tags = rt_mpc.namespace['u'].copy()

#%% Define functions for simulator and controller clients

def send_mpc_to_process(read_tags, write_tags, runtime):
    # Setup do-mpc model
    model_p = cstr_model.get_model()
    # Setup mpc controller
    controller_p, _ = cstr_controller.get_mpc(model_p, bound_dict)
    # Define initial state
    controller_p.x0 = np.array([0.2, 0.5, 120.0, 120.0]).reshape(-1,1)
    # Set initial guess
    controller_p.set_initial_guess()

    # Define client optiones
    client_opts = do_mpc.opcua.ClientOpts("Bio Reactor OPCUA Client_MPC","opc.tcp://localhost:4840/freeopcua/server/",4840)
    # Initialize OPC UA client wrapper
    rt_mpc_p = do_mpc.opcua.RTBase(controller_p, client_opts)
    # Set tags to read and write
    rt_mpc_p.set_write_tags(write_tags)
    rt_mpc_p.set_read_tags(read_tags)
    # Connect client to server
    rt_mpc_p.connect()
    # Start mpc operation
    rt_mpc_p.async_step_start()
    time.sleep(runtime)
    # Stop mpc operation
    rt_mpc_p.async_step_stop()
    # Disconnect client from server
    rt_mpc_p.disconnect()


def send_sim_to_process(q, read_tags, write_tags, runtime):
    # Setup do-mpc model
    model_p = cstr_model.get_model()
    # Initiate simulator class
    simulator_p = cstr_simulator.get_simulator(model_p, t_step=0.000556)
    # Set initial guess
    simulator_p.x0 = np.array([0.2, 0.5, 120.0, 120.0]).reshape(-1,1)

    # Define Client options
    client_opts = do_mpc.opcua.ClientOpts("Bio Reactor OPCUA Client_SIM","opc.tcp://localhost:4840/freeopcua/server/",4840)
    # Initialize OPC UA client wrapper
    rt_sim_p = do_mpc.opcua.RTBase(simulator_p, client_opts)
    # Set tags to read and write
    rt_sim_p.set_write_tags(write_tags)
    rt_sim_p.set_read_tags(read_tags)
    # Connect to server
    rt_sim_p.connect()
    # Start simulator operation
    rt_sim_p.write_to_tags(simulator_p.x0)
    rt_sim_p.async_step_start()
    time.sleep(runtime)
    # Stop simulator operation
    rt_sim_p.async_step_stop()
    # Get simulator.data to main process
    q.put(rt_sim_p.do_mpc_object.data)
    # Disconnect from server
    rt_sim_p.disconnect()


# Observer client for main process
obs_client_opts = do_mpc.opcua.ClientOpts("OPCUA Observer","opc.tcp://localhost:4840/freeopcua/server/",4840)
obs_client = do_mpc.opcua.RTClient(obs_client_opts, [])





#%% Run all processes

if __name__ == '__main__':

    runtime = 520 # Runtime of closed loop operation
    q = Queue()

    # Start server and observer client
    Server.start()
    obs_client.connect()

    # Start simulator and mpc on seperate processes
    sim_process = Process(target=send_sim_to_process, args=(q, sim_read_tags, sim_write_tags, runtime))
    mpc_process = Process(target=send_mpc_to_process, args=(mpc_read_tags, mpc_write_tags, runtime))
    sim_process.start()
    mpc_process.start()

    # A loop to observe the closed loop operation
    for i in range(runtime + 20):
        mpc_test = np.array([obs_client.readData(mpc_write_tags[0]), obs_client.readData(mpc_write_tags[1])]) 
        sim_test = np.array([obs_client.readData(sim_write_tags[0]), obs_client.readData(sim_write_tags[1]), obs_client.readData(sim_write_tags[2]), obs_client.readData(sim_write_tags[3])])
        print(f'controller write (u): {mpc_test} simulator write (X0-X3): {sim_test}')
        time.sleep(1)
    
    # Get data from child process
    res = q.get()
    
    # Save data as pickle
    with open('cstr_res_closed_loop_real_time.pkl','wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Close child processes and Server/Client operation
    sim_process.join(timeout=20)
    mpc_process.join(timeout=20)
    obs_client.disconnect()
    Server.stop()


