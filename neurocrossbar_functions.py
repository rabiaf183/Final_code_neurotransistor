# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:01:14 2025

@author: risc915d
"""

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np


from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def create_neurocrossbar_circuit(N, M, V_left, V_right, V_top, 
                                 R_load_left, R_load_right, R_load_top, R_wire_wl, R_wire_bl, 
                                 memristor, x0, C_cross, C_wire, C_gb,
                                 NMOS_Level, NMOS_L, NMOS_W, NMOS_Vto, NMOS_Tox,
                                 Rt, R_out_left, R_out_right, V_out_left, V_out_right):
    """
    Create a neurocrossbar circuit with specified dimensions, parameters, and NMOS transistors.

    Parameters:
    - N: Number of rows (wordlines)
    - M: Number of columns (bitlines)
    - V_left, V_right, V_top: Voltage sources for each side of the crossbar (lists or single values)
    - R_load_left, R_load_right, R_load_top : Load resistors for each side of the crossbar (lists or single values)
    - R_wire_wl: Wire resistance for word lines in ohms
    - R_wire_bl: Wire resistance for bit lines in ohms
    - memristor: Name of the memristor subcircuit to include
    - x0: Initial conditions for memristor state variables (NxM numpy array)
    - C_cross: Capacitance at the crosspoints in farads (optional)
    - C_wire: Wire capacitance for lines in farads (optional)
    - NMOS_L, NMOS_W, NMOS_Vto, NMOS_Tox: NMOS transistor model parameters
    - C_gb: Capacitance between gate and bulk
    - Rt: Resistance between NMOS transistors in ohms
    - R_out_left: Output resistance on the left side in ohms
    - R_out_right: Output resistance on the right side in ohms
    - V_out_left: Output voltage on the left side in volts
    - V_out_right: Output voltage on the right side in volts

    Returns:
    - circuit: The created Circuit object
    - initial_conditions: Dictionary holding initial conditions for memristor state variables

    Notes:
    - The circuit includes a crossbar array with memristors at each junction.
    - NMOS transistors are added at the bottom of the crossbar, connected in series.
    - The leftmost transistor's source is connected to ground via R_out_left and V_out_left.
    - The rightmost transistor's drain is connected to V_out_right via R_out_right.
    - All transistor bulks are connected to ground via a 1 Ohm resistor.
    """
    
    initial_conditions = {}
    circuit = Circuit('Neurocrossbar Network with NMOS Transistors')

    ### Crossbar
    # Add word lines (horizontal wires)
    for i in range(N):
        for j in range(M - 1):  
            circuit.R(f'R_wire_row_{i}_{j}', f'row_{i}_{j}', f'row_{i}_{j + 1}', R_wire_wl)
            if C_wire is not None:
                circuit.C(f'C_wire_row_{i}_{j}', f'row_{i}_{j}', f'row_{i}_{j + 1}', C_wire)

    # Add bit lines (vertical wires)
    for j in range(M):
        for i in range(N - 1):
            circuit.R(f'R_wire_col_{i}_{j}', f'col_{i}_{j}', f'col_{i + 1}_{j}', R_wire_bl)
            if C_wire is not None:
                circuit.C(f'C_wire_col_{i}_{j}', f'col_{i}_{j}', f'col_{i + 1}_{j}', C_wire)

    # Add crosspoint resistors and capacitors
    for i in range(N):
        for j in range(M):
            circuit.X(f'M_{i}_{j}', memristor, f'row_{i}_{j}', f'col_{i}_{j}', f'sv_{i}_{j}')
            initial_conditions[f'sv_{i}_{j}'] = x0[i, j]
            if C_cross is not None:
                circuit.C(f'C_cross_{i}_{j}', f'row_{i}_{j}', f'col_{i}_{j}', C_cross)

    # Left side (first column)
    for i in range(N):
        circuit.R(f'R_load_left_{i}', f'row_{i}_0', f'left_{i}', R_load_left[i])
        circuit.V(f'V_left_{i}', f'left_{i}', circuit.gnd, V_left[i])
    
    # Right side (last column)
    for i in range(N):
        circuit.R(f'R_load_right_{i}', f'row_{i}_{M-1}', f'right_{i}', R_load_right[i])
        circuit.V(f'V_right_{i}', f'right_{i}', circuit.gnd, V_right[i])

    # Top side (first row)
    for j in range(M):
        circuit.R(f'R_load_top_{j}', f'col_0_{j}', f'top_{j}', R_load_top[j])
        circuit.V(f'V_top_{j}', f'top_{j}', circuit.gnd, V_top[j])
    
    ### Add NMOS transistors to the bottom
    
    # NMOS transistor model, see definition here: \\ge-mobil-13\C$\EigeneProgramme\Anaconda3\envs\pyspice-env\Lib\site-packages\PySpice\Spice\Netlist.py
    #C_gbo = C_gb/NMOS_W
    #circuit.model('IHM_NMOS', 'nmos', LEVEL=NMOS_Level, L=NMOS_L@u_m, W=NMOS_W@u_m, Vto=NMOS_Vto@u_V, Tox=NMOS_Tox@u_m, CGBO=C_gbo@u_F)
    #print(NMOS_Tox)
    circuit.model('IHM_NMOS', 'nmos', LEVEL=NMOS_Level, L=NMOS_L, W=NMOS_W, Vto=NMOS_Vto, Tox=NMOS_Tox)
        
    for j in range(M):
        # MXXXXXXX drain gate source bulk modelname
        circuit.MOSFET(f'M_bottom_{j}', f'drain_{j}', f'col_{N-1}_{j}', f'source_{j}', 'bulk', model='IHM_NMOS')
        # gate bulk capacitance
        circuit.C(f'C_gb_{j}', f'col_{N-1}_{j}', 'bulk', C_gb)
        
        # parasitic resistances
        circuit.R(f'R_gb_{j}', f'col_{N-1}_{j}', 'bulk', 10e6)
        circuit.R(f'R_db_{j}', f'drain_{j}', 'bulk', 10e6)
        circuit.R(f'R_sb_{j}', f'source_{j}', 'bulk', 10e6)
        
        if j == 0:
            # Connect the leftmost transistor's source to ground via R_out and V_out
            circuit.R(f'R_out_left', 'out_left', f'source_{j}', R_out_left)
            circuit.V('V_out_left', 'out_left' , circuit.gnd, V_out_left)
        else:
            # Connect the drain to the previous transistor's source via Rt
            circuit.R(f'Rt_{j}', f'drain_{j-1}', f'source_{j}', Rt)
       
    # Connect the rightmost transistor's drain to the input
    circuit.R('R_out_right', 'out_right', f'drain_{M-1}', R_out_right)
    circuit.V('V_out_right', 'out_right', circuit.gnd, V_out_right)
    
    # Connect all transistor bulks to a common node via 1Ohm
    circuit.R('R_bulk', 'bulk', circuit.gnd, 0.1)

    return circuit, initial_conditions


def run_neurocrossbar_circuit(N, M, step_time, end_time, rel_tol, memristor_subcircuit,
                                  V_left,V_right,V_top,
                                  R_load_left,R_load_right,R_load_top,R_wire_wl,R_wire_bl,
                                  memristor, x0, C_cross=None, C_wire=None, C_gb=100e-12,
                                  NMOS_Level=1, NMOS_L=26e-6, NMOS_W=94e-6, NMOS_Vto=0.3, NMOS_Tox=22e-9,
                                  Rt=10, R_out_left=1, R_out_right=100, V_out_left=0, V_out_right=1,
                                  save_currents=False, save_nodes=False, save_states=True, save_edge_nodes=False,
                                  save_power=False, run_neurocrossbar=False):
    #import time
    #start_time = time.time()
    #print(C_cross)
    
    circuit,initial_conditions = create_neurocrossbar_circuit(N, M, V_left, V_right, V_top,
                                     R_load_left, R_load_right, R_load_top,  R_wire_wl, R_wire_bl, 
                                     memristor, x0, C_cross, C_wire, C_gb,
                                     NMOS_Level, NMOS_L, NMOS_W, NMOS_Vto, NMOS_Tox,
                                     Rt, R_out_left, R_out_right, V_out_left, V_out_right)
    
    circuit.raw_spice += memristor_subcircuit # include Namlab model    
    
    if save_edge_nodes:
        for i in range(N):  # N rows (wordlines)
            circuit.raw_spice += f'.save row_{i}_{M-1} \n' 
            circuit.raw_spice += f'.save row_{i}_0 \n' 
        for j in range(M): # M columns(bitlines)
            circuit.raw_spice += f'.save col_{N-1}_{j} \n' 
            circuit.raw_spice += f'.save col_0_{j} \n' 
                
    if save_nodes:
        for i in range(N):     # N rows (wordlines)
            for j in range(M): # M columns(bitlines)
                circuit.raw_spice += f'.save V(row_{i}_{j}) \n' 
                circuit.raw_spice += f'.save V(col_{i}_{j}) \n' 
                
    if save_currents:
        #simulator.save_currents = True  # Set to True to save all currents
        #for i in range(N):
        #    for j in range(M):
        #        circuit.raw_spice += f'.probe I(XM_{i}_{j},1) \n'
        
        for j in range(M):
            circuit.raw_spice += f'.probe I(VV_top_{j}) \n'
            #circuit.raw_spice += f'.save I(VV_bottom_{j}) \n'
            #circuit.raw_spice += f'.probe I(RR_load_top_{j}) \n'
    
    if run_neurocrossbar:
        for i in range(N): # word line inputs
            circuit.raw_spice += f'.save left_{i} \n'          
        for j in range(M): # gate voltages
            circuit.raw_spice += f'.save col_{N-1}_{j} \n'          
        for j in range(M):
            circuit.raw_spice += f'.probe I(RR_out_right) \n'
            #circuit.raw_spice += f'.probe I(RR_out_left) \n'
            
    # save map of last state values
    if save_states:
        for i in range(N):     # N rows (wordlines)
            for j in range(M): # M columns(bitlines)
                #circuit.raw_spice += f'.probe V(sv_{i}_{j}) \n'
                circuit.raw_spice += f'.save V(sv_{i}_{j}) \n' # save its better than probe
    
    #if save_power: # store the power
    #    circuit=add_power_analysis_saves(circuit, N, M)   
        #circuit.raw_spice += ".probe alli \n" 
        #circuit.raw_spice += ".save all \n" 
        #print("Save the power")
    
    # Simulate the circuit
    circuit.raw_spice += ".options plotwinsize=0 \n" # do not compress the data output
    circuit.raw_spice += f".options reltol={rel_tol} \n" # relative tolerance
    #circuit.raw_spice += f".options abstol=10p \n" # relative tolerance
    #circuit.raw_spice += f".options vntol=10u \n" # relative tolerance
    #circuit.raw_spice += f".options trtol=40 \n" # relative tolerance
    #circuit.raw_spice += f".options method=gear \n" # method gear
    
    
    
    #print(f"Execution time: {time.time()-start_time} seconds")    
    #print(circuit)
    
    simulator = circuit.simulator(temperature=25, nominal_parameters=[])
    simulator.initial_condition(**initial_conditions)
    
    #print(simulator)
    #print(f"Execution time: {time.time()-start_time} seconds")
        
    # Perform a transient simulation as an example
    #analysis = simulator.transient(step_time=step_time,end_time=end_time,max_time=None,use_initial_condition=False)
    analysis = simulator.transient(step_time=step_time,end_time=end_time)
    
    #print(analysis)
    #print(analysis.nodes)
    #print(analysis.branches)
    #print(f"Execution time: {time.time()-start_time} seconds")
    
    return analysis


   

def update_write_voltage(last_voltage, target_conductance, current_conductance, 
                            set_step_size,reset_step_size, set_voltage_range, reset_voltage_range):
    
    step_set   = (set_voltage_range[1] - set_voltage_range[0])/set_step_size
    step_reset = (reset_voltage_range[1] - reset_voltage_range[0])/reset_step_size
    #print(f"RESET step {step_reset} V, SET step {step_set} V")
    
    if target_conductance < current_conductance:
        # RESET
        sign_reset = np.sign(reset_voltage_range[0])
        # after switch from SET to RESET, start with range[0], else update step
        if np.sign(last_voltage) != sign_reset:   
            write_voltage = reset_voltage_range[0] # start reset voltage
            #print(f"write_voltage {write_voltage} V, last_voltage {last_voltage} V")
        else:
            write_voltage = last_voltage + step_reset # next step  
        # limit in the range
        write_voltage = np.clip(write_voltage,min(np.array(reset_voltage_range)),max(np.array(reset_voltage_range)))
        #print(f"RESET with {write_voltage} V")
    else:  
        # SET
        sign_set = np.sign(set_voltage_range[0])
        # after switch from RESET to SET, start with range[0], else update step
        if np.sign(last_voltage) != sign_set:   
            write_voltage = set_voltage_range[0] # minimum set voltage
            #print(f"write_voltage {write_voltage} V, last_voltage {last_voltage} V")
        else:
            write_voltage = last_voltage + step_set # next step 
        #print(f"SET with {write_voltage} V")
        # limit in the range
        write_voltage = np.clip(write_voltage,min(np.array(set_voltage_range)),max(np.array(set_voltage_range)))
                
    return write_voltage



def update_write_input(last_voltage, target_conductance, current_conductance, 
                            set_step_size,reset_step_size, set_voltage_range, reset_voltage_range,
                            write_times):
    
    step_set   = (set_voltage_range[1] - set_voltage_range[0])/set_step_size
    step_reset = (reset_voltage_range[1] - reset_voltage_range[0])/reset_step_size
    #print(f"RESET step {step_reset} V, SET step {step_set} V")
    
    if target_conductance < current_conductance:
        # RESET
        sign_reset = np.sign(reset_voltage_range[0])
        # after switch from SET to RESET, start with range[0], else update step
        if last_voltage is None or  np.sign(last_voltage) != sign_reset:   
            write_voltage = reset_voltage_range[0] # start reset voltage
            #print(f"write_voltage {write_voltage} V, last_voltage {last_voltage} V")
        else:
            write_voltage = last_voltage + step_reset # next step  
        # limit in the range
        write_voltage = np.clip(write_voltage,min(np.array(reset_voltage_range)),max(np.array(reset_voltage_range)))
        #print(f"RESET with {write_voltage} V")
        write_time = write_times[1]
    else:  
        # SET
        sign_set = np.sign(set_voltage_range[0])
        # after switch from RESET to SET, start with range[0], else update step
        if last_voltage is None or  np.sign(last_voltage) != sign_set:   
            write_voltage = set_voltage_range[0] # minimum set voltage
            #print(f"write_voltage {write_voltage} V, last_voltage {last_voltage} V")
        else:
            write_voltage = last_voltage + step_set # next step 
        #print(f"SET with {write_voltage} V")
        # limit in the range
        write_voltage = np.clip(write_voltage,min(np.array(set_voltage_range)),max(np.array(set_voltage_range)))
        write_time = write_times[0]
            
    return write_voltage, write_time


## read all memristors
def read_neurocrossbar(N, M, read_time_step, read_time, rel_tol, memristor_subcircuit, read_voltage, Rload, Ropen, Rsense,
                  R_wire_wl,R_wire_bl, memristor, x):

    G = np.zeros((N,M))
    for wordline in range(N):     # N rows (wordlines)
        for bitline in range(M):  # M columns(bitlines)
            
            # reading scheme
            V_left = np.zeros(N) 
            V_right = np.zeros(N) 
            V_top  = np.zeros(M) 
            R_load_left     = np.full(N, Rload)  
            R_load_right    = np.full(N, Ropen)  
            R_load_top      = np.full(M, Rsense)  
            
            V_left[wordline] = read_voltage  # Left side voltages
            
            analysis = run_neurocrossbar_circuit(N, M, read_time_step, read_time, rel_tol, memristor_subcircuit,
                                 V_left,V_right,V_top,
                                 R_load_left,R_load_right,R_load_top,
                                 R_wire_wl,R_wire_bl,memristor,x,save_currents=True)
            
            # save map of last state values
            for i in range(N):     # N rows (wordlines)
                for j in range(M): # M columns(bitlines)
                    x[i,j] = analysis[f'sv_{i}_{j}'][-1] # required to update the state after each measurement
            
            I_read = analysis[f'VV_top_{bitline}'][-1]
            #print(I_read)
            # compensate read error for known line resistance Rl
            R_wire_sum = R_wire_wl*(N-1-wordline) + R_wire_bl*bitline            
            G_read = float(I_read) / (read_voltage - float(I_read)*(R_wire_sum + Rload + Rsense))
            #print(f'initial G{wordline}{bitline}={G_read*1e6:.2f} uS ({1/G_read*1e-3:.2f} kOhm)')
            
            G[wordline, bitline] = G_read
            
    #print('read crossbar finished')
    return x,G
        



def update_write_voltage_G_error( target_conductance, current_conductance, 
                            set_step_size,reset_step_size, set_voltage_range, reset_voltage_range,normalized_error):
    
    
    if target_conductance < current_conductance:
        # RESET
        sign_reset = np.sign(reset_voltage_range[0])
        
        # Scale voltage within reset range based on normalized error
        write_voltage = reset_voltage_range[0] + normalized_error * (reset_voltage_range[1] - reset_voltage_range[0])
        #print(f"RESET: {write_voltage} = {reset_voltage_range[0]} + {normalized_error:.3f} * ({reset_voltage_range[1]} - {reset_voltage_range[0]})")
        # Ensure voltage stays within reset range
        write_voltage = np.clip(write_voltage, min(reset_voltage_range), max(reset_voltage_range))
        
    else:  
        # SET
        sign_set = np.sign(set_voltage_range[0])
        
        # Scale voltage within set range based on normalized error
        write_voltage = set_voltage_range[0] + normalized_error * (set_voltage_range[1] - set_voltage_range[0])
        #print(f"SET: {write_voltage} = {set_voltage_range[0]} + {normalized_error:.3f} * ({set_voltage_range[1]} - {set_voltage_range[0]})")
        
        # Ensure voltage stays within set range
        write_voltage = np.clip(write_voltage, min(set_voltage_range), max(set_voltage_range))
                
    return write_voltage


def generate_pulse_train(total_time, pulse_time, pulse_voltage, pulse_interval, pulse_width, pulse_slope):
    """
    Generate time and voltage arrays for a pulse train.

    Parameters:
    - total_time: Total simulation time
    - pulse_time: Duration of the pulse train
    - pulse_voltage: Voltage of each pulse
    - pulse_interval: Time between the start of each pulse
    - pulse_width: Width of each pulse
    - pulse_slope: Rise and fall time of each pulse

    Returns:
    - time: Array of time points
    - voltage: Array of voltage values corresponding to the time points
    """
    # Calculate number of pulses
    num_pulses = int(pulse_time / pulse_interval)

    # Define pulse shape
    pulse_points = np.array([0, pulse_slope, pulse_width - pulse_slope, pulse_width])
    pulse_values = np.array([0, pulse_voltage, pulse_voltage, 0])

    # Generate time and voltage arrays
    time = np.tile(pulse_points, num_pulses) + np.repeat(np.arange(num_pulses) * pulse_interval, 4)
    voltage = np.tile(pulse_values, num_pulses)

    return time, voltage
