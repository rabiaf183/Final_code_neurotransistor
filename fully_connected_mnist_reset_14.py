"""Fully Connected Layer with Gate Reset: 3 Neurons → 1 Output
Based on the reset circuit code structure with proper MNIST integration.
"""

from PySpice.Spice.Netlist import Circuit
import numpy as np
import os
import matplotlib.pyplot as plt
from memcap_model import get_subcircuit
from MNIST import create_MNIST_pulse_train, load_mnist

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS (matching reset circuit code)
# ══════════════════════════════════════════════════════════════════════════════

N = 6  # Rows in crossbar (matches MNIST 2x3 = 6 pixels)
M = 3  # Columns in crossbar
num_neurons = 3

# Crossbar parameters
C_cross = 10e-12
C_gb = 100e-12
R_wire_wl = 0.33
R_wire_bl = 60
Rload = 1
Ropen = 1e9

# Output stage parameters
Rs = 10
Rout = 200e3
VDD = 1.0

# NMOS parameters (matching reset circuit)
NMOS_PARAMS = dict(
    LEVEL=14, L=26e-6, W=94e-6, VTH0=0.3,
    TOXE=26e-9, EPSROX=22, CGBO=69e-9, CGDO=56e-9, CGSO=56e-9
)
NMOS_VTH0 = 0.3

# PMOS parameters (matching reset circuit)
PMOS_PARAMS = dict(
    LEVEL=14, L=26e-6, W=94e-6, VTH0=-0.3, TOXE=22e-9, EPSROX=22
)

PMOS2_PARAMS = dict(
    LEVEL=14, L=26e-6, W=24e-6, VTH0=-0.3, TOXE=22e-9, EPSROX=22
)

# Reset transistor parameters (matching reset circuit)
RESET_PARAMS = dict(
    LEVEL=14, L=10e-6, W=10e-6, VTH0=0.6,
    TOXE=26e-9, EPSROX=22, CGBO=69e-9, CGDO=56e-9, CGSO=56e-9
)
RESET_VTH0 = 0.6

# Pulse parameters
pulse_voltage = 1.0
pulse_width = 20e-6
pulse_slope = 200e-9

# Weight parameters
xmin = 0.1
xmax = 0.284
seed = 42
np.random.seed(seed)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MNIST DATA
# ══════════════════════════════════════════════════════════════════════════════

imx, imy = load_mnist('raw', kind='train')
digits = [3, 3, 7]
digit_images = []
pulse_trains_list = []

for i, digit in enumerate(digits):
    pulse_trains, digit_image = create_MNIST_pulse_train(
        imx, imy, N, pulse_voltage, pulse_width, pulse_slope,
        selected_digits=[digit], do_plot=None, specific_image_index=i
    )
    pulse_trains_list.append(pulse_trains[digit])
    digit_images.append(digit_image)

times_list = [pt['times'] for pt in pulse_trains_list]
voltages_list = [pt['voltages'] for pt in pulse_trains_list]
total_time = times_list[0][0][-1]

# Weight matrices for input neurons (random initialization)
x0_neurons = [np.random.uniform(low=xmin, high=xmax, size=(N, M)) for _ in range(num_neurons)]
# Weight matrix for output neuron (num_neurons x M)
x0_output = np.random.uniform(low=xmin, high=xmax, size=(num_neurons, M))

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit():
    circuit = Circuit('Fully Connected Layer with Reset')
    ic = {}
    
    # Define models (matching reset circuit exactly)
    circuit.model('IHM_NMOS_HFOX', 'nmos', **NMOS_PARAMS)
    circuit.model('IHM_PMOS', 'pmos', **PMOS_PARAMS)
    circuit.model('IHM_PMOS2', 'pmos', **PMOS2_PARAMS)
    circuit.model('IHM_NMOS_HFOX_RESET', 'nmos', **RESET_PARAMS)
    
    circuit.V('VDD', 'vdd', circuit.gnd, VDD)
    
    # ══════════════════════════════════════════════════════════════════════════
    # BUILD THREE INPUT NEURONS (Layer 1)
    # ══════════════════════════════════════════════════════════════════════════
    
    for neuron in range(num_neurons):
        p = f'n{neuron}_'
        times = times_list[neuron]
        voltages = voltages_list[neuron]
        x0 = x0_neurons[neuron]
        
        # Row wires
        for i in range(N):
            for j in range(M - 1):
                circuit.R(f'{p}wire_row_{i}_{j}', f'{p}row_{i}_{j}', f'{p}row_{i}_{j+1}', R_wire_wl)
        
        # Column wires
        for j in range(M):
            for i in range(N - 1):
                circuit.R(f'{p}wire_col_{i}_{j}', f'{p}col_{i}_{j}', f'{p}col_{i+1}_{j}', R_wire_bl)
        
        # Memcapacitors and cross capacitors
        for i in range(N):
            for j in range(M):
                circuit.X(f'{p}M{i*M + j}', 'MEMCAP', f'{p}row_{i}_{j}', f'{p}col_{i}_{j}', f'{p}sv_{i}_{j}')
                ic[f'{p}sv_{i}_{j}'] = x0[i, j]
                circuit.C(f'{p}C_cross_{i}_{j}', f'{p}row_{i}_{j}', f'{p}col_{i}_{j}', C_cross)
        
        # Input voltage sources (PWL from MNIST)
        for i in range(N):
            pwl_values = [(float(t), float(v)) for t, v in zip(times[i], voltages[i])]
            circuit.R(f'{p}R_load_left_{i}', f'{p}row_{i}_0', f'{p}left_{i}', Rload)
            circuit.PieceWiseLinearVoltageSource(f'{p}V{i}', f'{p}left_{i}', circuit.gnd, values=pwl_values)
        
        # Right terminations (open)
        for i in range(N):
            circuit.R(f'{p}R_load_right_{i}', f'{p}row_{i}_{M-1}', f'{p}right_{i}', Ropen)
            circuit.V(f'{p}V_right_{i}', f'{p}right_{i}', circuit.gnd, 0)
        
        # Top terminations (open)
        for j in range(M):
            circuit.R(f'{p}R_load_top_{j}', f'{p}col_0_{j}', f'{p}top_{j}', Ropen)
            circuit.V(f'{p}V_top_{j}', f'{p}top_{j}', circuit.gnd, 0)
        
        # Gate capacitors
        for j in range(M):
            circuit.C(f'{p}C{j}', f'{p}col_{N-1}_{j}', circuit.gnd, C_gb)
        
        # Output NMOS transistors (stacked)
        circuit.MOSFET(f'{p}M_out0', f'{p}drain_0', f'{p}col_{N-1}_0', f'{p}source_0', f'{p}bulk', model='IHM_NMOS_HFOX')
        circuit.R(f'{p}R_out_left', f'{p}out_left', f'{p}source_0', 1)
        circuit.V(f'{p}V_out_left', f'{p}out_left', circuit.gnd, 0)
        
        circuit.MOSFET(f'{p}M_out1', f'{p}drain_1', f'{p}col_{N-1}_1', f'{p}source_1', f'{p}bulk', model='IHM_NMOS_HFOX')
        circuit.R(f'{p}Rs1', f'{p}drain_0', f'{p}source_1', Rs)
        
        circuit.MOSFET(f'{p}M_out2', f'{p}drain_2', f'{p}col_{N-1}_2', f'{p}source_2', f'{p}bulk', model='IHM_NMOS_HFOX')
        circuit.R(f'{p}Rs2', f'{p}drain_1', f'{p}source_2', Rs)
        
        circuit.R(f'{p}R_bulk', f'{p}bulk', circuit.gnd, 0.1)
        
        # Reset transistors (connected to N006)
        circuit.MOSFET(f'{p}M_reset0', f'{p}col_{N-1}_0', f'{p}N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
        circuit.MOSFET(f'{p}M_reset1', f'{p}col_{N-1}_1', f'{p}N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
        circuit.MOSFET(f'{p}M_reset2', f'{p}col_{N-1}_2', f'{p}N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
        
        # Current mirror
        circuit.R(f'{p}Rout_s', f'{p}drain_2', f'{p}N009', Rs)
        circuit.MOSFET(f'{p}M_cm1', f'{p}N009', f'{p}N009', 'vdd', 'vdd', model='IHM_PMOS2')
        circuit.MOSFET(f'{p}M_cm2', f'{p}N006', f'{p}N009', 'vdd', 'vdd', model='IHM_PMOS')
        circuit.R(f'{p}Rout', f'{p}N006', circuit.gnd, Rout)
    
    # ══════════════════════════════════════════════════════════════════════════
    # BUILD OUTPUT NEURON (Layer 2)
    # ══════════════════════════════════════════════════════════════════════════
    
    p = 'out_'
    x0 = x0_output
    N_out = num_neurons  # 3 inputs from layer 1
    M_out = M  # Same number of columns
    
    # Row wires for output neuron (connecting inputs from layer 1)
    for i in range(N_out):
        for j in range(M_out - 1):
            circuit.R(f'{p}wire_row_{i}_{j}', f'{p}row_{i}_{j}', f'{p}row_{i}_{j+1}', R_wire_wl)
    
    # Column wires
    for j in range(M_out):
        for i in range(N_out - 1):
            circuit.R(f'{p}wire_col_{i}_{j}', f'{p}col_{i}_{j}', f'{p}col_{i+1}_{j}', R_wire_bl)
    
    # Connect input neuron outputs (N006) to output neuron rows
    for i in range(num_neurons):
        circuit.R(f'{p}R_in_{i}', f'n{i}_N006', f'{p}row_{i}_0', 1)
    
    # Memcapacitors and cross capacitors for output neuron
    for i in range(N_out):
        for j in range(M_out):
            circuit.X(f'{p}M{i*M_out + j}', 'MEMCAP', f'{p}row_{i}_{j}', f'{p}col_{i}_{j}', f'{p}sv_{i}_{j}')
            ic[f'{p}sv_{i}_{j}'] = x0[i, j]
            circuit.C(f'{p}C_cross_{i}_{j}', f'{p}row_{i}_{j}', f'{p}col_{i}_{j}', C_cross)
    
    # Right terminations (open)
    for i in range(N_out):
        circuit.R(f'{p}R_load_right_{i}', f'{p}row_{i}_{M_out-1}', f'{p}right_{i}', Ropen)
        circuit.V(f'{p}V_right_{i}', f'{p}right_{i}', circuit.gnd, 0)
    
    # Top terminations (open)
    for j in range(M_out):
        circuit.R(f'{p}R_load_top_{j}', f'{p}col_0_{j}', f'{p}top_{j}', Ropen)
        circuit.V(f'{p}V_top_{j}', f'{p}top_{j}', circuit.gnd, 0)
    
    # Gate capacitors
    for j in range(M_out):
        circuit.C(f'{p}C{j}', f'{p}col_{N_out-1}_{j}', circuit.gnd, C_gb)
    
    # Output NMOS transistors (stacked)
    circuit.MOSFET(f'{p}M_out0', f'{p}drain_0', f'{p}col_{N_out-1}_0', f'{p}source_0', f'{p}bulk', model='IHM_NMOS_HFOX')
    circuit.R(f'{p}R_out_left', f'{p}out_left', f'{p}source_0', 1)
    circuit.V(f'{p}V_out_left', f'{p}out_left', circuit.gnd, 0)
    
    circuit.MOSFET(f'{p}M_out1', f'{p}drain_1', f'{p}col_{N_out-1}_1', f'{p}source_1', f'{p}bulk', model='IHM_NMOS_HFOX')
    circuit.R(f'{p}Rs1', f'{p}drain_0', f'{p}source_1', Rs)
    
    circuit.MOSFET(f'{p}M_out2', f'{p}drain_2', f'{p}col_{N_out-1}_2', f'{p}source_2', f'{p}bulk', model='IHM_NMOS_HFOX')
    circuit.R(f'{p}Rs2', f'{p}drain_1', f'{p}source_2', Rs)
    
    circuit.R(f'{p}R_bulk', f'{p}bulk', circuit.gnd, 0.1)
    
    # Reset transistors for output neuron
    circuit.MOSFET(f'{p}M_reset0', f'{p}col_{N_out-1}_0', f'{p}N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
    circuit.MOSFET(f'{p}M_reset1', f'{p}col_{N_out-1}_1', f'{p}N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
    circuit.MOSFET(f'{p}M_reset2', f'{p}col_{N_out-1}_2', f'{p}N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
    
    # Current mirror for output neuron
    circuit.R(f'{p}Rout_s', f'{p}drain_2', f'{p}N009', Rs)
    circuit.MOSFET(f'{p}M_cm1', f'{p}N009', f'{p}N009', 'vdd', 'vdd', model='IHM_PMOS2')
    circuit.MOSFET(f'{p}M_cm2', f'{p}N006', f'{p}N009', 'vdd', 'vdd', model='IHM_PMOS')
    circuit.R(f'{p}Rout', f'{p}N006', circuit.gnd, Rout)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SAVE NODES AND OPTIONS
    # ══════════════════════════════════════════════════════════════════════════
    
    circuit.raw_spice += get_subcircuit('MEMCAP')
    
    # Save input neuron signals
    for neuron in range(num_neurons):
        p = f'n{neuron}_'
        for i in range(N):
            circuit.raw_spice += f".save v({p}left_{i})\n"
        for j in range(M):
            circuit.raw_spice += f".save v({p}col_{N-1}_{j})\n"
        circuit.raw_spice += f".save v({p}N009)\n"
        circuit.raw_spice += f".save v({p}N006)\n"
    
    # Save output neuron signals
    p = 'out_'
    for j in range(M):
        circuit.raw_spice += f".save v({p}col_{num_neurons-1}_{j})\n"
    circuit.raw_spice += f".save v({p}N009)\n"
    circuit.raw_spice += f".save v({p}N006)\n"
    
    circuit.raw_spice += ".options plotwinsize=0 method=gear reltol=1e-4\n"
    
    # Initial conditions
    for neuron in range(num_neurons):
        circuit.raw_spice += f".ic V(n{neuron}_N006)=0\n"
    circuit.raw_spice += ".ic V(out_N006)=0\n"
    
    return circuit, ic

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION AND DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(circuit, ic):
    sim = circuit.simulator(temperature=25, nominal_parameters=[])
    sim.initial_condition(**ic)
    return sim.transient(step_time=100e-9, end_time=total_time)

def extract_data(analysis):
    data = {
        't': np.array(analysis.time),
        'V_input': [],
        'V_gate_in': [],
        'V_Rout_in': [],
        'V_Rout1_in': [],
        'V_gate_out': [],
        'V_Rout_out': None,
        'V_Rout1_out': None,
    }
    
    # Extract data for each input neuron
    for neuron in range(num_neurons):
        p = f'n{neuron}_'
        data['V_input'].append([np.array(analysis[f'{p}left_{i}']) for i in range(N)])
        data['V_gate_in'].append([np.array(analysis[f'{p}col_{N-1}_{j}']) for j in range(M)])
        data['V_Rout_in'].append(np.array(analysis[f'{p}n009']))
        data['V_Rout1_in'].append(np.array(analysis[f'{p}n006']))
    
    # Extract data for output neuron
    p = 'out_'
    data['V_gate_out'] = [np.array(analysis[f'{p}col_{num_neurons-1}_{j}']) for j in range(M)]
    data['V_Rout_out'] = np.array(analysis[f'{p}n009'])
    data['V_Rout1_out'] = np.array(analysis[f'{p}n006'])
    
    return data

# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTION (matching your format exactly)
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(data, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    t_us = data['t'] * 1e6
    
    colors_neuron = ['#E91E63', '#9C27B0', '#2196F3']
    colors_gate = ['#E74C3C', '#27AE60', '#3498DB']
    colors_input = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#795548']
    
    # ══════════════════════════════════════════════════════════════════════════
    # WINDOW 1: INPUT NEURONS (Layer 1)
    # ══════════════════════════════════════════════════════════════════════════
    
    fig1, axes1 = plt.subplots(5, 3, figsize=(16, 12))
    
    for neuron in range(num_neurons):
        # Row 0: Digit image
        axes1[0, neuron].imshow(digit_images[neuron], cmap='gray')
        axes1[0, neuron].set_title(f'Neuron {neuron+1} (Digit {digits[neuron]})', fontsize=11, fontweight='bold')
        axes1[0, neuron].axis('off')
        
        # Row 1: V(in) [V]
        for i in range(N):
            axes1[1, neuron].plot(t_us, data['V_input'][neuron][i], color=colors_input[i], lw=0.8)
        axes1[1, neuron].set_ylim(-0.1, 1.1)
        axes1[1, neuron].set_xlim(0, t_us[-1])
        axes1[1, neuron].grid(True, alpha=0.3)
        if neuron == 0:
            axes1[1, neuron].set_ylabel('V(in) [V]')
        
        # Row 2: V(gate) [V]
        for j in range(M):
            axes1[2, neuron].plot(t_us, data['V_gate_in'][neuron][j], color=colors_gate[j], lw=1)
        axes1[2, neuron].axhline(NMOS_VTH0, color='k', ls='--', lw=1.5, alpha=0.7)
        axes1[2, neuron].set_ylim(-0.05, 0.8)
        axes1[2, neuron].set_xlim(0, t_us[-1])
        axes1[2, neuron].grid(True, alpha=0.3)
        if neuron == 0:
            axes1[2, neuron].set_ylabel('V(Gate) [V]')
        
        # Row 3: V(Rout) [V]
        axes1[3, neuron].plot(t_us, data['V_Rout_in'][neuron], color='#8E44AD', lw=1.2)
        axes1[3, neuron].set_xlim(0, t_us[-1])
        axes1[3, neuron].grid(True, alpha=0.3)
        if neuron == 0:
            axes1[3, neuron].set_ylabel('Input_CM [V]')
        
        # Row 4: V(Rout1) [V] → to Layer 2
        axes1[4, neuron].plot(t_us, data['V_Rout1_in'][neuron], color='#16A085', lw=1.2)
        axes1[4, neuron].axhline(RESET_VTH0, color='r', ls='--', lw=1, alpha=0.7)
        axes1[4, neuron].set_ylim(-0.05, 1.1)
        axes1[4, neuron].set_xlim(0, t_us[-1])
        axes1[4, neuron].set_xlabel('Time (µs)')
        axes1[4, neuron].grid(True, alpha=0.3)
        if neuron == 0:
            axes1[4, neuron].set_ylabel('Output [V]')
        
        # Add max voltage annotation
        axes1[4, neuron].annotate(f'Max: {np.max(data["V_Rout1_in"][neuron]):.2f}V',
                                   xy=(0.98, 0.85), xycoords='axes fraction', ha='right', va='top', fontsize=8,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # ══════════════════════════════════════════════════════════════════════════
    # WINDOW 2: OUTPUT NEURON (Layer 2)
    # ══════════════════════════════════════════════════════════════════════════
    
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Row 0: Input from L1 V(Rout1) [V]
    for neuron in range(num_neurons):
        axes2[0].plot(t_us, data['V_Rout1_in'][neuron], color=colors_neuron[neuron], lw=1.2, 
                      label=f'From Neuron {neuron+1}')
    axes2[0].set_ylabel('Input from L1\n[V]')
    axes2[0].set_ylim(-0.05, 1.1)

    axes2[0].grid(True, alpha=0.3)
    
    # Row 1: V(gate) [V]
    for j in range(M):
        axes2[1].plot(t_us, data['V_gate_out'][j], color=colors_gate[j], lw=1.2, label=f'Col {j}')
    axes2[1].axhline(NMOS_VTH0, color='k', ls='--', lw=1.5, alpha=0.7, label='Vth')
    axes2[1].set_ylabel('V(Gate) [V]')
    axes2[1].set_ylim(-0.05, 0.8)
    axes2[1].grid(True, alpha=0.3)
    
    # Row 2: V(Rout) [V]
    axes2[2].plot(t_us, data['V_Rout_out'], '#8E44AD', lw=1.2)
    axes2[2].set_ylabel('Input_CM [V]')
    axes2[2].grid(True, alpha=0.3)
    
    # Row 3: V(Rout1) [V] Final Output
    axes2[3].plot(t_us, data['V_Rout1_out'], '#16A085', lw=1.2)
    axes2[3].axhline(RESET_VTH0, color='r', ls='--', lw=1, alpha=0.7, label='Reset Vth')
    axes2[3].set_ylabel('Output[V]')
    axes2[3].set_xlabel('Time (µs)')
    axes2[3].set_ylim(-0.05, 1.1)
    axes2[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
   
    
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    
    circuit, ic = build_circuit()
    
    analysis = run_simulation(circuit, ic)
    
    data = extract_data(analysis)

    plot_results(data)
    
