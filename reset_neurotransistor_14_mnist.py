"""Neurotransistor with Gate Reset - MNIST Input + I(Rout) Plot
   CORRECTED: All MOSFETs use LEVEL=14 (BSIM4)
"""

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from memcap_model import get_subcircuit
from MNIST import create_MNIST_pulse_train, load_mnist

mpl.rcParams['axes.formatter.useoffset'] = False

# ══════════════════════════════════════════════════════════════════════════════
# ARRAY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

N = 6
M = 3

# ══════════════════════════════════════════════════════════════════════════════
# CAPACITOR & RESISTOR PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

C_cross = 10e-12
C_gb = 100e-12
R_wire_wl = 0.33
R_wire_bl = 60
Rload = 1
Ropen = 1e9

# ══════════════════════════════════════════════════════════════════════════════
# NMOS PARAMETERS (LEVEL=14 BSIM4) - M66, M67, M68
# ══════════════════════════════════════════════════════════════════════════════

NMOS_LEVEL = 14
NMOS_L = 26e-6
NMOS_W = 94e-6
NMOS_VTH0 = 0.3
NMOS_TOXE = 26e-9
NMOS_EPSROX = 22
NMOS_CGBO = 69e-9
NMOS_CGDO = 56e-9
NMOS_CGSO = 56e-9

# ══════════════════════════════════════════════════════════════════════════════
# PMOS PARAMETERS (LEVEL=14 BSIM4) - M69, M70
# CORRECTED: Changed from LEVEL=3 to LEVEL=14
# ══════════════════════════════════════════════════════════════════════════════

PMOS_LEVEL = 14           # CORRECTED: Was 3
PMOS_L = 26e-6
PMOS_W = 94e-6            # M70 (mirror output)
PMOS_VTH0 = -0.3          # CORRECTED: Was Vto
PMOS_TOXE = 22e-9         # CORRECTED: Was Tox
PMOS_EPSROX = 22          # ADDED

PMOS2_W = 24e-6           # M69 (diode-connected)
PMOS2_L = 26e-6

# ══════════════════════════════════════════════════════════════════════════════
# RESET NMOS PARAMETERS (LEVEL=14 BSIM4) - M71, M72, M73
# ══════════════════════════════════════════════════════════════════════════════

RESET_LEVEL = 14
RESET_L = 10e-6
RESET_W = 10e-6
RESET_VTH0 = 0.6
RESET_TOXE = 26e-9
RESET_EPSROX = 22
RESET_CGBO = 69e-9
RESET_CGDO = 56e-9
RESET_CGSO = 56e-9

# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

C_out = 0
Rs = 10
Rout = 200e3
Rt = 10
VDD = 1.0
R_out_left = 1
V_out_left = 0
rel_tol = 1e-6

# ══════════════════════════════════════════════════════════════════════════════
# PULSE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

pulse_voltage = 1.0
pulse_width = 20e-6
pulse_slope = 200e-9

# ══════════════════════════════════════════════════════════════════════════════
# INITIAL STATE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

xmin = 0.1
xmax = 0.284
seed = 42
np.random.seed(seed)
x0 = np.random.uniform(low=xmin, high=xmax, size=(N, M))

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MNIST DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading MNIST...")
imx, imy = load_mnist('raw', kind='train')

digit = 3
index = 0

pulse_trains, digit_image = create_MNIST_pulse_train(
    imx, imy, N, pulse_voltage, pulse_width, pulse_slope,
    selected_digits=[digit], do_plot=None, specific_image_index=index
)

times = pulse_trains[digit]['times']
voltages = pulse_trains[digit]['voltages']
total_time = times[0][-1]

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CIRCUIT (WITH RESET)
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit():
    circuit = Circuit('Neurotransistor with Gate Reset - MNIST')
    initial_conditions = {}
    
    # ──────────────────────────────────────────────────────────────────────────
    # MOSFET MODELS - ALL LEVEL=14 (BSIM4)
    # ──────────────────────────────────────────────────────────────────────────
    
    # IHM_NMOS_HFOX for M66, M67, M68
    circuit.model('IHM_NMOS_HFOX', 'nmos', 
                  LEVEL=NMOS_LEVEL, L=NMOS_L, W=NMOS_W,
                  VTH0=NMOS_VTH0, TOXE=NMOS_TOXE, EPSROX=NMOS_EPSROX,
                  CGBO=NMOS_CGBO, CGDO=NMOS_CGDO, CGSO=NMOS_CGSO)
    
    # IHM_PMOS for M70 (current mirror output, W=94µm) - CORRECTED!
    circuit.model('IHM_PMOS', 'pmos', 
                  LEVEL=PMOS_LEVEL, L=PMOS_L, W=PMOS_W,
                  VTH0=PMOS_VTH0, TOXE=PMOS_TOXE, EPSROX=PMOS_EPSROX)
    
    # IHM_PMOS2 for M69 (diode-connected, W=24µm) - CORRECTED!
    circuit.model('IHM_PMOS2', 'pmos', 
                  LEVEL=PMOS_LEVEL, L=PMOS2_L, W=PMOS2_W,
                  VTH0=PMOS_VTH0, TOXE=PMOS_TOXE, EPSROX=PMOS_EPSROX)
    
    # IHM_NMOS_HFOX_RESET for M71, M72, M73
    circuit.model('IHM_NMOS_HFOX_RESET', 'nmos', 
                  LEVEL=RESET_LEVEL, L=RESET_L, W=RESET_W,
                  VTH0=RESET_VTH0, TOXE=RESET_TOXE, EPSROX=RESET_EPSROX,
                  CGBO=RESET_CGBO, CGDO=RESET_CGDO, CGSO=RESET_CGSO)
    
    # ──────────────────────────────────────────────────────────────────────────
    # POWER SUPPLY
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.V('VDD', 'vdd', circuit.gnd, VDD)
    
    # ──────────────────────────────────────────────────────────────────────────
    # WIRE RESISTANCES - Wordlines
    # ──────────────────────────────────────────────────────────────────────────
    
    for i in range(N):
        for j in range(M - 1):
            circuit.R(f'wire_row_{i}_{j}', f'row_{i}_{j}', f'row_{i}_{j+1}', R_wire_wl)
    
    # ──────────────────────────────────────────────────────────────────────────
    # WIRE RESISTANCES - Bitlines
    # ──────────────────────────────────────────────────────────────────────────
    
    for j in range(M):
        for i in range(N - 1):
            circuit.R(f'wire_col_{i}_{j}', f'col_{i}_{j}', f'col_{i+1}_{j}', R_wire_bl)
    
    # ──────────────────────────────────────────────────────────────────────────
    # MEMCAPACITORS + CROSS CAPACITORS
    # ──────────────────────────────────────────────────────────────────────────
    
    for i in range(N):
        for j in range(M):
            circuit.X(f'M{57 + i*M + j}', 'MEMCAP', f'row_{i}_{j}', f'col_{i}_{j}', f'sv_{i}_{j}')
            initial_conditions[f'sv_{i}_{j}'] = x0[i, j]
            circuit.C(f'C_cross_{i}_{j}', f'row_{i}_{j}', f'col_{i}_{j}', C_cross)
    
    # ──────────────────────────────────────────────────────────────────────────
    # INPUT SOURCES
    # ──────────────────────────────────────────────────────────────────────────
    
    for i in range(N):
        pwl_values = [(float(t), float(v)) for t, v in zip(times[i], voltages[i])]
        circuit.R(f'R_load_left_{i}', f'row_{i}_0', f'left_{i}', Rload)
        circuit.PieceWiseLinearVoltageSource(f'V{10+i}', f'left_{i}', circuit.gnd, values=pwl_values)
    
    # ──────────────────────────────────────────────────────────────────────────
    # RIGHT SIDE - Open circuit
    # ──────────────────────────────────────────────────────────────────────────
    
    for i in range(N):
        circuit.R(f'R_load_right_{i}', f'row_{i}_{M-1}', f'right_{i}', Ropen)
        circuit.V(f'V_right_{i}', f'right_{i}', circuit.gnd, 0)
    
    # ──────────────────────────────────────────────────────────────────────────
    # TOP SIDE - Open circuit
    # ──────────────────────────────────────────────────────────────────────────
    
    for j in range(M):
        circuit.R(f'R_load_top_{j}', f'col_0_{j}', f'top_{j}', Ropen)
        circuit.V(f'V_top_{j}', f'top_{j}', circuit.gnd, 0)
    
    # ──────────────────────────────────────────────────────────────────────────
    # COLUMN CAPACITORS C13, C14, C15 (100pF)
    # ──────────────────────────────────────────────────────────────────────────
    
    for j in range(M):
        circuit.C(f'C{13+j}', f'col_{N-1}_{j}', circuit.gnd, C_gb)
    
    # R11 = 0 (short circuit)
    circuit.R('R11', f'col_{N-1}_{M-1}', f'N005_short', 1e-3)
    
    # ──────────────────────────────────────────────────────────────────────────
    # BOTTOM NMOS TRANSISTORS M66, M67, M68
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.MOSFET('M66', 'drain_0', f'col_{N-1}_0', 'source_0', 'bulk',
                   model='IHM_NMOS_HFOX')
    circuit.R('R_out_left', 'out_left', 'source_0', R_out_left)
    circuit.V('V_out_left', 'out_left', circuit.gnd, V_out_left)
    
    circuit.MOSFET('M67', 'drain_1', f'col_{N-1}_1', 'source_1', 'bulk',
                   model='IHM_NMOS_HFOX')
    circuit.R('R10', 'drain_0', 'source_1', Rs)
    
    circuit.MOSFET('M68', 'drain_2', f'col_{N-1}_2', 'source_2', 'bulk',
                   model='IHM_NMOS_HFOX')
    circuit.R('R9', 'drain_1', 'source_2', Rs)
    
    circuit.R('R_bulk', 'bulk', circuit.gnd, 0.1)
    
    # ──────────────────────────────────────────────────────────────────────────
    # RESET TRANSISTORS M71, M72, M73
    # Gates connected to N006 (current mirror output)
    # Drains connected to column nodes (membrane capacitors)
    # Sources connected to GND
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.MOSFET('M71', f'col_{N-1}_0', 'N006', circuit.gnd, circuit.gnd,
                   model='IHM_NMOS_HFOX_RESET')
    circuit.MOSFET('M72', f'col_{N-1}_1', 'N006', circuit.gnd, circuit.gnd,
                   model='IHM_NMOS_HFOX_RESET')
    circuit.MOSFET('M73', f'col_{N-1}_2', 'N006', circuit.gnd, circuit.gnd,
                   model='IHM_NMOS_HFOX_RESET')
    
    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT STAGE - Current Mirror
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.R('Rout9', 'drain_2', 'N009', Rs)
    
    # M69: Diode-connected (gate=drain=N009)
    circuit.MOSFET('M69', 'N009', 'N009', 'vdd', 'vdd', model='IHM_PMOS2')
    
    # M70: Mirror output (gate=N009, drain=N006)
    circuit.MOSFET('M70', 'N006', 'N009', 'vdd', 'vdd', model='IHM_PMOS')
    
    # Output load
    circuit.R('Rout10', 'N006', circuit.gnd, Rout)
    circuit.C('C_out', 'N006', circuit.gnd, C_out)
    
    # ──────────────────────────────────────────────────────────────────────────
    # SPICE DIRECTIVES
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.raw_spice += get_subcircuit('MEMCAP')
    
    for i in range(N):
        circuit.raw_spice += f".save v(left_{i})\n"
    
    for j in range(M):
        circuit.raw_spice += f".save v(col_{N-1}_{j})\n"
    
    circuit.raw_spice += ".save v(N009)\n"
    circuit.raw_spice += ".save v(N006)\n"
    circuit.raw_spice += ".probe I(RRout9)\n"
    
    for i in range(N):
        for j in range(M):
            circuit.raw_spice += f".save v(sv_{i}_{j})\n"
    
    circuit.raw_spice += ".options plotwinsize=0\n"
    circuit.raw_spice += ".options method=gear\n"
    circuit.raw_spice += f".options reltol={rel_tol}\n"
    circuit.raw_spice += ".ic V(N006)=0\n"
    
    return circuit, initial_conditions

# ══════════════════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(circuit, initial_conditions):
    simulator = circuit.simulator(temperature=25, nominal_parameters=[])
    simulator.initial_condition(**initial_conditions)
    return simulator.transient(step_time=100e-9, end_time=total_time)

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT DATA
# ══════════════════════════════════════════════════════════════════════════════

def extract_data(analysis):
    return {
        't': np.array(analysis.time),
        'V_input': [np.array(analysis[f'left_{i}']) for i in range(N)],
        'V_gate': [np.array(analysis[f'col_{N-1}_{j}']) for j in range(M)],
        'V_N009': np.array(analysis['n009']),
        'V_N006': np.array(analysis['n006']),
        'I_Rout9': np.array(analysis['rrout9']),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PLOT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(data, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    t_us = data['t'] * 1e6

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    
    fig.suptitle(f'Neurotransistor with Reset - MNIST Digit {digit}\n'
                 f'Vth = {NMOS_VTH0}V | Reset Vth = {RESET_VTH0}V | Rout = {int(Rout/1e3)}kΩ | All LEVEL=14',
                 fontsize=14, fontweight='bold')

    colors_input = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#795548']
    colors_gate = ['#E74C3C', '#27AE60', '#3498DB']

    # Panel 1: Input Voltage
    axes[0].set_title('Input Voltage', fontsize=11)
    for i in range(N):
        axes[0].plot(t_us, data['V_input'][i], color=colors_input[i], lw=0.8)
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_ylim(-0.1, 1.2)
    axes[0].grid(True, alpha=0.3)
    
    inset = axes[0].inset_axes([0.85, 0.55, 0.12, 0.4])
    inset.imshow(digit_image, cmap='gray')
    inset.text(0.5, 1.15, f'{digit}', transform=inset.transAxes, 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    inset.axis('off')

    # Panel 2: Gate Voltage
    axes[1].set_title('Gate Voltage (Membrane Potential)', fontsize=11)
    for j in range(M):
        axes[1].plot(t_us, data['V_gate'][j], color=colors_gate[j], lw=1.2)
    axes[1].axhline(NMOS_VTH0, color='k', ls='--', lw=1.5, alpha=0.7, label=f'Vth = {NMOS_VTH0}V')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].set_ylim(-0.05, 0.8)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: V(Rout9) at N009
    axes[2].set_title('V(Rout9) at N009 - Current Mirror Input', fontsize=11)
    axes[2].plot(t_us, data['V_N009'], '#16A085', lw=1.2)
    axes[2].set_ylabel('Voltage (V)')
    axes[2].grid(True, alpha=0.3)

    # Panel 4: V(Rout10) at N006 - WITH RESET THRESHOLD LINE
    axes[3].set_title('V(Rout10) at N006 - Output / Reset Control', fontsize=11)
    axes[3].plot(t_us, data['V_N006'], '#8E44AD', lw=1.2)
    axes[3].axhline(RESET_VTH0, color='red', ls='--', lw=1.5, alpha=0.8)
    axes[3].text(t_us[-1] * 0.02, RESET_VTH0 + 0.02, f'Reset Vth = {RESET_VTH0}V', 
                 color='red', fontsize=9, va='bottom')
    axes[3].set_ylabel('Voltage (V)')
    y_max = max(np.max(data['V_N006']) * 1.2, RESET_VTH0 * 1.2)
    axes[3].set_ylim(-0.05, y_max)
    axes[3].grid(True, alpha=0.3)

    # Panel 5: I(Rout9)
    axes[4].set_title('I(Rout9) - Current Mirror Current', fontsize=11)
    I_uA = data['I_Rout9'] * 1e6
    axes[4].plot(t_us, I_uA, '#E74C3C', lw=1.2)
    axes[4].set_ylabel('Current (µA)')
    axes[4].set_xlabel('Time (µs)')
    axes[4].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, t_us[-1])

    plt.tight_layout()
    plt.savefig(f'{save_path}/neuron_reset_mnist_digit{digit}_LEVEL14.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print reset status
    V_N006_max = np.max(data['V_N006'])
    print(f"\n{'='*60}")
    print(f"RESET STATUS CHECK")
    print(f"{'='*60}")
    print(f"Reset Threshold (Vth):  {RESET_VTH0} V")
    print(f"Max V(N006):            {V_N006_max:.4f} V")
    print(f"{'='*60}")
    
    if V_N006_max >= RESET_VTH0:
        print(f"RESET TRIGGERED!")
        reset_times = t_us[data['V_N006'] >= RESET_VTH0]
        if len(reset_times) > 0:
            print(f"   First reset at t = {reset_times[0]:.2f} µs")
    else:
        print(f" RESET NOT TRIGGERED.")
    print(f"{'='*60}\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building circuit...")
    circuit, initial_conditions = build_circuit()
    
    print("Running simulation...")
    analysis = run_simulation(circuit, initial_conditions)
    
    print("Extracting data...")
    data = extract_data(analysis)
    
    print("Plotting results...")
    plot_results(data)
    
    print("Done!")
