"""Neurotransistor WITHOUT Reset - MNIST Input
   Matching LTspice schematic (without reset NMOS)
   CORRECTED: All MOSFETs use LEVEL=14 (BSIM4)
"""

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from memcap_model import get_memcap_subcircuit
from MNIST import create_MNIST_pulse_train, load_mnist

mpl.rcParams['axes.formatter.useoffset'] = False

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

memristor = 'MEMCAP'
memristor_subcircuit = get_memcap_subcircuit()

N = 6  # Rows (6 inputs)
M = 3  # Columns

# ══════════════════════════════════════════════════════════════════════════════
# CAPACITOR & RESISTOR PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

C_cross = 10e-12      # Cross capacitor (10pF)
C_col = 100e-12       # Column capacitors C1, C2, C3 (100pF)
R_wire_wl = 0.33      # Wordline wire resistance
R_wire_bl = 60        # Bitline wire resistance
Rload = 1             # Input load resistor
Ropen = 1e9           # Open circuit resistor

# ══════════════════════════════════════════════════════════════════════════════
# NMOS PARAMETERS (LEVEL=14 BSIM4) - M10, M11, M12
# ══════════════════════════════════════════════════════════════════════════════

NMOS_LEVEL = 14
NMOS_W = 94e-6
NMOS_L = 26e-6
NMOS_VTH0 = 0.3
NMOS_TOXE = 26e-9
NMOS_EPSROX = 22
NMOS_CGBO = 69e-9
NMOS_CGDO = 56e-9
NMOS_CGSO = 56e-9

# ══════════════════════════════════════════════════════════════════════════════
# PMOS PARAMETERS (LEVEL=14 BSIM4) - Current Mirror M13, M14
# ══════════════════════════════════════════════════════════════════════════════

PMOS_LEVEL = 14       # CORRECTED: Was 3, now 14
PMOS_W = 94e-6        # M14 (IHM_PMOS) - larger, mirror output
PMOS_L = 26e-6
PMOS_VTH0 = -0.3      # CORRECTED: Was Vto, now VTH0
PMOS_TOXE = 22e-9     # CORRECTED: Was Tox, now TOXE
PMOS_EPSROX = 22      # ADDED: Required for LEVEL=14

PMOS2_W = 24e-6       # M13 (IHM_PMOS2) - smaller, diode-connected
PMOS2_L = 26e-6

# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

Rs = 10               # {Rs} = 10Ω (for Rout1, R1, R2)
Rout = 200e3          # {Rout} = 200kΩ (for Rout2)
Rt = 10               # Inter-stage resistor (R1, R2)

VDD = 1.0
R_out_left = 1
V_out_left = 0
V_out_right = VDD

pulse_voltage = 1
pulse_width = 20e-6
pulse_slope = 200e-9
rel_tol = 1e-4

# ══════════════════════════════════════════════════════════════════════════════
# INITIAL STATE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

xmin = 0.1
xmax = 0.284
seed = 42
np.random.seed(seed)
x0 = np.random.uniform(low=xmin, high=xmax, size=(N, M))

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MNIST
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
# BUILD CIRCUIT - WITHOUT RESET
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit():
    circuit = Circuit('Neurotransistor WITHOUT Reset - MNIST')
    initial_conditions = {}
    
    # ──────────────────────────────────────────────────────────────────────────
    # MOSFET MODELS - ALL LEVEL=14 (BSIM4)
    # ──────────────────────────────────────────────────────────────────────────
    
    # IHM_NMOS_HFOX for M10, M11, M12
    circuit.model('IHM_NMOS_HFOX', 'nmos', 
                  LEVEL=NMOS_LEVEL, L=NMOS_L, W=NMOS_W,
                  VTH0=NMOS_VTH0, TOXE=NMOS_TOXE, EPSROX=NMOS_EPSROX,
                  CGBO=NMOS_CGBO, CGDO=NMOS_CGDO, CGSO=NMOS_CGSO)
    
    # IHM_PMOS for M14 (current mirror output, W=94µ) - CORRECTED!
    circuit.model('IHM_PMOS', 'pmos', 
                  LEVEL=PMOS_LEVEL, L=PMOS_L, W=PMOS_W,
                  VTH0=PMOS_VTH0, TOXE=PMOS_TOXE, EPSROX=PMOS_EPSROX)
    
    # IHM_PMOS2 for M13 (diode-connected, W=24µ) - CORRECTED!
    circuit.model('IHM_PMOS2', 'pmos', 
                  LEVEL=PMOS_LEVEL, L=PMOS2_L, W=PMOS2_W,
                  VTH0=PMOS_VTH0, TOXE=PMOS_TOXE, EPSROX=PMOS_EPSROX)
    
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
    # MEMCAPACITORS M1-M9 + CROSS CAPACITORS
    # ──────────────────────────────────────────────────────────────────────────
    
    for i in range(N):
        for j in range(M):
            circuit.X(f'M_{i}_{j}', memristor, f'row_{i}_{j}', f'col_{i}_{j}', f'sv_{i}_{j}')
            initial_conditions[f'sv_{i}_{j}'] = x0[i, j]
            circuit.C(f'C_cross_{i}_{j}', f'row_{i}_{j}', f'col_{i}_{j}', C_cross)
    
    # ──────────────────────────────────────────────────────────────────────────
    # INPUT SOURCES V1-V6
    # ──────────────────────────────────────────────────────────────────────────
    
    for i in range(N):
        pwl_values = [(t, v) for t, v in zip(times[i], voltages[i])]
        circuit.R(f'R_load_left_{i}', f'row_{i}_0', f'left_{i}', Rload)
        circuit.PieceWiseLinearVoltageSource(f'V{i+1}', f'left_{i}', circuit.gnd, values=pwl_values)
    
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
    # COLUMN CAPACITORS C1, C2, C3 (100pF) - Connected to GND
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.C('C1', f'col_{N-1}_0', circuit.gnd, C_col)
    circuit.C('C2', f'col_{N-1}_1', circuit.gnd, C_col)
    circuit.C('C3', f'col_{N-1}_2', circuit.gnd, C_col)
    
    # ──────────────────────────────────────────────────────────────────────────
    # BOTTOM NMOS TRANSISTORS M10, M11, M12
    # ──────────────────────────────────────────────────────────────────────────
    
    # M10: gate=col_{N-1}_0
    circuit.MOSFET('M10', 'drain_0', f'col_{N-1}_0', 'source_0', 'bulk',
                   model='IHM_NMOS_HFOX')
    circuit.R('R_out_left', 'out_left', 'source_0', R_out_left)
    circuit.V('V_out_left', 'out_left', circuit.gnd, V_out_left)
    
    # M11: gate=col_{N-1}_1
    circuit.MOSFET('M11', 'drain_1', f'col_{N-1}_1', 'source_1', 'bulk',
                   model='IHM_NMOS_HFOX')
    circuit.R('R2', 'drain_0', 'source_1', Rs)
    
    # M12: gate=col_{N-1}_2
    circuit.MOSFET('M12', 'drain_2', f'col_{N-1}_2', 'source_2', 'bulk',
                   model='IHM_NMOS_HFOX')
    circuit.R('R1', 'drain_1', 'source_2', Rs)
    
    # Bulk connection
    circuit.R('R_bulk', 'bulk', circuit.gnd, 0.1)
    
    # Rout1: drain_2 → N025 (10Ω)
    circuit.R('Rout1', 'drain_2', 'N025', Rs)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PMOS CURRENT MIRROR M13, M14
    # ──────────────────────────────────────────────────────────────────────────
    
    # VDD for PMOS current mirror
    circuit.V('V_out_right', 'Vright', circuit.gnd, V_out_right)
    
    # M13 (IHM_PMOS2): diode-connected, W=24µm
    # Drain=N025, Gate=N025, Source=VDD, Bulk=VDD
    circuit.MOSFET('M13', 'N025', 'N025', 'Vright', 'Vright', model='IHM_PMOS2')
    
    # M14 (IHM_PMOS): mirror output, W=94µm
    # Drain=N023, Gate=N025, Source=VDD, Bulk=VDD
    circuit.MOSFET('M14', 'N023', 'N025', 'Vright', 'Vright', model='IHM_PMOS')
    
    # Rout2: N023 → GND (200kΩ)
    circuit.R('Rout2', 'N023', circuit.gnd, Rout)
    
    # ──────────────────────────────────────────────────────────────────────────
    # SPICE DIRECTIVES
    # ──────────────────────────────────────────────────────────────────────────
    
    circuit.raw_spice += memristor_subcircuit
    
    # Save input voltages
    for i in range(N):
        circuit.raw_spice += f".save v(left_{i})\n"
    
    # Save column/gate voltages
    for j in range(M):
        circuit.raw_spice += f".save v(col_{N-1}_{j})\n"
    
    # Save output voltages
    circuit.raw_spice += ".save v(N025)\n"
    circuit.raw_spice += ".save v(N023)\n"
    circuit.raw_spice += ".probe I(RRout1)\n"
    
    # Save state variables
    for i in range(N):
        for j in range(M):
            circuit.raw_spice += f".save v(sv_{i}_{j})\n"
    
    # Simulation options
    circuit.raw_spice += ".options plotwinsize=0\n"
    circuit.raw_spice += ".options method=gear\n"
    circuit.raw_spice += f".options reltol={rel_tol}\n"
    
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
        'V_Rout1': np.array(analysis['n025']),
        'V_Rout2': np.array(analysis['n023']),
        'I_Rout1': np.array(analysis['rrout1']),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PLOT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(data, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    t_us = data['t'] * 1e6

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    fig.suptitle(f'Neurotransistor MNIST Digit {digit}\n'
                 f'Vth = {NMOS_VTH0}V | Rout = {Rout/1e3:.0f}kΩ | All LEVEL=14',
                 fontsize=12, fontweight='bold')

    # Panel 1: Input Voltage
    colors_input = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i in range(N):
        axes[0].plot(t_us, data['V_input'][i], color=colors_input[i % len(colors_input)], lw=0.8)
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_title('Input Voltage')
    axes[0].grid(True, alpha=0.3)
    
    inset = axes[0].inset_axes([0.92, 0.3, 0.07, 0.6])
    inset.imshow(digit_image, cmap='gray')
    inset.set_title(f'{digit}', fontsize=9)
    inset.axis('off')

    # Panel 2: Gate Voltage
    axes[1].plot(t_us, data['V_gate'][0], '#F44336', lw=1)
    axes[1].plot(t_us, data['V_gate'][1], '#4CAF50', lw=1)
    axes[1].plot(t_us, data['V_gate'][2], '#2196F3', lw=1)
    axes[1].axhline(NMOS_VTH0, color='black', ls='--', lw=1.5)
    axes[1].set_ylabel('Voltage (V)')
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title('Gate Voltage')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: V(Rout1) at N025
    axes[2].plot(t_us, data['V_Rout1'], '#2196F3', lw=1)
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('V(Rout1) at N025')
    axes[2].grid(True, alpha=0.3)

    # Panel 4: V(Rout2) at N023
    axes[3].plot(t_us, data['V_Rout2'], '#9C27B0', lw=1)
    axes[3].set_ylabel('Voltage (V)')
    axes[3].set_title('V(Rout2) at N023')
    axes[3].grid(True, alpha=0.3)

    # Panel 5: I(Rout1)
    axes[4].plot(t_us, data['I_Rout1']*1e6, '#E91E63', lw=1)
    axes[4].set_ylabel('Current (µA)')
    axes[4].set_xlabel('Time (µs)')
    axes[4].set_title('I(Rout1)')
    axes[4].axhline(0, color='black', ls='-', lw=0.5)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/neuron_no_reset_mnist_digit{digit}.png', dpi=300, bbox_inches='tight')
    plt.show()

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
