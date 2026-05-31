"""Neurotransistor WITH RESET - Auto-Scaling Y-Axis
"""

from PySpice.Spice.Netlist import Circuit
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from memcap_model import get_subcircuit

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

N, M = 3, 3

C_cross = 10e-12
C_gb = 100e-12
R_wire_wl = 0.33
R_wire_bl = 60
Rload = 1
Ropen = 1e9

Rs = 10
Rout = 200e3
VDD = 1.0

NMOS_PARAMS = dict(
    LEVEL=14, L=26e-6, W=94e-6, VTH0=0.3,
    TOXE=26e-9, EPSROX=22, CGBO=69e-9, CGDO=56e-9, CGSO=56e-9
)

PMOS_PARAMS = dict(
    LEVEL=14, L=26e-6, W=94e-6, VTH0=-0.3, TOXE=22e-9, EPSROX=22
)

PMOS2_PARAMS = dict(
    LEVEL=14, L=26e-6, W=24e-6, VTH0=-0.3, TOXE=22e-9, EPSROX=22
)

RESET_PARAMS = dict(
    LEVEL=14, L=10e-6, W=10e-6, VTH0=0.6,
    TOXE=26e-9, EPSROX=22, CGBO=69e-9, CGDO=56e-9, CGSO=56e-9
)

# PULSE PARAMETERS - Change these as needed
pulse_voltage = 0.8  # Can change to any value
pulse_width = 10e-6
pulse_slope = 1e-6
pulse_period = 30e-6
pulse_end_time = 100e-6
total_time = 120e-6

X_ON = 0.10
X_OFF = 0.28

# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════════════════════

def get_window1_cases():
    cases = {}
    cases['All Rows OFF\nAll Cols OFF'] = np.ones((N, M)) * X_OFF
    
    x0 = np.ones((N, M)) * X_OFF
    x0[:, 0] = X_ON
    cases['All Rows ON\nAll Cols OFF'] = x0.copy()
    
    x0 = np.ones((N, M)) * X_OFF
    x0[0, :] = X_ON
    cases['All Rows OFF\nAll Cols ON'] = x0.copy()
    
    cases['All Rows ON\nAll Cols ON'] = np.ones((N, M)) * X_ON
    return cases

def get_window2_cases():
    cases = {}
    
    x0 = np.ones((N, M)) * X_OFF
    x0[2, :] = X_ON
    x0[:, 1] = X_ON
    cases['1 Row ON\n1 Col ON'] = x0.copy()
    
    x0 = np.ones((N, M)) * X_OFF
    x0[1, :] = X_ON
    x0[2, :] = X_ON
    x0[:, 0] = X_ON
    x0[:, 1] = X_ON
    cases['2 Rows ON\n2 Cols ON'] = x0.copy()
    
    return cases

# ══════════════════════════════════════════════════════════════════════════════
# PULSE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_pulses():
    times, voltages = [], []
    n_pulses = int(pulse_end_time / pulse_period)
    
    for _ in range(N):
        t, v = [0], [0]
        for cycle in range(n_pulses):
            t_start = cycle * pulse_period + 2e-6
            t.extend([t_start, t_start + pulse_slope, 
                      t_start + pulse_slope + pulse_width,
                      t_start + pulse_slope + pulse_width + pulse_slope])
            v.extend([0, pulse_voltage, pulse_voltage, 0])
        t.extend([pulse_end_time, total_time])
        v.extend([0, 0])
        times.append(t)
        voltages.append(v)
    return times, voltages

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit(x0, times, voltages):
    circuit = Circuit('Neurotransistor WITH RESET')
    ic = {}
    
    circuit.model('IHM_NMOS_HFOX', 'nmos', **NMOS_PARAMS)
    circuit.model('IHM_PMOS', 'pmos', **PMOS_PARAMS)
    circuit.model('IHM_PMOS2', 'pmos', **PMOS2_PARAMS)
    circuit.model('IHM_NMOS_HFOX_RESET', 'nmos', **RESET_PARAMS)
    
    circuit.V('VDD', 'vdd', circuit.gnd, VDD)
    
    for i in range(N):
        for j in range(M - 1):
            circuit.R(f'wire_row_{i}_{j}', f'row_{i}_{j}', f'row_{i}_{j+1}', R_wire_wl)
    
    for j in range(M):
        for i in range(N - 1):
            circuit.R(f'wire_col_{i}_{j}', f'col_{i}_{j}', f'col_{i+1}_{j}', R_wire_bl)
    
    for i in range(N):
        for j in range(M):
            circuit.X(f'M{57 + i*M + j}', 'MEMCAP', f'row_{i}_{j}', f'col_{i}_{j}', f'sv_{i}_{j}')
            ic[f'sv_{i}_{j}'] = x0[i, j]
            circuit.C(f'C_cross_{i}_{j}', f'row_{i}_{j}', f'col_{i}_{j}', C_cross)
    
    for i in range(N):
        pwl = [(float(t), float(v)) for t, v in zip(times[i], voltages[i])]
        circuit.R(f'R_load_left_{i}', f'row_{i}_0', f'left_{i}', Rload)
        circuit.PieceWiseLinearVoltageSource(f'V{10+i}', f'left_{i}', circuit.gnd, values=pwl)
    
    for i in range(N):
        circuit.R(f'R_load_right_{i}', f'row_{i}_{M-1}', f'right_{i}', Ropen)
        circuit.V(f'V_right_{i}', f'right_{i}', circuit.gnd, 0)
    
    for j in range(M):
        circuit.R(f'R_load_top_{j}', f'col_0_{j}', f'top_{j}', Ropen)
        circuit.V(f'V_top_{j}', f'top_{j}', circuit.gnd, 0)
    
    for j in range(M):
        circuit.C(f'C{13+j}', f'col_{N-1}_{j}', circuit.gnd, C_gb)
    
    circuit.MOSFET('M66', 'drain_0', f'col_{N-1}_0', 'source_0', 'bulk', model='IHM_NMOS_HFOX')
    circuit.R('R_out_left', 'out_left', 'source_0', 1)
    circuit.V('V_out_left', 'out_left', circuit.gnd, 0)
    
    circuit.MOSFET('M67', 'drain_1', f'col_{N-1}_1', 'source_1', 'bulk', model='IHM_NMOS_HFOX')
    circuit.R('R10', 'drain_0', 'source_1', Rs)
    
    circuit.MOSFET('M68', 'drain_2', f'col_{N-1}_2', 'source_2', 'bulk', model='IHM_NMOS_HFOX')
    circuit.R('R9', 'drain_1', 'source_2', Rs)
    
    circuit.R('R_bulk', 'bulk', circuit.gnd, 0.1)
    
    circuit.MOSFET('M71', f'col_{N-1}_0', 'N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
    circuit.MOSFET('M72', f'col_{N-1}_1', 'N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
    circuit.MOSFET('M73', f'col_{N-1}_2', 'N006', circuit.gnd, circuit.gnd, model='IHM_NMOS_HFOX_RESET')
    
    circuit.R('Rout9', 'drain_2', 'N009', Rs)
    circuit.MOSFET('M69', 'N009', 'N009', 'vdd', 'vdd', model='IHM_PMOS2')
    circuit.MOSFET('M70', 'N006', 'N009', 'vdd', 'vdd', model='IHM_PMOS')
    circuit.R('Rout10', 'N006', circuit.gnd, Rout)
    
    circuit.raw_spice += get_subcircuit('MEMCAP')
    for i in range(N):
        circuit.raw_spice += f".save v(left_{i})\n"
    for j in range(M):
        circuit.raw_spice += f".save v(col_{N-1}_{j})\n"
    circuit.raw_spice += ".save v(N009)\n"
    circuit.raw_spice += ".save v(N006)\n"
    circuit.raw_spice += ".options plotwinsize=0 method=gear reltol=1e-4\n"
    circuit.raw_spice += ".ic V(N006)=0\n"
    
    return circuit, ic

def run_simulation(circuit, ic):
    sim = circuit.simulator(temperature=25, nominal_parameters=[])
    sim.initial_condition(**ic)
    return sim.transient(step_time=100e-9, end_time=total_time)

def extract_data(analysis):
    return {
        't': np.array(analysis.time),
        'V_in': [np.array(analysis[f'left_{i}']) for i in range(N)],
        'V_gate': [np.array(analysis[f'col_{N-1}_{j}']) for j in range(M)],
        'V_N009': np.array(analysis['n009']),
        'V_N006': np.array(analysis['n006']),
    }

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-SCALE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def auto_scale(data_list, margin=0.15):
    """
    Automatically calculate y-axis limits based on data.
    data_list: list of numpy arrays or single array
    margin: fraction of range to add as padding (default 15%)
    """
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    
    all_data = np.concatenate([d.flatten() for d in data_list])
    v_min = np.min(all_data)
    v_max = np.max(all_data)
    
    v_range = v_max - v_min
    if v_range < 0.01:  # If range is very small, set minimum range
        v_range = 0.1
    
    padding = v_range * margin
    
    y_min = v_min - padding
    y_max = v_max + padding
    
    # Don't go below 0 for voltage plots
    if v_min >= 0:
        y_min = max(-0.02, y_min)
    
    return y_min, y_max

# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTION - AUTO-SCALING
# ══════════════════════════════════════════════════════════════════════════════

def plot_window(cases, results, window_title, filename, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    case_names = list(cases.keys())
    n_cases = len(case_names)
    
    in_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    gate_colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    fig = plt.figure(figsize=(5 * n_cases + 1.5, 13))
    gs = GridSpec(5, n_cases, figure=fig, 
                  left=0.12, right=0.98, top=0.92, bottom=0.08,
                  hspace=0.25, wspace=0.3)
    
    fig.suptitle(window_title, fontsize=14, fontweight='bold')
    
    row_labels = ['Weight\nMatrix', 'Input\nVoltage (V)', 'Membrane\nVoltage (V)', 
                  'CM Input\nVoltage (V)', 'CM Output\nVoltage (V)']
    
    # Calculate global y-limits for each row (for consistent scaling across columns)
    all_inputs = []
    all_gates = []
    all_cm_in = []
    all_cm_out = []
    
    for name in case_names:
        data = results[name]
        all_inputs.extend(data['V_in'])
        all_gates.extend(data['V_gate'])
        all_cm_in.append(data['V_N009'])
        all_cm_out.append(data['V_N006'])
    
    input_ylim = auto_scale(all_inputs)
    gate_ylim = auto_scale(all_gates)
    cm_in_ylim = auto_scale(all_cm_in)
    cm_out_ylim = auto_scale(all_cm_out)
    
    for col, name in enumerate(case_names):
        x0 = cases[name]
        data = results[name]
        t_us = data['t'] * 1e6
        
        # Row 0: Weight Matrix
        ax = fig.add_subplot(gs[0, col])
        for i in range(N):
            for j in range(M):
                facecolor = 'white' if x0[i, j] < 0.20 else 'black'
                textcolor = 'black' if x0[i, j] < 0.20 else 'white'
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      facecolor=facecolor,
                                      edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
                ax.text(j, i, f'{x0[i, j]:.2f}',
                        ha='center', va='center',
                        color=textcolor, fontsize=9, fontweight='bold')
        ax.set_xlim(-0.5, M - 0.5)
        ax.set_ylim(N - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(name, fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=10, fontweight='bold',
                         labelpad=10, rotation=0, ha='right', va='center')
        
        # Row 1: Input Voltage - AUTO-SCALED
        ax = fig.add_subplot(gs[1, col])
        for i in range(N):
            ax.plot(t_us, data['V_in'][i], color=in_colors[i], lw=1)
        ax.set_ylim(input_ylim)
        ax.set_xlim(0, total_time * 1e6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=10, fontweight='bold',
                         labelpad=10, rotation=0, ha='right', va='center')
        
        # Row 2: Membrane Voltage - AUTO-SCALED
        ax = fig.add_subplot(gs[2, col])
        for j in range(M):
            ax.plot(t_us, data['V_gate'][j], color=gate_colors[j], lw=1.2)
        ax.axhline(NMOS_PARAMS['VTH0'], color='black', ls='--', lw=1.5)
        ax.set_ylim(gate_ylim)
        ax.set_xlim(0, total_time * 1e6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(row_labels[2], fontsize=10, fontweight='bold',
                         labelpad=10, rotation=0, ha='right', va='center')
        
        # Row 3: CM Input Voltage - AUTO-SCALED
        ax = fig.add_subplot(gs[3, col])
        ax.plot(t_us, data['V_N009'], '#16A085', lw=1.2)
        ax.set_ylim(cm_in_ylim)
        ax.set_xlim(0, total_time * 1e6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=10, fontweight='bold',
                         labelpad=10, rotation=0, ha='right', va='center')
        
        # Row 4: CM Output Voltage - AUTO-SCALED
        ax = fig.add_subplot(gs[4, col])
        ax.plot(t_us, data['V_N006'], '#8E44AD', lw=1.2)
        ax.axhline(RESET_PARAMS['VTH0'], color='red', ls='--', lw=1, alpha=0.7)
        ax.set_ylim(cm_out_ylim)
        ax.set_xlim(0, total_time * 1e6)
        ax.set_xlabel('Time (µs)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(row_labels[4], fontsize=10, fontweight='bold',
                         labelpad=10, rotation=0, ha='right', va='center')
    
    plt.savefig(f'{save_path}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("Neurotransistor WITH RESET - Validation Test")
    print(f"Input Voltage: {pulse_voltage} V")
    print("="*70)
    
    times, voltages = generate_pulses()
    
    # Window 1
    print("\nWINDOW 1: All ON/OFF Combinations")
    cases1 = get_window1_cases()
    results1 = {}
    for name, x0 in cases1.items():
        print(f"  Simulating: {name.replace(chr(10), ' + ')}")
        circuit, ic = build_circuit(x0, times, voltages)
        analysis = run_simulation(circuit, ic)
        results1[name] = extract_data(analysis)
    plot_window(cases1, results1, 'All ON/OFF Combinations', 'window1', save_path='results')
    
    # Window 2
    print("\nWINDOW 2: Partial Activation")
    cases2 = get_window2_cases()
    results2 = {}
    for name, x0 in cases2.items():
        print(f"  Simulating: {name.replace(chr(10), ' + ')}")
        circuit, ic = build_circuit(x0, times, voltages)
        analysis = run_simulation(circuit, ic)
        results2[name] = extract_data(analysis)
    plot_window(cases2, results2, 'Partial Activation', 'window2', save_path='results')
    
   
