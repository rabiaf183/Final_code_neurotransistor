# Current-Mirror-Based Neuro-Transistor Circuits

This repository contains the simulation code for a current-mirror-based neuro-transistor circuit for spiking neural networks. The work focuses on improving the neuro-transistor so it can operate as a reusable integrate-and-fire neuron and pass spikes to another layer without disturbing the membrane/gate node.

The circuit combines a memristive crossbar, membrane capacitance, stacked NMOS output transistors, a PMOS current mirror, and a threshold-triggered reset circuit.

## What This Project Does

The code simulates:

- memristive synapses using a compact memcapacitor/memristor model,
- membrane voltage integration at the neuro-transistor gate,
- output spike generation through NMOS transistors,
- current-mirror-based output isolation,
- reset behavior after firing,
- MNIST-based pulse inputs,
- and a small two-layer fully connected spiking network.

## Key Results

### Reset circuit result

![Reset circuit results](readme-images/reset_circuit_results.png)

The reset circuit successfully turns the neuro-transistor into a reusable integrate-and-fire neuron. When the output crosses the reset threshold of 0.6 V, the membrane voltage is discharged back close to 0.05 V.

### ON/OFF weight validation

![Extreme ON/OFF cases](readme-images/on_off_extreme_cases.png)

Different ON/OFF memristor weight configurations were tested. The circuit produces output spikes and reset behavior across the tested cases, showing that the current mirror and reset path work together.

### Partial ON/OFF validation

![Partial ON/OFF cases](readme-images/on_off_partial_cases.png)

Partial weight patterns show that different programmed weights affect the membrane voltage traces. A limitation is that with 0.8 V input pulses, even weak/OFF cases can still cross threshold.

### Layer 1 fully connected network result

![Layer 1 results](readme-images/layer1_results.png)

Three input neuro-transistors process MNIST digit pulse trains independently. Their outputs reach about 0.93-0.94 V and are strong enough to drive the next layer.

### Layer 2 fully connected network result

![Layer 2 results](readme-images/layer2_results.png)

The output neuron integrates spikes from the three Layer-1 neurons and fires at about 75 us. This confirms that spikes can propagate through a two-layer neuro-transistor network.

## Main Conclusion

The current mirror isolates the high-impedance membrane/gate node from downstream loading, while the reset circuit restores the membrane after firing. Together, they allow the neuro-transistor to behave like a reusable spiking neuron and support multi-layer spike propagation.

## How To Run

Install the required Python packages:

```bash
python3 -m pip install numpy scipy matplotlib PySpice
