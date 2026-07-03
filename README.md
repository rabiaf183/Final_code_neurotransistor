# Neuro-Transistor Circuit Design and Simulation

This repository contains simulation code for a neuro-transistor-based spiking neuron circuit. The project develops a mathematical model, verifies device parameters, adds circuit improvements, and demonstrates multi-layer spike propagation.

## What This Project Solves

- Developed a mathematical model for the neuro-transistor using memristive conductance, membrane capacitance, and EKV transistor equations.
- Verified NMOS behavior using Id-Vg, Id-Vd, and C-V characteristics.
- Added a PMOS current mirror to isolate the high-impedance gate node.
- Implemented a reset circuit so the membrane voltage discharges after firing.
- Tested different ON/OFF memristor weight configurations.
- Built a connected two-layer network where three input neurons drive a fourth output neuron.

## Results

### NMOS and C-V Parameter Extraction

![NMOS and CV validation](images/slide-09.png)

The NMOS model was validated using Id-Vg, Id-Vd, and C-V characteristics. Extracted parameters include approximately `Vth = 0.277 V`, `kappa = 0.516`, and `C = 100 pF`.

### Mathematical Model Validation

![Mathematical model validation](images/slide-10.png)

The analytical neuro-transistor model was compared with SPICE simulations for single-synaptic and three-synaptic input cases. The model closely follows the simulated LIF membrane behavior.

### Current Mirror Circuit

![Current mirror circuit](images/slide-11.png)

The PMOS current mirror isolates the gate/membrane node from downstream loading. The PMOS width ratio gives a current gain of about `94/24 = 3.9`.

### Reset Circuit

![Reset circuit](images/slide-12.png)

The reset circuit discharges the membrane after a spike. This allows the neuro-transistor to return close to baseline and fire repeatedly.

### Weight Configuration Testing

![Weight configuration testing](images/slide-13.png)

Different ON/OFF memristor weight states were tested. The circuit shows spike and reset behavior across weight configurations, though strong input pulses can still trigger firing in weak/OFF cases.

### Fully Connected Network: Layer 1

![Fully connected layer 1](images/slide-14.png)

Three input neuro-transistors process MNIST-based pulse inputs independently and generate output spikes for the next layer.

### Fully Connected Network: Layer 2

![Fully connected layer 2](images/slide-15.png)

The output neuron integrates spikes from the three input neurons and fires, showing successful spike propagation between connected layers.

### Strong Input Feedforward Test

![Feedforward connected neurons](images/slide-16.png)

The connected-neuron architecture was also tested with stronger input activity to observe how synchronized first-layer spikes affect the output neuron response.

## Conclusion

The project shows that the neuro-transistor can be modelled, simulated, reset after firing, and connected into a small feedforward spiking network. The current mirror solves gate-node loading, the reset circuit enables repeated firing, and the connected-layer simulations demonstrate multi-neuron spike propagation.

## How To Run

Install the required packages:

```bash
python3 -m pip install numpy scipy matplotlib PySpice
```

PySpice also requires ngspice.

Run the main simulations:

```bash
python3 reset_on_off_cases.py
python3 current_mirror_circuit_14_mnsit.py
python3 reset_neurotransistor_14_mnist.py
python3 fully_connected_mnist_reset_14.py
```

For MNIST simulations, place the MNIST files in:

```text
raw/
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
```

## Main Files

- `memcap_model.py`: memristor/memcapacitor SPICE model
- `MNIST.py`: MNIST loading and pulse-train generation
- `neurocrossbar_functions.py`: crossbar helper functions
- `current_mirror_circuit_14_mnsit.py`: current mirror simulation
- `reset_neurotransistor_14_mnist.py`: reset circuit simulation
- `reset_on_off_cases.py`: ON/OFF validation cases
- `fully_connected_mnist_reset_14.py`: connected two-layer network simulation
