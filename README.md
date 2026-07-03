# Neuro-Transistor Circuit Design and Simulation

This repository contains the simulation code for a neuro-transistor-based spiking neuron circuit for spiking neural networks. The project solves five main problems: mathematical modelling, NMOS parameter extraction, gate-node loading, reset after firing, and multi-layer spike propagation.

## Project Summary

The work implements and verifies:

- a mathematical model for the neuro-transistor using memristive conductance, membrane capacitance, and EKV transistor equations,
- NMOS parameter extraction from Id-Vg, Id-Vd, and C-V characteristics,
- a PMOS current mirror to isolate the high-impedance gate node,
- a reset circuit to discharge the membrane after a spike,
- MNIST-based pulse input testing,
- and a connected two-layer architecture where three input neurons drive a fourth output neuron.

## Results

### NMOS Model and C-V Verification

![Id Vg characteristics](images/slide09_image1.png)
![Id Vd characteristics](images/slide09_image2.png)
![CV characteristics](images/slide09_image3.png)

The NMOS model was verified using EKV-based equations and PySpice simulations. The extracted parameters include approximately `Vth = 0.277 V`, `kappa = 0.516`, and membrane capacitance around `100 pF`.

### Mathematical Model Validation

![Single synaptic LIF validation](images/slide10_image1.png)
![Three synaptic LIF validation](images/slide10_image2.png)

The analytical neuro-transistor model was compared with SPICE simulation for single-synaptic and three-synaptic input cases. The membrane voltage prediction follows the simulated LIF behavior closely.

### Current Mirror Circuit

![Current mirror circuit](images/slide11_image3.png)
![Current mirror simulation](images/slide11_image2.png)

The PMOS current mirror isolates the gate/membrane node from downstream loading. The PMOS width ratio gives a current gain of about `94/24 = 3.9`.

### Reset Circuit

![Reset circuit](images/slide12_image2.png)
![Reset circuit simulation](images/slide12_image1.png)

The reset circuit discharges the membrane after an output spike. When the output crosses the reset threshold, the gate voltage returns close to baseline, enabling repeated integrate-and-fire behavior.

### Weight Configuration Testing

![Weight testing ON OFF cases](images/slide13_image1.png)

Different ON/OFF memristor weight configurations were tested. The circuit produces spike and reset behavior across the tested cases, but strong `0.8 V` input pulses can still make weak/OFF cases fire.

### Fully Connected Two-Layer Network

![Layer 1 behavior](images/slide14_image1.png)
![Fully connected circuit](images/slide14_image2.png)
![Layer 2 behavior](images/slide15_image1.png)

Three input neuro-transistors process MNIST pulse trains and drive one output neuro-transistor. The output neuron integrates the incoming spikes and fires, confirming successful spike propagation between layers.

### Strong Input Feedforward Test

![Strong input feedforward behavior](images/slide16_image1.png)
![Strong input neuron behavior](images/slide16_image2.png)

The connected-neuron architecture was also tested with stronger input activity, showing how synchronized first-layer spikes affect the output neuron response.

## Main Conclusion

The project demonstrates that the neuro-transistor can be extended into a reusable spiking neuron circuit. The mathematical model predicts its behavior, the extracted NMOS parameters support accurate simulation, the current mirror solves the loading problem, the reset circuit enables repeated firing, and the connected-layer simulation shows that spikes can propagate through a small feedforward network.

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
- `neurocrossbar_functions.py`: crossbar circuit helper functions
- `current_mirror_circuit_14_mnsit.py`: current mirror simulation
- `reset_neurotransistor_14_mnist.py`: reset circuit simulation
- `reset_on_off_cases.py`: ON/OFF weight validation
- `fully_connected_mnist_reset_14.py`: connected two-layer network simulation
