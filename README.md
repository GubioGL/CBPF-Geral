# Hybrid quantum physics-informed neural networks with a fixed quantum layer for forward and inverse problems

## Abstract

We study a physics-informed neural network whose middle block is a fixed quantum circuit with no trainable parameters
(HQPINN), on the forward and inverse problems of the viscous Burgers equation, u_t + u u_x = ν u_xx with ν = 0.01. In the
forward problem, with one common protocol and noiseless simulation, the hybrid model reaches the error of a classical
network of the same form with about half of the optimizer steps and fewer parameters at the short horizon, and a lower
error at the long horizon. Trained with a 120-qubit superconducting processor (`ibm_berlin`) in the loop, with one job
per epoch and no error mitigation, it reaches a relative error of 5.7 % in 110 epochs at the short horizon. In the
inverse problem, where the viscosity is learned together with the solution, both models identify ν at the short
horizon on a noisy emulator of the device, and with a long budget and the noiseless layer both identify it at both
horizons. Since the noiseless layer is classically computable, the simulations measure the merit of an architecture,
not a quantum advantage.

<!-- Figure: replace the line below with the image, e.g. ![HQPINN architecture](docs/figure.png) -->
<p align="center"><i>Figure: to be added.</i></p>

## Reference

G. G. de Lima *et al.*, *Hybrid quantum physics-informed neural networks with a fixed quantum layer for forward and
inverse problems*, in preparation (2026).

## Repository structure

```text
CBPF-Geral/
├── src/                 library: PINN models, quantum layer, physics, training loops, hardware machinery
├── run/                 entry points: training runs (forward and inverse), QPU and emulator runs, plots
├── data/                raw results (see data/README.md)
│   ├── Forward/                         forward problem, simulations
│   ├── quantum_forward_hardware_real/   forward problem on ibm_berlin
│   ├── Inverso_200/                     inverse problem, 200 epochs, HQPINN on FakeAachen
│   └── Inverso/                         inverse problem, 20 000 epochs, ideal circuit
├── docs/                notes on the results
├── requirements.txt     dependencies (requirements-lock.txt: exact versions)
└── README.md
```
