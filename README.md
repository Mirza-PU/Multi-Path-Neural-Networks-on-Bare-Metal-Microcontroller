# Resource-Adaptive Edge Inference via Telemetry-Driven Multi-Path Neural Networks on Bare-Metal Microcontrollers

[![Board](https://img.shields.io/badge/Board-ESP32--S3-red.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![Framework](https://img.shields.io/badge/Framework-Bare--Metal_C%2B%2B-blue.svg)]()
[![Language](https://img.shields.io/badge/Language-Python_3.10_%7C_C%2B%2B17-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and deployment suite for the research article:  
**"Resource-Adaptive Edge Inference via Telemetry-Driven Multi-Path Neural Networks on Bare-Metal Microcontrollers"**.

---

## 📌 Abstract

Edge AI applications running on sub-milliwatt microcontrollers face severe constraints regarding latency, memory, and prediction accuracy. Traditional TinyML frameworks deploy fixed execution graphs that cannot adapt to dynamic runtime environments—leading to Out-Of-Memory (OOM) crashes under high system memory loads or wasted clock cycles during static environmental conditions.

This repository presents a resource-aware Edge AI framework designed for bare-metal deployment on the **ESP32-S3**. The architecture features a multi-path neural network with a shared representation backbone and three specialized branches ($\mathcal{P}_{\text{small}}$, $\mathcal{P}_{\text{medium}}$, $\mathcal{P}_{\text{large}}$). A closed-loop, deterministic gating engine evaluates system telemetry ($\text{Free Heap RAM}$) alongside physical signal volatility ($\vert{}\Delta T\vert{}$) to dynamically route inferences at runtime.

### Key Highlights
- **Zero Dynamic Memory Overhead:** All neural network parameters are exported into static C++ structures (`model_weights.h`), eliminating dynamic memory allocations (`malloc`) and runtime fragmentation.
- **Deterministic Microsecond Latency:** Measured inference execution times on ESP32-S3 (240 MHz):
  - $\mathcal{P}_{\text{small}}$ (2 layers, 19 params): **$12\,\mu\text{s}$**
  - $\mathcal{P}_{\text{medium}}$ (3 layers, 59 params): **$28\,\mu\text{s}$**
  - $\mathcal{P}_{\text{large}}$ (4 layers, 179 params): **$65\,\mu\text{s}$**
- **System Stability Guarantee:** Prevents OOM heap exceptions under memory pressure while automatically scaling capacity during physical signal spikes.

---

## 🏛️ System Architecture

```text
                       +-------------------------------+
                       |    Sensor Telemetry Vector    |
                       |    Input: Temp / Delta Temp   |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |   Shared Backbone Layer       |
                       |   h_shared = ReLU(W*x + b)    |
                       +---------------+---------------+
                                       |
                   +-------------------+-------------------+
                   | Telemetry-Driven Gating Engine        |
                   | Rules:                                |
                   |  1. Free RAM < 150 KB  => P_small     |
                   |  2. |Delta T| > 2.0°C  => P_large     |
                   |  3. Otherwise          => P_medium    |
                   +-------------------+-------------------+
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
         v                             v                             v
+------------------+         +-------------------+         +-------------------+
|  Path: P_small   |         |   Path: P_medium  |         |   Path: P_large   |
|  2 Layers        |         |   3 Layers        |         |   4 Layers        |
|  Latency: 12 us  |         |   Latency: 28 us  |         |   Latency: 65 us  |
+--------+---------+         +---------+---------+         +---------+---------+
         |                             |                             |
         +-----------------------------+-----------------------------+
                                       |
                                       v
                       +-------------------------------+
                       |       Prediction Output       |
                       |      OLED Display / Serial    |
                       +-------------------------------+

---

## 📐 Mathematical Formulation

### 1. Multi-Path Forward Pass
Given input tensor $\mathbf{x}_t \in \mathbb{R}^d$, the shared feature representation is computed as:
$$\mathbf{h}_{\text{shared}} = \sigma\left(\mathbf{W}_{\text{shared}} \mathbf{x}_t + \mathbf{b}_{\text{shared}}\right)$$

The dynamic execution branches are defined by:
- **Small Path ($\mathcal{P}_{\text{small}}$):** Direct linear transformation
  $$\mathcal{P}_{\text{small}}(\mathbf{x}_t) = \mathbf{W}_{\text{out}}^{(1)} \mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{out}}^{(1)}$$
- **Medium Path ($\mathcal{P}_{\text{medium}}$):** Single hidden layer non-linear network
  $$\mathcal{P}_{\text{medium}}(\mathbf{x}_t) = \mathbf{W}_{\text{out}}^{(2)} \sigma\left(\mathbf{W}_{\text{mid}} \mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{mid}}\right) + \mathbf{b}_{\text{out}}^{(2)}$$
- **Large Path ($\mathcal{P}_{\text{large}}$):** Dual hidden layer non-linear network
  $$\mathcal{P}_{\text{large}}(\mathbf{x}_t) = \mathbf{W}_{\text{out}}^{(3)} \sigma\left(\mathbf{W}_{\text{l2}} \sigma\left(\mathbf{W}_{\text{l1}} \mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{l1}}\right) + \mathbf{b}_{\text{l2}}\right) + \mathbf{b}_{\text{out}}^{(3)}$$

### 2. Dual-Trigger Telemetry Gating
The routing state decision $\mathcal{G}(t) \in \{\text{SMALL}, \text{MEDIUM}, \text{LARGE}\}$ is evaluated deterministically at each step:

$$\mathcal{G}(t) = \begin{cases} \text{SMALL}, & \text{if } M_{\text{free}}(t) < 150\text{~KB} \\ \text{LARGE}, & \text{if } M_{\text{free}}(t) \ge 150\text{~KB} \text{ and } \vert{}\Delta T(t)\vert{} > 2.0^\circ\text{C} \\ \text{MEDIUM}, & \text{otherwise} \end{cases}$$

---

## 📦 Repository Structure

```text
.
├── firmware/
│   ├── src/
│   │   ├── main.cpp             # ESP32-S3 Bare-metal firmware entry point
│   │   └── model_weights.h      # Exported static weights & model parameters
│   └── platformio.ini           # PlatformIO hardware configuration
├── training/
│   ├── train_multipath.py       # PyTorch joint multi-task training script
│   └── export_header.py         # PyTorch-to-C++ header exporter
├── docs/
│   ├── System_architecture.png  # System block diagram
│   ├── latency_chart.png        # Latency evaluation chart
│   └── trace_simulation.png     # Telemetry routing simulation trace
├── LICENSE
└── README.md                    # Project documentation
