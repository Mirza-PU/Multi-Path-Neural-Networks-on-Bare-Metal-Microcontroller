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

----
## 📐 Mathematical Formulation
1. Multi-Path Forward Pass
Given input tensor x 
t
​
 ∈R 
d
 , the shared feature representation is computed as:

h 
shared
​
 =σ(W 
shared
​
 x 
t
​
 +b 
shared
​
 )
The dynamic execution branches are defined by:

Small Path (P 
small
​
 ): Direct linear transformation

P 
small
​
 (x 
t
​
 )=W 
out
(1)
​
 h 
shared
​
 +b 
out
(1)
​
 
Medium Path (P 
medium
​
 ): Single hidden layer non-linear network

P 
medium
​
 (x 
t
​
 )=W 
out
(2)
​
 σ(W 
mid
​
 h 
shared
​
 +b 
mid
​
 )+b 
out
(2)
​
 
Large Path (P 
large
​
 ): Dual hidden layer non-linear network

P 
large
​
 (x 
t
​
 )=W 
out
(3)
​
 σ(W 
l2
​
 σ(W 
l1
​
 h 
shared
​
 +b 
l1
​
 )+b 
l2
​
 )+b 
out
(3)
​
 
2. Dual-Trigger Telemetry Gating
The routing state decision G(t)∈{SMALL,MEDIUM,LARGE} is evaluated deterministically at each step:

G(t)= 
⎩

⎨

⎧
​
  
SMALL,
LARGE,
MEDIUM,
​
  
if M 
free
​
 (t)<150 KB
if M 
free
​
 (t)≥150 KB and ∣ΔT(t)∣>2.0 
∘
 C
otherwise
​
 
## 📦 Repository Structure
Plaintext


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
🛠️ Hardware Requirements & Setup
Components
Microcontroller: ESP32-S3 DevKitC-1 (240 MHz Xtensa LX7, 512 KB SRAM, 8 MB Flash)

Sensor: DHT22 Digital Temperature & Humidity Sensor

Display: 0.96" SSD1306 OLED Display (128×64, I2C interface)

Pin Mapping
Component	Pin Function	ESP32-S3 GPIO
DHT22	Data Line	GPIO 18
SSD1306	SDA	GPIO 21
SSD1306	SCL	GPIO 22
Power	VCC / GND	3.3V / GND

🚀 Quick Start Guide
1. Model Training & Export (Python)
To train the multi-path network using PyTorch and generate the static C++ header model_weights.h:

Bash


# Clone repository
git clone [https://github.com/your-username/resource-adaptive-edge-inference.git](https://github.com/your-username/resource-adaptive-edge-inference.git)
cd resource-adaptive-edge-inference/training

# Install dependencies
pip install torch numpy matplotlib

# Train the multi-branch model & generate C++ header
python train_multipath.py --export-path ../firmware/src/model_weights.h
2. Embedded Firmware Deployment (PlatformIO)
Install PlatformIO IDE (VS Code extension or CLI).

Connect your ESP32-S3 board via USB.

Build and flash the firmware:

Bash


cd ../firmware

# Build project
pio run

# Flash to ESP32-S3
pio run --target upload

# Open Serial Monitor for microsecond latency profiling logs
pio device monitor --baud 115200
📊 Benchmark Summary
Path	Layers	Parameters	Flash Size	Microsecond Latency	Dynamic Memory
P 
small
​
 	2	19	76 Bytes	12.0μs	0 Bytes
P 
medium
​
 	3	59	236 Bytes	28.0μs	0 Bytes
P 
large
​
 	4	179	716 Bytes	65.0μs	0 Bytes

## 👥 Authors & Affiliations
Mirza Mudassar Hussain — Institute of Mathematics, University of the Punjab, Lahore, Pakistan (muddasser.mh@gmail.com)

Muhammad Nasim Aftab — Department of Mathematics, University of Engineering and Technology, Lahore, Pakistan (nasim.aftab@uet.edu.pk)

Apostolos Xenakis — Department of Digital Systems, University of Thessaly, Larissa, Greece (axenakis@uth.gr)

George Floros (Corresponding Author) — Department of Electronic and Electrical Engineering, Trinity College Dublin, Ireland (florrosg@tcd.ie)

## ✍️ Citation
If you use this work, framework, or code in your research, please cite our manuscript:

Code snippet


@article{hussain2026resource,
  title={Resource-Adaptive Edge Inference via Telemetry-Driven Multi-Path Neural Networks on Bare-Metal Microcontrollers},
  author={Hussain, Mirza Mudassar and Aftab, Muhammad Nasim and Xenakis, Apostolos and Floros, George},
  journal={AMS Mathematics/Computer Science Repository},
  year={2026}
}
## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
