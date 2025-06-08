# 🧠 TPU-Enabled Layer-Wise Runtime Quantization Without Accuracy Degradation

## 📌 Project Overview

This project introduces a **hardware-embedded runtime quantization framework** for **Tensor Processing Units (TPUs)** that performs **layer-wise quantization** dynamically, based on each layer’s real-time contribution to model accuracy. The framework considers **environmental context**, **energy constraints**, and **layer sensitivity**, and dynamically adjusts precision without compromising predictive performance.

## 🧩 TPU Architecture Summary

The system is inspired by **Google TPU v1** and built around a **systolic array of MAC (Multiply-Accumulate) units** using a **weight-stationary dataflow**. Key architectural elements:

- 🎯 **Systolic PE Array**:  
  2D mesh of processing elements that hold weights locally while streaming inputs and partial sums through the array.

- 📦 **On-Chip BRAM Buffers**:
  - `Weight Buffer`: Stores preloaded weights.
  - `Input Buffer`: Feeds input activations into the systolic array.
  - `Partial Sum Buffer`: Accumulates outputs before write-back.

- 🚀 **DMA-Controlled Pipelined Execution**:  
  A lightweight DMA engine facilitates efficient data movement between:
  - Host ↔ FPGA memory
  - External DDR ↔ TPU buffers  
  Enabling pipelined, low-latency execution.

## 🔍 Motivation

Modern deep learning accelerators often employ **static quantization**, which:
- Applies a uniform quantization policy across all layers.
- Ignores the fact that not all layers contribute equally to the model's final accuracy.
- Leads to unnecessary degradation in performance.

### ⚠️ Challenge:
> How can we **quantize only the less critical layers** during runtime to save energy **without sacrificing accuracy**?

---

## 🎯 Core Idea

At runtime, each layer `Lᵢ` is evaluated using a hardware-accelerated controller that uses:
- **Gradient sensitivity**
- **Sparsity metrics**
- **Latency/energy profiles**
- **Environment-driven priorities**

A utility score determines whether `Lᵢ` should remain in high precision or be quantized (e.g., from FP32 → INT8 or INT4). This score is computed **per forward pass**, allowing the system to adapt to runtime constraints **on the fly**.

## 🔁 End-to-End System Workflow

### **Step ①: Environment Modeling**  
Sensors yield an environmental context vector:  
**E = [α, λ, ε]**, where:
- **α** ∈ [0,1] — Accuracy demand  
- **λ** ∈ [0,1] — Latency sensitivity  
- **ε** ∈ [0,1] — Energy constraint  

This vector governs real-time quantization decisions.

---

### **Step ②: Layer-Wise Gradient & Sparsity Analysis**  
For each layer `l`:
- Compute the gradient of the loss: **∇W(l)L**
- Apply a threshold **τ** to obtain a sparse weight mask:

ΔW(l) = W(l) − W_threshold(l)
Where:
W_threshold(i,j) = 0 if |W(i,j)| < τ
W(i,j) otherwise

---

### **Step ③: Sensitivity Score Calculation**  
Calculate the element-wise product:

δ(l) = ΔW(l) ⊙ ∇W(l)L


Then compute the normalized sensitivity score:

Ŝ(l) = Σ |ΔW(i,j) × ∇W(i,j)L| / (‖W(l)‖_F × ‖∇W(l)L‖_F)
This gives a normalized sensitivity score per layer, where **‖·‖_F** is the Frobenius norm.

---

### **Step ④: Layer Impact Vector Construction**  
Each layer is assigned a vector:

Sᵢ = [gᵢ, lᵢ, eᵢ]

Where:
- **gᵢ**: Gradient impact on accuracy  
- **lᵢ**: Latency contribution (profiled)  
- **eᵢ**: Energy cost at full precision  

---

### **Step ⑤: Quantization Decision Function**  
Given the environment vector **E = [α, λ, ε]**, compute:

If:
Impactᵢ < θ

Then quantize layer **Lᵢ** to a lower bit-width at runtime using a hardware FSM controller. Precision is adjusted dynamically per inference based on this decision.

---



## ⚙️ Hardware Implementation

- ⏱️ **FSM Controller**: Tracks execution layer-by-layer.
- 🧠 **Quantization LUT**: Holds bit-width, latency, and energy costs.
- 🔁 **Runtime Registers**: Dynamically reprogram quantization.
- 🧮 **PE Array**: Performs MAC operations in the assigned precision.
- 🛜 **DMA Engine**: Handles high-throughput memory streaming.

---

## 💾 Software Interface (In Progress)

We are developing a host software stack that:
- Loads model weights into **TPU BRAMs**
- Streams activations into **input buffers**
- Sends layer metadata for quantization control
- Coordinates execution through **DMA** and **control registers**

---

## 📈 Roadmap

| Milestone                                    | Status       |
|---------------------------------------------|--------------|
| Layer-wise gradient/sparsity analyzer       | ✅ Complete   |
| Quantization control logic (simulated)      | ✅ Complete   |
| Verilog RTL for runtime quantization        | 🚧 In Progress |
| Software to deploy instruction and memory   | 🚧 In Progress |
| PyTorch export with layer quant tags        | 🔜 Planned    |
| Full FPGA prototyping                       | 🔜 Planned    |

---

## 🔬 Applications

- 🚗 Autonomous driving: Context-aware quantization
- 📱 Mobile inference: Power-aware runtime adaptation
- 🛰 Edge AI: Dynamic precision scaling for efficiency

---

## 💬 Discussion Topics

- How often should quantization be updated during inference?
- Should precision scale in fixed steps (e.g. FP32 → INT8) or finer granularity?
- How can we best estimate real-time energy per layer?

---

## 🤝 Contributors Welcome

We invite help with:
- Verilog/SystemVerilog for FSM and datapath
- Compiler metadata support for per-layer tags
- PyTorch QAT (quantization-aware training) tooling
- DMA integration for FPGA-host pipelines

---

## 📚 References

- Google TPUv1 Whitepaper  
- QAT for Edge Accelerators  
- Dynamic Precision Inference Strategies  
- Systolic Array Design for Deep Learning

---

## 📬 Let’s Collaborate

> Open an issue or start a discussion to get involved in redefining runtime-efficient deep learning inference.
