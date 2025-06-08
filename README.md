# 🧠 TPU-Enabled Layer-Wise Runtime Quantization Without Accuracy Degradation

## 📌 Project Synopsis

This project presents a **novel hardware-integrated framework** that performs **layer-wise quantization dynamically at runtime** within a **Tensor Processing Unit (TPU)**. The system intelligently determines the quantization bit-width for each neural network layer based on its real-time **contribution to model accuracy**, combined with the **environmental context** and **resource constraints**. The primary goal is to enable **energy-efficient inference** without compromising predictive performance.

---

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

---

## 🔁 End-to-End System Workflow

### **Step ①: Environment Modeling**
Sensors yield an environment vector **𝐄 = [α, λ, ε]**, where:
- `α ∈ [0,1]`: Accuracy demand
- `λ ∈ [0,1]`: Latency sensitivity
- `ε ∈ [0,1]`: Energy constraint level

---

### **Step ②: Layer-Wise Gradient & Sparsity Analysis**
For each layer `l`:
- Compute the gradient of the loss: ∇W(l)L
- Apply sparsity thresholding τ to obtain sparse mask ΔW(l):  
  \[
  ΔW^{(l)} = W^{(l)} - W^{(l)}_{\text{threshold}}
  \]

---

### **Step ③: Sensitivity Score Calculation**
Determine each layer’s relevance using:
\[
δ^{(l)} = ΔW^{(l)} \circ ∇W^{(l)}L
\]
Then normalize:
\[
\hat{S}^{(l)} = \frac{\sum_{i,j} |δ^{(l)}_{i,j}|}{\|W^{(l)}\|_F \cdot \|\nabla W^{(l)}L\|_F}
\]

---

### **Step ④: Layer Impact Vector Construction**
Each layer is tagged with:
\[
\mathbf{S}_i = [g_i, l_i, e_i]
\]
- `gᵢ`: sensitivity to gradient
- `lᵢ`: latency contribution (pre-profiled)
- `eᵢ`: energy cost at full precision

---

### **Step ⑤: Quantization Decision Function**
Given the environment vector 𝐄 = [α, λ, ε], compute:
\[
\text{Impact}_i = α \cdot g_i - λ \cdot l_i - ε \cdot e_i
\]

If `Impactᵢ` < threshold `θ`, then **quantize layer `i`** to a lower bit-width dynamically using on-chip control FSMs.

---

## ⚙️ Hardware Integration Strategy

- ⏱️ **FSM Controller** tracks layer execution in real time.
- 📉 **Quantization LUT** holds latency & energy cost per bit-width.
- 🧮 **Runtime Evaluator** computes utility scores during inference.
- 🔄 **Bit-Width Register File** reprograms quantization modules on-the-fly.

---

## 🔬 Use Cases

- 🚗 Autonomous Systems: Terrain and speed-based bit-width selection
- 🛰️ Drones: Runtime adaptation based on battery and mission phase
- 📱 Mobile AI: On-device quantization for power-aware inference

---

## 📈 Roadmap

| Milestone | Status |
|----------|--------|
| ✅ Per-layer sensitivity computation | Done |
| ✅ Runtime quantization control logic (simulated) | Done |
| 🚧 Hardware RTL for quantization controller | In Progress |
| 🔜 End-to-end TPU simulation (Verilog) | Planned |
| 🔜 Integration with PyTorch model export | Planned |
| 🔜 Multi-bit reconfigurable quantization LUT | Planned |

---

## 💬 Discussion Topics

- Should quantization bit-widths be discrete (INT8, INT4) or continuous?
- What’s the best frequency to re-evaluate layer utility? Every frame? Batch?
- Can this method extend to activation quantization dynamically?
- Could reinforcement learning help improve the mode decision policy?

---

## 🤝 Looking for Contributors

We're seeking contributors with expertise in:
- 🛠️ Verilog/SystemVerilog for RTL logic
- 📐 Architecture modeling (TPU, systolic arrays)
- 📊 Compiler metadata injection (e.g., per-layer profiling tags)
- 🧪 PyTorch and quantization-aware training (QAT)

---

## 📚 Reference Resources

- *On-Device Dynamic Quantization for Efficient AI*  
- *Sparsity-Aware Gradient Masking in Edge Inference*  
- *TPU Architecture Whitepapers (Google, 2018–2023)*

---

## 🧵 Let’s Collaborate

> Open an issue or discussion to contribute results, questions, or ideas.

---
