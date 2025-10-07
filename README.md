# Selective-SSM
A reimplementation of a selective state space model.

<img width="8296" height="3093" alt="image" src="https://github.com/user-attachments/assets/c010fb65-3cb9-4520-8b46-d9bf28660932" />

> Mamba: Linear-Time Sequence Modeling with Selective State Spaces  
> Albert Gu*, Tri Dao*  
> Paper: https://arxiv.org/abs/2312.00752  

# Reimplementation Roadmap


## Phase 1: Foundations & Core SSM Understanding 

### Theory & Background
- [ ] Read and understand Section 2 (State Space Models) thoroughly
- [ ] Understand continuous-time SSM equations (1a, 1b) and their discretization
- [ ] Study the difference between LTI (Linear Time Invariant) and selective SSMs
- [ ] Review the motivation for selection mechanism (Section 3.1)

### Basic Infrastructure Setup
- [ ] Set up development environment (PyTorch, CUDA toolkit)
- [ ] Create project structure with proper testing framework
- [ ] Implement basic utilities (parameter initialization, layer normalization)
- [ ] Set up synthetic tasks (Selective Copying, Induction Heads) for validation

---

## Phase 2: Core SSM Implementation 

### Standard S4 Layer (Non-selective baseline)
- [ ] Implement discretization formulas (Equation 4 - ZOH)
- [ ] Implement the recurrent form (Equation 2a, 2b)
- [ ] Implement the convolutional form (Equation 3a, 3b) using FFT
- [ ] Test equivalence between recurrent and convolutional modes
- [ ] Validate on simple copying task

### Selection Mechanism (Algorithm 2)
- [ ] Implement input-dependent Δ with Linear projection and softplus
- [ ] Implement selective B and C parameters
- [ ] Add broadcasting operations for Δ across D channels
- [ ] Verify Theorem 1 (connection to RNN gating) numerically

---

## Phase 3: Hardware-Aware Parallel Scan 

### Parallel Scan Algorithm
- [ ] Implement basic associative scan in PyTorch (naive version)
- [ ] Study parallel scan algorithms (Blelloch 1990 reference)
- [ ] Implement work-efficient parallel scan
- [ ] Test correctness against recurrent computation

### Kernel Fusion (Critical for efficiency)
- [ ] Write custom CUDA kernel for fused discretization + scan + output projection
- [ ] Implement memory-efficient scan (HBM → SRAM data flow)
- [ ] Add recomputation strategy for backward pass
- [ ] Benchmark against naive PyTorch implementation (target: 20-40× speedup)

**Note:** If CUDA experience is limited, consider using Triton for kernel implementation as an alternative.

---

## Phase 4: Mamba Block Architecture 

### Block Components
- [ ] Implement input projection (expansion to ED dimensions, E=2)
- [ ] Add 1D convolution layer (before SSM)
- [ ] Implement SSM layer with selection mechanism
- [ ] Add SiLU/Swish activation function
- [ ] Implement output projection (contract back to D dimensions)
- [ ] Add optional normalization layer

### Full Mamba Block (Figure 3)
- [ ] Combine all components following the architecture diagram
- [ ] Implement residual connections
- [ ] Add RMSNorm between blocks
- [ ] Test single block forward/backward pass

---

## Phase 5: Complete Model & Training

### Model Assembly
- [ ] Stack multiple Mamba blocks to create full model
- [ ] Implement embedding layer and output head
- [ ] Add proper weight initialization (S4D-Real or S4D-Lin)
- [ ] Configure model sizes following Table 12 specifications

### Training Pipeline
- [ ] Implement AdamW optimizer with proper hyperparameters
- [ ] Add learning rate scheduler (warmup + cosine decay)
- [ ] Implement gradient clipping (value 1.0)
- [ ] Add weight decay (0.1) without applying to biases/norms

---

## Phase 6: Validation & Testing

### Synthetic Task Validation
- [ ] Test on Selective Copying task (Table 1 target: ~99% accuracy)
- [ ] Test on Induction Heads (Table 2 target: perfect extrapolation)
- [ ] Verify selection mechanism is working (compare S4 vs S6)

### Small-Scale Language Modeling
- [ ] Set up training on small dataset (e.g., WikiText-103)
- [ ] Train 125M parameter model
- [ ] Compare perplexity against baseline Transformer
- [ ] Verify training stability and convergence

### Performance Benchmarks
- [ ] Measure training speed (tokens/second)
- [ ] Measure inference throughput vs Transformer
- [ ] Profile memory usage
- [ ] Compare against reported benchmarks (Figure 8)
