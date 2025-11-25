# Selective-SSM
A reimplementation of a selective state space model.

<img width="8296" height="3093" alt="image" src="https://github.com/user-attachments/assets/c010fb65-3cb9-4520-8b46-d9bf28660932" />

> Mamba: Linear-Time Sequence Modeling with Selective State Spaces
> Albert Gu*, Tri Dao*
> Paper: https://arxiv.org/abs/2312.00752


# Reimplementation Roadmap


## Phase 1: Core SSM Implementation

### Standard S4 Layer (Non-selective baseline)
- [x] Implement discretization formulas (Equation 4 - ZOH)
- [x] Implement the recurrent form (Equation 2a, 2b)
- [x] Implement the convolutional form (Equation 3a, 3b) using FFT
- [x] Test equivalence between recurrent and convolutional modes
- [x] Validate on simple copying task

### Selection Mechanism (Algorithm 2)
- [x] Implement input-dependent Δ with Linear projection and softplus
- [x] Implement selective B and C parameters
- [x] Add broadcasting operations for Δ across D channels
- [x] Verify Theorem 1 (connection to RNN gating) numerically


## Phase 2: Hardware-Aware Parallel Scan

### Parallel Scan Algorithm
- [x] Implement basic associative scan in PyTorch (naive version)
- [x] Study parallel scan algorithms (Blelloch 1990 reference)
- [x] Implement work-efficient parallel scan
- [x] Test correctness against recurrent computation

### Kernel Fusion (Critical for efficiency)
- [ ] Write custom CUDA kernel for fused discretization + scan + output projection
- [ ] Implement memory-efficient scan (HBM → SRAM data flow)
- [ ] Add recomputation strategy for backward pass
- [ ] Benchmark against naive PyTorch implementation (target: 20-40× speedup)


## Phase 3: Mamba Block Architecture

### Block Components
- [x] Implement input projection (expansion to ED dimensions, E=2)
- [x] Add 1D convolution layer (before SSM)
- [x] Implement SSM layer with selection mechanism
- [x] Add SiLU/Swish activation function
- [x] Implement output projection (contract back to D dimensions)
- [x] Add optional normalization layer

### Full Mamba Block (Figure 3)
- [x] Combine all components following the architecture diagram
- [x] Implement residual connections
- [x] Add RMSNorm between blocks
- [x] Test single block forward/backward pass


## Phase 4: Complete Model & Training

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


## Phase 5: Validation & Testing

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
