# Stability-Preserving Adaptive EEG Encoder

This document gives the layer-level specification of the **Stability-Preserving Adaptive EEG Encoder** used in the THINGS-EEG experiments.

The implementation class is:

```python
EEGProjectLayerLiteAdapterSafeV32
```

The specification below follows the released implementation rather than a conceptual approximation.

---

## 1. Input and output

For THINGS-EEG:

```text
Number of EEG channels: C = 17
Number of time samples: T = 250
Embedding dimension:    D = 1024
```

Therefore:

```text
Input:  x in R^(B x 17 x 250)
Output: z in R^(B x 1024)
```

The implementation also accepts an input ordered as `[B, 250, 17]`; in that case, the channel and time axes are transposed internally.

The flattened input dimension is:

```text
17 x 250 = 4250
```

---

## 2. Architecture summary

For the THINGS-EEG configuration, the encoder is composed of the following blocks:

| Block | Layer / operation | Dimensions |
|---|---|---:|
| Input calibration | Learnable channel scale | `1 x 17 x 1` |
| Temporal calibration | Linear | `250 -> 31` |
|  | GELU | `31` |
|  | Linear | `31 -> 250` |
| Core projector | Flatten | `17 x 250 -> 4250` |
|  | Linear | `4250 -> 1024` |
| Core residual block | GELU | `1024` |
|  | Linear | `1024 -> 1024` |
|  | Dropout | `p = 0.20` |
|  | Residual addition | `1024` |
| Core normalization | LayerNorm | `1024` |
| Residual adapter | Linear | `1024 -> 128` |
|  | GELU | `128` |
|  | Linear | `128 -> 1024` |
|  | Dropout | `p = 0.10` |
| Sample-wise residual gate | Linear | `1024 -> 128` |
|  | GELU | `128` |
|  | Linear | `128 -> 1` |
| Feature reweighting | Linear | `1024 -> 256` |
|  | GELU | `256` |
|  | Linear | `256 -> 1024` |
| Output | Adaptive residual + feature reweighting | `1024` |

The total number of trainable parameters for this THINGS-EEG configuration is approximately:

```text
6.341 M
```

The exact count from the layer specification is approximately **6.3407 M**, consistent with the rounded value reported in the computational-efficiency experiment.

---

## 3. Input calibration

### 3.1 Channel scaling

A learnable scale parameter is defined for each EEG channel:

```text
channel_scale shape = [1, 17, 1]
```

It is initialized to one, so the initial transformation is an identity mapping:

\[
x'_{b,c,t}=x_{b,c,t}s_c ,
\]

with

\[
s_c=1
\]

at initialization.

This design allows channel amplitudes to adapt during training without perturbing the original input representation at the start of optimization.

---

### 3.2 Temporal modulation

After channel scaling, the encoder computes a channel-averaged temporal context:

\[
u_{b,t}=\frac{1}{C}\sum_{c=1}^{C}x'_{b,c,t}.
\]

The temporal context is passed through a two-layer MLP:

```text
Linear(250, 31)
GELU
Linear(31, 250)
```

where:

```text
time_hidden = max(8, T // 8) = 31
```

The last linear layer is initialized to zero.

The temporal modulation is:

\[
\Delta_t=\tanh(f_{\mathrm{temp}}(u)),
\]

and the calibrated EEG signal is:

\[
\tilde{x}_{b,c,t}
=
x'_{b,c,t}
\left(1+\lambda_t\Delta_{b,t}\right),
\]

where the initial temporal scale is:

```text
lambda_t = 0.14
```

Because the final temporal-gate layer is zero-initialized, the temporal branch initially leaves the EEG signal unchanged.

---

## 4. Core projection backbone

The calibrated signal is flattened:

```text
[B, 17, 250] -> [B, 4250]
```

and projected to the shared embedding dimension using:

```text
Linear(4250, 1024)
```

This is followed by a residual transformation:

```text
ResidualAdd(
    GELU
    Linear(1024, 1024)
    Dropout(p=0.20)
)
```

and:

```text
LayerNorm(1024)
```

In compact form:

\[
h_0=W_0\operatorname{vec}(\tilde{x})+b_0,
\]

\[
h_1=h_0+
\operatorname{Dropout}
\left(
W_1\operatorname{GELU}(h_0)+b_1
\right),
\]

\[
h=\operatorname{LayerNorm}(h_1).
\]

This shallow projector is intentionally retained as the main representation path.

---

## 5. Bounded residual adapter

A lightweight bottleneck adapter is applied to the core representation:

```text
Linear(1024, 128)
GELU
Linear(128, 1024)
Dropout(p=0.10)
```

The hidden dimension is:

```text
adapter_hidden = max(32, D // 8) = 128
```

The final linear layer of the adapter is initialized to zero.

The adapter residual is bounded using `tanh`:

\[
\delta=\tanh(f_{\mathrm{adapter}}(h)).
\]

The base residual scale is initialized as:

```text
delta_scale = 0.10
```

The zero initialization ensures that this branch does not perturb the core projector at the beginning of training.

---

## 6. Sample-wise residual-strength gate

Rather than applying an identical adapter correction to every EEG sample, the model predicts a scalar modulation factor for each sample.

The gate is:

```text
Linear(1024, 128)
GELU
Linear(128, 1)
```

with:

```text
alpha_hidden = max(16, D // 8) = 128
```

The final layer is zero-initialized.

The sample-wise coefficient is:

\[
\alpha
=
1+\lambda_{\alpha}
\tanh(f_{\alpha}(h)),
\]

where:

```text
delta_alpha_scale = 0.05
```

The representation after the residual adapter is:

\[
h_{\mathrm{adapt}}
=
h+
\alpha\lambda_{\delta}\delta.
\]

At initialization, `f_alpha(h)=0`, so:

\[
\alpha=1.
\]

Thus the sample-wise gate introduces adaptive behavior without producing a large random perturbation at initialization.

---

## 7. Feature reweighting

The adapted 1,024-dimensional representation is further processed by a lightweight feature-wise gate:

```text
Linear(1024, 256)
GELU
Linear(256, 1024)
```

where:

```text
gate_hidden = max(16, D // 4) = 256
```

The final layer is initialized to zero.

The feature gate is:

\[
g=\tanh(f_g(h_{\mathrm{adapt}})).
\]

The final embedding is:

\[
z
=
h_{\mathrm{adapt}}
\odot
\left(1+\lambda_g g\right),
\]

with:

```text
feature scale lambda_g = 0.03
```

At initialization, the feature gate is zero and therefore the reweighting is also identity-preserving.

---

## 8. Initialization strategy

A central design principle of the encoder is **stability-preserving adaptation**.

The following components are identity- or zero-initialized:

| Component | Initialization |
|---|---|
| Channel scale | all ones |
| Temporal-gate final linear layer | zeros |
| Residual-adapter final linear layer | zeros |
| Sample-wise gate final linear layer | zeros |
| Feature-gate final linear layer | zeros |

Consequently, the adaptive branches begin close to an identity transformation and progressively learn sample- and feature-dependent corrections.

The initial adaptive scale values are:

```text
temporal modulation scale     0.14
adapter residual scale        0.10
sample-wise alpha scale       0.05
feature-reweight scale        0.03
```

---

## 9. Forward computation

The implementation can be summarized as:

```python
# x: [B, 17, 250]

# 1. Input calibration
x = x * channel_scale
t_context = x.mean(dim=1)
t_delta = tanh(temporal_gate(t_context))
x = x * (1 + temp_scale * t_delta.unsqueeze(1))

# 2. Core projection
x = x.reshape(B, 4250)
base = core(x)                       # [B, 1024]

# 3. Bounded residual adapter
delta = tanh(adapter(base))

# 4. Sample-wise adapter strength
alpha = 1 + delta_alpha_scale * tanh(delta_alpha_head(base))

x = base + alpha * delta_scale * delta

# 5. Feature reweighting
g = tanh(feature_gate(x))
x = x * (1 + feat_scale * g)

return x                             # [B, 1024]
```

---

## 10. Parameter count

For `C=17`, `T=250`, and `D=1024`, the approximate parameter allocation is:

| Component | Parameters |
|---|---:|
| Channel scale | 17 |
| Temporal calibration MLP | 15,781 |
| Core input projection `4250 -> 1024` | 4,353,024 |
| Core residual linear `1024 -> 1024` | 1,049,600 |
| Core LayerNorm | 2,048 |
| Residual adapter | 263,296 |
| Sample-wise residual gate | 131,329 |
| Feature-reweighting gate | 525,568 |
| Adaptive scalar parameters and compatibility parameter(s) | small |
| **Total** | **≈ 6.341 M** |

The computational-efficiency evaluation reports:

```text
Parameters:          6.341 M
FLOPs:               0.0127 G / sample
Training time:       3.293 ± 0.129 ms / batch
Inference time:      0.451 ± 0.059 ms / sample
200-way retrieval:   0.500 ± 0.016 ms / query
Peak training memory 143.2 MB
```

The timing measurements were performed in FP32 using a batch size of 256 for training-time benchmarking. Training latency includes EEG encoder forward propagation, symmetric contrastive loss, backward propagation, and the AdamW update.
