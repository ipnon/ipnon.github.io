---
layout: post
title: 'D-Parameter Shape Mismatch Breaks Mamba2''s Forward–Step Parity'
---

Mamba2 is a state space model (SSM). In a transformer architecture, every token peeks at every other token using the attention mechanism, causing linear growth in the transformer's KV cache (the store of every past token's key and value) and quadratic growth in attention computation. In contrast, SSM uses a fixed size hidden state vector $h$, allowing for both constant memory and compute:

$$h_t = A \cdot h_{t-1} + B \cdot x_t$$

$$y_t = C \cdot h_t + D \cdot x_t$$

The hidden state $h$ carries information forward through time, similarly to the attention mechanism. At each step it decays by $A$, gets new input through $B$, and produces output through $C$. The skip connection $D$ lets input pass directly to the output.

Mamba2 has different ways to create the same outputs. The first is `forward()`. This processes an entire sequence at once. In scenarios like training where all tokens are known in advance, this is much faster. The second is `step()`, which processes inputs one token at a time. In scenarios like inference where all tokens are not known in advance, tokens are fed in sequence. This is called autoregressive decoding, a wonderfully obtuse piece of ML jargon that means "keep calling `step()` until we reach the end of sequence token."

Now, for each matrix $A$, $B$, $C$, $D$, and $h$, we have different shapes:

  - $A$ has one decay scalar per head: `(nheads,)` 
  - $B$ has input-to-state projection per group: `(batch, ngroups, d_state)`
  - $C$ has state-to-output projection per group: `(batch, ngroups, d_state)`
  - $D$ has two possible shapes: `(nheads,)` or `(nheads * headdim,)`
  - $h$ connecting to all other four: `(batch, nheads, headdim, d_state)`

Now as you can see $D$ is the odd one out having two options. This arises from the hyperparameter `D_has_hdim`. If we do not set this hyperparameter, we get shape `(nheads,)` for $D$. Every element within a head gets scaled by the same value. If we do set it, $D$ has shape `(nheads * headdim,)` giving us one scalar per element. Every head element can weight its own skip connection. This allows some coarseness tuning for $D$.

Now what happens if we do not account for these separate shapes? `forward()` handles both cases correctly:

```python
D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D
```

If `D_has_hdim` is set, the 1D tensor `(nheads * headdim,)` is reshaped to 2D `(nheads, headdim)`. Now what happens in `step()`? Here `x` is the post-conv1d SSM input with shape `(batch, nheads, headdim)`:

```python
# First path with no `rmsnorm`
y = y + rearrange(self.D.to(dtype), "h -> h 1") * x

# Second path with `rmsnorm`
D = repeat(self.D, "h -> h p", p = self.headdim)
```

No such handling occurs. Both paths assume $D$ has shape `(nheads,)`. When `D_has_hdim=True` and $D$ is actually `(nheads * headdim,)`, say `(256,)`, the first path reshapes it to `(256, 1)` instead of `(4, 64)`. Note that PyTorch doesn't crash and each head-element gets scaled by the wrong $D$ value. We need the same conditional handling of the different $D$ shapes in both `forward()` and `step()`.

```python
# Path 1: was rearrange("h -> h 1"), now:
if self.D_has_hdim:
    y = y + rearrange(self.D.to(dtype), "(h p) -> h p", p=self.headdim) * x
else:
    y = y + rearrange(self.D.to(dtype), "h -> h 1") * x

# Path 2: was repeat("h -> h p"), now:
if self.D_has_hdim:
    D = rearrange(self.D, "(h p) -> h p", p=self.headdim)
else:
    D = repeat(self.D, "h -> h p", p=self.headdim)
```

Now how did this bug occur? Why wasn't it found sooner?[^issues] First and most important is that PyTorch will broadcast nearly anything, and it will not inform you when dimensions don't align. Secondly, $D$ is initialized to all 1s by default with `torch.ones`. Note that $1 \cdot x = x$ no matter which dimension we have for $x$. Any test using default initialization would pass, because you need non-uniform $D$ values to expose the mismatch.

We can test correctness by creating a Mamba2 model, randomizing $D$, and check that both `forward()` and `step()` produce matching outputs for both hyperparameter settings:

```python
# Randomize D
with torch.no_grad():
    model.D.copy_(torch.randn_like(model.D))

# Run both paths with identical x
out_forward = model(x)

conv_state, ssm_state = model.allocate_inference_cache(batch, seqlen, dtype=dtype)
for t in range(seqlen):
    out_t, conv_state, ssm_state = model.step(
        x[:, t : t + 1, :], conv_state, ssm_state
    )

# After warmup assert any differences are within reasonable bounds for numerical noise
assert torch.allclose(out_fwd_tail, out_step_tail, rtol=1e-3, atol=1e-3)
```

The fix landed in [PR #893](https://github.com/state-spaces/mamba/pull/893).[^pr]

[^issues]: Originally reported in [Issue #887: Mamba2.step() handles D incorrectly when D_has_hdim=True](https://github.com/state-spaces/mamba/issues/887) and [Issue #888: Mamba2 step() silent misbehavior with D_has_hdim=True](https://github.com/state-spaces/mamba/issues/888).
[^pr]: [PR #893: Fix Mamba2 step() D handling when D_has_hdim=True](https://github.com/state-spaces/mamba/pull/893).
