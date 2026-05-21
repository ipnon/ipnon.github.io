---
layout: post
title: 'D-Parameter Shape Mismatch Breaks Mamba2''s Forward–Step Parity'
---

<div class="tikz-figure">
<script type="text/tikz">
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\begin{tikzpicture}[
  >={Stealth[length=2.5mm, inset=0.5mm]},
  font=\small,
  box/.style={draw, align=center, inner sep=5pt, rounded corners=1pt, minimum height=11mm},
  smbox/.style={box, font=\footnotesize},
  edgelbl/.style={font=\scriptsize, fill=white, inner sep=2pt},
]
\useasboundingbox (-12, -7) rectangle (16, 5);
\node[smbox] (fwd) at (-8.5, 3.5) {\texttt{forward()} \\ \emph{mamba2.py:191}};
\node[smbox, dashed] (s1) at (0, 3.5) {\texttt{step()} path 1 \\ \emph{mamba2.py:319, \texttt{selective\_state\_update is None}}};
\node[smbox, dashed] (s2) at (8.5, 3.5) {\texttt{step()} path 2 \\ \emph{mamba2.py:327, else branch}};
\node[box, align=center] (dpar) at (0, 0) {\texttt{self.D} parameter \\ \emph{shape depends on} \texttt{D\_has\_hdim}};
\node[smbox, align=left] (dfalse) at (-6, -2.5) {\texttt{D\_has\_hdim=False} \\ shape \texttt{(nheads,)}};
\node[smbox, dashed, align=left] (dtrue) at ( 6, -2.5) {\texttt{D\_has\_hdim=True} \\ shape \texttt{(nheads * headdim,)} \\ \emph{the unhandled case}};
\node[box, minimum width=110mm] (mul) at (0, -5) {broadcast multiply with \texttt{x:\,(batch,\,nheads,\,headdim)} $\Rightarrow$ \texttt{y\,+\,D\,*\,x}};
\node[smbox, dashed, align=left] (haz) at (12.5, 0.5) {
  failure (\texttt{D\_has\_hdim=True}): \\
  \quad path 1: \texttt{"h -> h 1"} \\
  \quad\quad yields \texttt{(h*p,\,1)} \\
  \quad path 2: \texttt{"h -> h p"} \\
  \quad\quad makes \texttt{(h*p,\,p)} \\
  \quad broadcasts silently \\
  \quad wrong \texttt{D} per head element \\
  \quad masked by default \texttt{D=ones} \\
  \quad \texttt{step()} drifts from \texttt{forward()}
};
\node[draw, dotted, fit=(s1)(s2), inner sep=10pt] (stepgrp) {};
\node[font=\footnotesize, anchor=south west] at ([xshift=3pt]stepgrp.north west) {Mamba2.step()};
\draw[->] (fwd.south) to[bend right=10] node[edgelbl, pos=0.55, sloped, above] {\texttt{rearrange("(h p) -> h p")} \emph{iff} \texttt{D\_has\_hdim}} (dpar.north west);
\draw[->, dashed] (s1.south) -- (dpar.north) node[edgelbl, pos=0.55, right] {\texttt{rearrange("h -> h 1")} \emph{unconditional}};
\draw[->, dashed] (s2.south) to[bend left=10] node[edgelbl, pos=0.55, sloped, above] {\texttt{repeat("h -> h p",\,p=headdim)} \emph{unconditional}} (dpar.north east);
\draw[->] (dpar.south west) to[bend right=10] node[edgelbl, pos=0.5, sloped, above] {if \texttt{D\_has\_hdim=False}} (dfalse.north);
\draw[->, dashed] (dpar.south east) to[bend left=10] node[edgelbl, pos=0.5, sloped, above] {if \texttt{D\_has\_hdim=True}} (dtrue.north);
\draw[->] (dfalse.south) to[bend right=15] (mul.north west);
\draw[->, dashed] (dtrue.south) to[bend left=15] (mul.north east);
\draw[->, dotted] (dtrue.east) to[bend left=10] (haz.west);
\end{tikzpicture}
</script>
<div class="caption"><b>Figure 1.</b> Pre-fix state. <code>forward()</code> reshapes <code>self.D</code> conditionally on <code>D_has_hdim</code>, but both <code>step()</code> paths apply a fixed <code>(nheads,)</code>-shaped transform; when <code>D_has_hdim=True</code> the resulting tensor still broadcasts against <code>x</code>, so each head-element is silently scaled by the wrong <code>D</code> value.</div>
</div>

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

<div class="tikz-figure">
<script type="text/tikz">
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\begin{tikzpicture}[
  >={Stealth[length=2.5mm, inset=0.5mm]},
  font=\small,
  box/.style={draw, align=center, inner sep=5pt, rounded corners=1pt, minimum height=11mm},
  smbox/.style={box, font=\footnotesize},
  edgelbl/.style={font=\scriptsize, fill=white, inner sep=2pt},
]
\useasboundingbox (-12, -7) rectangle (16, 5);
\node[smbox] (fwd) at (-8.5, 3.5) {\texttt{forward()} \\ \emph{mamba2.py:191}};
\node[smbox, line width=0.8pt] (s1) at (0, 3.5) {\texttt{step()} path 1 \\ \emph{mamba2.py:319, \texttt{selective\_state\_update is None}}};
\node[smbox, line width=0.8pt] (s2) at (8.5, 3.5) {\texttt{step()} path 2 \\ \emph{mamba2.py:327, else branch}};
\node[box, align=center] (dpar) at (0, 0) {\texttt{self.D} parameter \\ \emph{shape depends on} \texttt{D\_has\_hdim}};
\node[smbox, align=left] (dfalse) at (-6, -2.5) {\texttt{D\_has\_hdim=False} \\ shape \texttt{(nheads,)}};
\node[smbox, align=left] (dtrue) at ( 6, -2.5) {\texttt{D\_has\_hdim=True} \\ shape \texttt{(nheads * headdim,)}};
\node[box, minimum width=110mm] (mul) at (0, -5) {broadcast multiply with \texttt{x:\,(batch,\,nheads,\,headdim)} $\Rightarrow$ \texttt{y\,+\,D\,*\,x}};
\node[draw, dotted, fit=(s1)(s2), inner sep=10pt] (stepgrp) {};
\node[font=\footnotesize, anchor=south west] at ([xshift=3pt]stepgrp.north west) {Mamba2.step()};
\draw[->] (fwd.south) to[bend right=10] node[edgelbl, pos=0.55, sloped, above] {\texttt{rearrange("(h p) -> h p")} \emph{iff} \texttt{D\_has\_hdim}} (dpar.north west);
\draw[->, line width=0.8pt] (s1.south) -- (dpar.north) node[edgelbl, pos=0.55, right] {\textbf{fix}: same conditional as \texttt{forward()}};
\draw[->, line width=0.8pt] (s2.south) to[bend left=10] node[edgelbl, pos=0.55, sloped, above] {\textbf{fix}: same conditional as \texttt{forward()}} (dpar.north east);
\draw[->] (dpar.south west) to[bend right=10] node[edgelbl, pos=0.5, sloped, above] {if \texttt{D\_has\_hdim=False}} (dfalse.north);
\draw[->] (dpar.south east) to[bend left=10] node[edgelbl, pos=0.5, sloped, above] {if \texttt{D\_has\_hdim=True}} (dtrue.north);
\draw[->] (dfalse.south) to[bend right=15] (mul.north west);
\draw[->] (dtrue.south) to[bend left=15] (mul.north east);
\end{tikzpicture}
</script>
<div class="caption"><b>Figure 2.</b> Post-fix state. Both <code>step()</code> paths now branch on <code>D_has_hdim</code> exactly as <code>forward()</code> does, so the <code>(nheads * headdim,)</code> case is reshaped to <code>(nheads, headdim)</code> before the multiply and <code>step()</code> matches <code>forward()</code> for any non-uniform <code>D</code>.</div>
</div>

[^issues]: Originally reported in [Issue #887: Mamba2.step() handles D incorrectly when D_has_hdim=True](https://github.com/state-spaces/mamba/issues/887) and [Issue #888: Mamba2 step() silent misbehavior with D_has_hdim=True](https://github.com/state-spaces/mamba/issues/888).
[^pr]: [PR #893: Fix Mamba2 step() D handling when D_has_hdim=True](https://github.com/state-spaces/mamba/pull/893).
