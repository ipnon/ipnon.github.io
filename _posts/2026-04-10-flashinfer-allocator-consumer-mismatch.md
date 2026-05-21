---
layout: post
title: 'FlashInfer W4A8 Autotune: Allocator–Consumer Mismatch'
---

FlashInfer's[^flashinfer-paper] `cutlass_fused_moe` is a fused Mixture-of-Experts (MoE) kernel used by vLLM, SGLang, and TensorRT-LLM for serving large MoE models. It supports multiple Nvidia architectures, but the FP8 quantization paths discussed here require SM90 (Hopper) or later. Quantization is a simple method of saving memory in numerical algorithms by mapping from a high-precision numerical domain to a low-precision one:

$$q(x) = \mathrm{round}(x / s), \qquad \hat{x} = s \cdot q(x)$$

We quantize to reduce memory footprint and bandwidth, and to unlock faster low-precision tensor core paths. We dequantize because the accumulator and the rest of the network still want real numbers. We dequantize at the boundary, compute in high precision inside the tensor core, and requantize before storing back to memory. Each quantization algorithm has much more detail in implementation, but at the highest level we divide some number $x$ by $s$ to quantize it into a low-precision numerical domain in memory, and then multiply it by $s$ such that we get a natively supported floating point format that can operate on the tensor cores. On the SM90 Hopper devices this lowest precision native floating point type is FP8.

There are many different quantization formats, but why? And which do we pick? This is a bit out of the scope of this post, but it suffices to say that each format lives somewhere along the independent dimensions of accuracy, throughput, and memory, and choosing which to use is mostly a matter of experimentation.

For the purposes of this report, we need only concern ourselves with W4A8.[^awq] W4A8 is groupwise-scaled signed-INT4 weights paired with per-tensor (or per-token) FP8 E4M3 activations, computed on FP8 tensor cores with an FP32 accumulator and a fused $s_W s_A$ output rescale. See the following bit trace:

```asm
0. Pack the weights into memory before inference.
  w0 = -3 -> 0b1101 in 4-bit two's complement (note a)
  w1 =  5 -> 0b0101
  packed = (w1 << 4)   | (w0 & 0x0F)
         = 0b0101_0000 | 0b0000_1101
         = 0b0101_1101
         = 0x5D, one byte in memory
         
1. Load from memory.
  weight: 0101_1101 (0x5D, packed into `uint8_t` holding w0=-3 and w1=5)
  activation: [0][0111][100] = +1.5 in FP8 E4M3 (note b)
              S=0, Exp=0b0111=7, bias=7, exp=0, Man=0b100
              (-1)^0 * 2^(0) * (1 + (0.5 + 0 + 0)) = 1.5

2. Unpack the weights in the registers.
  w0: 0b0101_1101 << 4 -> 0b1101_0000
      (int8_t)         -> 0b1101_0000 (b[7] = 1, so this is negative)
      >> 4             -> 0b1111_1101 = -3
  w1: 0b0101_1101 & 0xF0 -> 0b0101_0000
      (int8_t)           -> 0b0101_0000 (b[7] = 0, so this is positive)
      >> 4               -> 0b0000_0101 = 5

3. Dequantize the weights in the registers.
  w0f = (float)(-3) * s_W = -3 * 1.75 = -5.25
  w1f = (float)(5)  * s_W = 5  * 1.75 = 8.75

4. Promote the activation in the registers.
  a0 = (float)1.5 = 1.5

5. Compute the partial inner product in the FP32 accumulator. (note c)
  acc += a0 * w0f = 1.5 * -5.25 = -7.875

6. Rescale the output.
  result = acc * s_A = -7.875 * 0.5 = -3.9375

7. Requantize from FP32 to FP8 E4M3 in the registers.
  -3.9375 -> nearest representable E4M3
  3.9375 = 2^1 * 1.96875 -> mantissa needs 0b1111 but only 3 bits are available
  if Man = 0b111 then 2^1 * 1.875 = 3.75 -> |3.9375 - 3.75| = 0.1875
  if Man = 0b000 then 2^2 * 1.0   = 4.0  -> |3.9375 - 4.0 | = 0.0625
  0.0625 < 0.1875 -> round to 4.0
  encode: [1][1001][000]
          S=1, Exp=0b1001=9, bias=7, exp=2, Man=0b000
          (-1)^1 * 2^(2) * (1 + (0 + 0 + 0)) = -4.0

8. Store the byte back to memory. (note d)
  [1][1001][000] = 0xC8
```

Expansions for the labelled notes follow at the end of the post.[^note-a][^note-b][^note-c][^note-d]

And the same in CUDA:

```cpp
#include <cuda_fp8.h>
#include <cstdint>

// W is effectively stored as packed INT4. Each `uint8_t` holds two signed nibbles.
// For each group along K dimension we scale by `s_W`.
// For each tensor we scale by `s_A`.
// `__restrict__` qualifier declares that the given pointer's address space has
// no intersection with the address spaces of other parameters. This allows for
// more aggressive compilation.
__global__ void w4a8_gemm(
  const __nv_fp8_e4m3* __restrict__ A,   // FP8 activations: (M, K) 
  const uint8_t*       __restrict__ Wq,  // Packed INT4 weights: (N, K/2)
  const float*         __restrict__ s_W, // Per-group weight scales: (N, K/G)
  float                             s_A, // Per-tensor activation scalar
  __nv_fp8_e4m3*       __restrict__ Y,   // FP8 output: (M, N)
  int M, int N, int K, int G)
{
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (M <= m || N <= n) return;
  float acc = 0.0f;
  for (int k = 0; k < K; k += 2) {
    // Load one packed byte of two INT4 weights for the output row n.
    uint8_t packed = Wq[n * (K / 2) + (k / 2)];
    int8_t w0 = (int8_t)(packed << 4) >> 4; // Sign-extend the low nibble.
    int8_t w1 = (int8_t)(packed & 0xF0) >> 4; // Sign-extend the high nibble.
    
    // Dequantize from INT4 to FP32 using the per-group scalar.
    float scale = s_W[n * (K / G) + (k / G)];
    float w0f = (float)w0 * scale;
    float w1f = (float)w1 * scale;
    
    // Load FP8 activations and promote to FP32 for the multiply.
    float a0 = (float)A[m * K + k];
    float a1 = (float)A[m * K + k + 1];
    
    acc += a0 * w0f + a1 * w1f; // Normally the tensor core handles this in `mma.sync`.
  }
  
  // Rescale the accumulated output.
  // Note that `s_W` is already distributed into `w0f` and `w1f`.
  float result = acc * s_A;
  
  // Requantize from FP32 to FP8 E4M3 and store back to memory.
  // When we compute the next layer of the neural net, it will need FP8 again.
  Y[m * N + n] = __nv_fp8_e4m3(result);
}
```

Alright, all of this lengthy preamble is to establish that quantization can be a bit pernicious, and there are many different quantization schemes, and it can clearly be seen that they require very precise bit operations, and so mixing them up isn't going to produce numerical degradation, it's going to result in undefined behavior.

The fused MoE kernel supports 7 schemes:

1. FP8 per-tensor
2. FP8 block scaling
3. INT8
4. INT4 groupwise
5. W4A8, our example, uses `kUINT8` for packing
6. BF16 with MXFP4, also uses `kUINT8` for packing
7. No quantization, just normal passthrough

Each scheme requires its own allocation and preparation logic. Rather than a single dispatch table, FlashInfer uses a cascade of booleans to flag the weight and activation types. The valid schemes are a sparse subset of the flag combinations. Only 6 of the 32 states are valid schemes:

$$
\text{state space} = 2^n, \qquad \text{bug surface} = n \cdot m
$$

$$
2^5 = 32 \text{ combinations}, \qquad 5 \times 4 = 20 \text{ sites to miss a case}
$$

Both the allocator and the consumer need to agree on the quantization scheme or the behavior will be undefined.[^issue] In our case, FlashInfer has two relevant functions: `getProfilerWorkspaces()` and `prepareQuantParams()`. The former computes how much memory is needed for the scale buffers, such as `s_W` and `s_A` in our example, and this is called a workspace. The latter reads pointers out of the workspace based on the supposedly agreed on quantization scheme. Apparently when the W4A8 scheme was added, `prepareQuantParams()` was updated to include `kUINT8` as a valid packing format, but `getProfilerWorkspaces()` was not. Suffice to say the code is complicated, the weight data type classification and cascade will be briefly summarized here:

```cpp
// getProfilerWorkspaces() — allocator side
// For W4A8, wtype == kUINT8 (packed INT4 pairs). None of these match it.

bool is_int_w_quant            = (wtype == kINT8);           // kUINT8? false
bool is_int_groupwise_w_quant  = (wtype == kINT4);           // kUINT8? false
size_t dtype_bytes             = (wtype == kINT4) ? ... : ...; // kUINT8? falls through

// The cascade from here:
// is_int_groupwise_w_quant = false
// is_w4afp8_quant = is_int_groupwise_w_quant && is_fp8_act_quant = false
// w4a8_alpha_size = 0                      (gated on is_w4afp8_quant)
// ADD(w4a8_alpha) registers zero bytes     → workspace map entry is empty
// ...
```

The workspace allocator isn't naive to `kUINT8`, it's just that it was only added for the BF16+MXFP4 path, the other quantization scheme that utilizes `kUINT8` for packing:

```cpp
bool is_wfp4a16_quant = (wtype == kUINT8) && (atype == kHALF || atype == kBF16);
```

Note that the activation type for W4A8, E4M3, is nowhere to be found. The fix is to add `kUINT8` as a matchable quantization type at the previously mentioned sites:[^pr]

```cpp
// getProfilerWorkspaces() — after fix
// Add kUINT8 to the three sites, matching the existing kINT4 behavior.

bool is_int_w_quant            = (wtype == kINT8);
bool is_int_groupwise_w_quant  = (wtype == kINT4 || wtype == kUINT8);  // kUINT8? true
size_t dtype_bytes             = (wtype == kINT4 || wtype == kUINT8) ? ... : ...;

// The cascade now succeeds:
// is_int_groupwise_w_quant = true
// is_w4afp8_quant = is_int_groupwise_w_quant && is_fp8_act_quant = true
// w4a8_alpha_size = num_experts_per_node * sizeof(float)   (non-zero)
// ADD(w4a8_alpha) registers a real buffer
// ...

// prepareQuantParams() — consumer side
// GET_WS_PTR(float const*, w4a8_alpha) returns a valid pointer
// assert(quant_1 && quant_2);              → both valid → succeeds
```

A natural question to ask for any error is why it wasn't found. In this case, the default kernel configuration in FlashInfer uses an entirely different allocator: `CutlassMoeFCRunner::getWorkspaceDeviceBufferSizes()`. Only when autotuning is enabled do we use the allocator `GemmProfilerBackend::getProfilerWorkspaces()`. It stands to reason that the other allocator has been working just fine, or was fixed earlier, as most people aren't autotuning. It takes a long time to compile. And there are several other quantization schemes to use other way. It takes a while for someone to need autotuning and W4A8 specifically at the same time.

A separate issue is that a dispatch table has measurable overhead, and FlashInfer's main selling point is that it's fast (it's not called BangInfer). It's evidently not worth it to write a dispatch table if it cuts into performance at any measurable amount. The kernel itself is built on NVIDIA's CUTLASS templates.[^cutlass]

[^note-a]: Note (a). *4-bit two's complement* means you have 1 sign bit and 3 magnitude bits. A simple way to think of it is that setting each bit contributes the corresponding integer to the overall sum in this array: `[-8, 4, 2, 1]`, where `-8` is referred to as the "minimum representable value". An occasionally useful property of this format is that you can apply the composition of the bitwise-not function and plus-one function to negate any number: `3 = 0b0011`; `(~)(0b0011) = 0b1100`; `(+1)(0b1100) = 0b1101 = -3`; and applying the same composition again yields `0b0011 = 3`.

[^note-b]: Note (b). *FP8 E4M3* means you have 1 sign bit, 4 exponent bits and 3 mantissa bits: `[S][E3 E2 E1 E0][M2 M1 M0]`. The bias is 7, which means to find the actual exponent we subtract 7 from the value of the exponent bits. An implicit leading 1 is prepended to the mantissa; a simple way to think of it is that setting each bit adds the corresponding term of the geometric sequence $2^{-i}$ to the rational $1 = 2^0$: `[1/2, 1/4, 1/8]`. We compute every value as $(-1)^S \cdot 2^{\text{Exp} - 7} \cdot (1 + M/8)$. For example, `[1][1000][110]` is $(-1)^1 \cdot 2^{8-7} \cdot (1 + 1/2 + 1/4) = -3.5$.

[^note-c]: Note (c). The astute reader will note that this step is unnecessary on any device supporting FP8 MMA, which is ~10% of deployed devices as of writing. We promote to FP32 for purposes of pedagogy.

[^note-d]: Note (d). This is a single step in what would continue over the entire K dimension. A tutorial on the entire process can be found at [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

[^issue]: Issue [flashinfer-ai/flashinfer#2501](https://github.com/flashinfer-ai/flashinfer/issues/2501).

[^pr]: Fix in [flashinfer-ai/flashinfer#2564](https://github.com/flashinfer-ai/flashinfer/pull/2564).

[^awq]: Lin et al., ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978) (MLSys 2024).

[^cutlass]: NVIDIA, [`examples/24_gemm_grouped/gemm_grouped.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu) and [`include/cutlass/gemm/device/gemm_grouped.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/gemm_grouped.h).

[^flashinfer-paper]: Ye et al., ["FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving"](https://arxiv.org/abs/2501.01005).
