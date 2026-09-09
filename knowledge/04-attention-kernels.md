# Attention 后端与底层算子

> 基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（v1 引擎）源码。本文聚焦 **架构** 与 **部署/调优**：attention backend 的抽象与选择机制、各 backend 的适用场景、vLLM 自研/集成的 CUDA kernel、Triton kernel 用途，以及切换 backend 的旋钮。

---

## 1. v1 Attention 后端抽象
<!-- tags: attention, backend, abstraction, 注意力 -->

v1 的 attention 抽象位于 `vllm/v1/attention/`，核心是"一个 backend = 三个类"的组合，全部定义在 `vllm/v1/attention/backend.py`：

| 抽象类 | 职责 |
|---|---|
| `AttentionBackend` (ABC) | 描述 backend 的能力与 KV cache 布局。静态方法：`get_name()`、`get_impl_cls()`、`get_builder_cls()`、`get_kv_cache_shape()`、`get_kv_cache_stride_order()`，以及一组 `supports_*()` 能力查询（`supports_head_size` / `supports_dtype` / `supports_kv_cache_dtype` / `supports_block_size` / `supports_sink` / `supports_sliding_window` / `supports_non_causal` / `supports_compute_capability` / `is_mla` / `is_sparse` / `is_ssm` 等）。`validate_configuration(...)` 汇总所有能力检查，返回"不可用原因列表"，是选择机制的核心。 |
| `AttentionMetadataBuilder` (ABC, Generic) | 每个 step 由 scheduler 产出的 `CommonAttentionMetadata` 构建出 backend 专属的 metadata（`build()`）。关键类属性：`_cudagraph_support`（`AttentionCGSupport` 枚举：`ALWAYS` / `UNIFORM_BATCH` / `UNIFORM_SINGLE_TOKEN_DECODE` / `NEVER`）、`reorder_batch_threshold`（是否把 decode 重排到 batch 前部）。 |
| `AttentionImpl` / `MLAAttentionImpl` (ABC) | 真正执行 attention 的 forward。标准 attention 用 `AttentionImpl.forward(layer, q, k, v, kv_cache, attn_metadata, output, ...)`；MLA 用 `MLAAttentionImpl.forward_mha()`（prefill，compute-friendly）+ `forward_mqa()`（decode，data-movement-friendly）。 |

`CommonAttentionMetadata`（`backend.py:457`）是跨 backend 共享的 per-batch 元数据，关键字段：`query_start_loc`（GPU+CPU 双份）、`seq_lens`、`num_actual_tokens`、`max_query_len`、`max_seq_len`、`block_table_tensor`、`slot_mapping`、`causal`（可为 bool 或 per-seq tensor，FA4 支持 per-seq causal）、`is_prefilling`、`mm_req_doc_ranges`（PrefixLM 双向区间）、`rswa_prefix_lens`（Reference Sliding Window Attention）。

**KV cache 布局**：backend 通过 `get_kv_cache_shape()` 给出逻辑 shape，通过 `get_kv_cache_stride_order()` 给出物理维度排列。标准 MHA 逻辑 shape 为 `(num_blocks, num_kv_heads, block_size, 2*head_size)`（K/V 打包进 content 维），物理布局由 `VLLM_KV_CACHE_LAYOUT`（`NHD`/`HND`）决定。MLA 的 cache 是 latent 布局（`num_kv_heads` 恒为 1，content 维 = `kv_lora_rank + qk_rope_head_dim`）。

**KV cache 写入与 attention 解耦**：`AttentionBackend.forward_includes_kv_cache_update` 标记 forward 是否包含 KV 写入。FLASH_ATTN / FLASHINFER / TRITON_ATTN 均为 `False`，KV 写入由 `torch.ops.vllm.unified_kv_cache_update` 单独完成（见 `vllm/model_executor/layers/attention/attention.py:701`）。

---

## 2. Backend 选择机制
<!-- tags: attention, backend-selection, sm, sm120, 选择机制 -->

### 2.1 入口与优先级
<!-- tags: backend-selection, 选择, priority, cuda, sm100 -->

选择入口是 `vllm/v1/attention/selector.py::get_attn_backend()`（`attention.py:230` 的 `Attention.__init__` 调用）。流程：

1. 组装 `AttentionSelectorConfig`（head_size、dtype、kv_cache_dtype、block_size、use_mla、has_sink、use_sparse、attn_type、has_sliding_window、use_non_causal、use_dcp/pcp 等）。
2. 读取 `attention_config.backend`（用户显式指定）与 `attention_config.backend_per_kind`（按 KV-cache-group kind 覆盖，见 §6）。
3. 调 `current_platform.get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)`（`vllm/platforms/cuda.py:404`）。

**CUDA 平台选择逻辑**（`vllm/platforms/cuda.py`）：
- 若用户显式指定 backend：先 `validate_configuration`，不合法直接 `ValueError`（不静默回退）。
- 否则按 `_get_backend_priorities()` 给出的优先级列表逐个 `validate_configuration`，取第一个合法者。若 `--block-size` 排除了更高优先级 backend，会打 warning 提示。

**CUDA 优先级**（`cuda.py:83 _get_backend_priorities`）：

- **非 MLA**（普通 MHA/GQA）：
  - SM100 (Blackwell) 且 causal：`FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT`
  - 其他（含 SM100 non-causal、SM90、SM80）：`FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT`
- **MLA**：
  - SM100：`FLASHINFER_MLA → TOKENSPEED_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA → *sparse_backends`（sparse 依 kv dtype / head 数在 `FLASHINFER_MLA_SPARSE` 与 `FLASHMLA_SPARSE` 间排序）
  - SM120：`TRITON_MLA → FLASHINFER_MLA_SPARSE_SM120`
  - 其他：`FLASH_ATTN_MLA → FLASHMLA → FLASHINFER_MLA → TRITON_MLA → FLASH_ATTN_MLA_SPARSE → FLASHMLA_SPARSE`

**ROCm 平台**（`vllm/platforms/rocm.py:618`）：非 MLA 为 `ROCM_ATTN → ROCM_AITER_FA → ROCM_AITER_UNIFIED_ATTN → TRITON_ATTN → TURBOQUANT`；MLA 为 `ROCM_AITER_MLA → TRITON_MLA → ROCM_AITER_TRITON_MLA`（sparse 用 `ROCM_AITER_MLA_SPARSE`）。

**CPU 平台**（`vllm/platforms/cpu.py:83`）：MLA 优先 `AMX_MLA`（x86 + AMX tile 支持），否则 `CPU_MLA`；非 MLA 用 `CPU_ATTN`。

### 2.2 Backend 注册表
<!-- tags: registry, backend-enum, 注册表, register-backend, mamba -->

`vllm/v1/attention/backends/registry.py` 定义 `AttentionBackendEnum`（每个枚举值 = 默认类路径字符串，可用 `register_backend()` 运行时覆盖，支持第三方 backend 通过 `CUSTOM` 注册）。主要成员：

- 通用：`FLASH_ATTN`、`FLASH_ATTN_DIFFKV`、`TRITON_ATTN`、`TRITON_ATTN_DIFFKV`、`FLASHINFER`、`FLEX_ATTENTION`、`TURBOQUANT`、`HPC_ATTN`、`NO_ATTENTION`、`TORCH_SDPA`（仅 ViT）
- MLA 专用：`FLASHINFER_MLA`、`FLASHMLA`、`FLASHMLA_SPARSE`、`TRITON_MLA`、`CUTLASS_MLA`、`FLASH_ATTN_MLA`、`FLASH_ATTN_MLA_SPARSE`、`TOKENSPEED_MLA`、`FLASHINFER_MLA_SPARSE`、`FLASHINFER_MLA_SPARSE_SM120`、`ROCM_AITER_MLA`、`AMX_MLA`、`CPU_MLA` 等
- 模型驱动 sparse：`FLASHMLA_SPARSE_DSV4`、`FLASHINFER_MLA_SPARSE_DSV4`、`MINIMAX_M3_SPARSE`、`CUTLASS_MSA`、`TRITON_MSA`

另有 `MambaAttentionBackendEnum`（`MAMBA1`/`MAMBA2`/`SHORT_CONV`/`LINEAR`/`GDN_ATTN`）用于 SSM/线性 attention 混合层，由 `--mamba-backend` 选择。

---

## 3. Prefill 与 Decode 两条路径
<!-- tags: attention, prefill, decode -->

v1 的 scheduler 把每个 step 的 batch 拆成 **prefill 段**（query_len > 1）与 **decode 段**（query_len == 1，或 spec-decode 的 1+draft）。不同 backend 处理这两段的方式不同：

- **FLASH_ATTN**：统一走 `flash_attn_varlen_func`（`flash_attn.py:1123`），prefill 与 decode 用同一个 varlen kernel，靠 `cu_seqlens_q`/`seqused_k`/`block_table` 区分。KV cache 逻辑 shape `(B, H, N, 2*D)`，forward 里 `kv_cache.transpose(1,2).split(head_size)` 拆出 K/V。支持 cascade attention（`use_cascade` 分支）。
- **FLASHINFER**：`FlashInferMetadata` 显式分 `prefill`（`FIPrefill`/`TRTLLMPrefill`）与 `decode`（`FIDecode`/`FlashInferTrtllmAPIDecode`）两个 wrapper（`flashinfer.py:655`）。decode kernel 由 `FlashInferDecodeKernel` 枚举选择：`XQA`（SM90）或 `TRTLLM_GEN`（SM100 trtllm-gen）。prefill 可选 TRTLLM ragged kernel。这是"prefill/decode 分路"最典型的 backend。
- **TRITON_ATTN**：默认走 `unified_attention`（`triton_attn.py:710`，`vllm/v1/attention/ops/triton_unified_attention.py` 的 `kernel_unified_attention`），单 kernel 同时处理 prefill+decode；`AttentionConfig.use_prefill_decode_attention=True` 时改用分离的 `context_attention_fwd`（prefill）+ decode kernel。
- **MLA**：`MLAAttention.forward_impl`（`mla_attention.py:731`）按 `num_mqa_tokens`（decode）/`num_mha_tokens`（prefill）切分：decode 段调 `impl.forward_mqa()`，prefill 段调 `impl.forward_mha()`（若实现）。prefill 后端由独立的 `MLAPrefillBackendEnum` 选择（见 §4.2）。
- **SSM/线性注意力（GDN 等）**：`backends/recoverssm_metadata.py` 的 `RecoverSSMMetadata` 抽象负责 spec decode 下 SSM 状态的"回滚/恢复"——`commit_recoverssm_state(num_accepted_tokens)` 按实际接受 token 数产出 `RecoverSSMPostprocessMetadata`（供 align-mode 前缀缓存的 postprocess）。这是混合架构（Gated DeltaNet 等）+ 投机解码的配套机制。

CUDA Graph 支持等级由 builder 的 `_cudagraph_support` 决定：FLASH_ATTN 在 FA3 下为 `ALWAYS`（支持混合 prefill-decode），FA2 下为 `UNIFORM_BATCH`；TRITON_ATTN 为 `ALWAYS`。

---

## 4. 各主要 Backend 特点与适用场景
<!-- tags: attention, flashattention, flashinfer, triton, mla, flex-attention -->

### 4.1 通用（非 MLA）
<!-- tags: backend, flash-attn, flashinfer, triton, 通用 -->

| Backend | 文件 | 特点 / 适用 |
|---|---|---|
| **FLASH_ATTN** | `backends/flash_attn.py` | 默认首选（非 SM100）。基于 vLLM 自带的 `vllm_flash_attn`（FA2/FA3/FA4，见 §5.1）。支持 sliding window、non-causal、sink（需 FA3/SM90+）、FP8 KV（需 FA3/SM90 或 FA4/SM100）、所有 attn_type。head_size 须 %8==0 且 ≤256（FA4 可到 512）。block_size 须 %16==0。 |
| **FLASHINFER** | `backends/flashinfer.py` | SM100 causal 首选。支持 head_size ∈ {64,128,256,512}，SM80–SM121。支持 FP8/FP8_e5m2/NVFP4 KV（NVFP4 需 SM100 + trtllm）。prefill/decode 分路，decode 可选 XQA / trtllm-gen。SM100 强制 `HND` KV layout（`get_required_kv_cache_layout`）。支持 fused output quant（FP8/NVFP4，需 trtllm-gen）。 |
| **TRITON_ATTN** | `backends/triton_attn.py` | 兜底/通用，`supports_compute_capability` 恒 True。支持 FP8/INT8/INT4 per-token-head KV 量化（inline scale 打包，见 `get_kv_cache_shape` 的 padded_hs 逻辑）、sink、alibi_sqrt、mm_prefix。cudagraph `ALWAYS`。适合需要 per-token-head 量化 KV 或老架构。 |
| **FLEX_ATTENTION** | `backends/flex_attention.py` | 基于 PyTorch `torch.compile` 的 FlexAttention，用 `mask_mod` 表达 causal/sliding-window/mm_prefix 等 mask。适合需要高度自定义 mask 或依赖 torch.compile 融合的场景。 |
| **TURBOQUANT** | `backends/turboquant_attn.py` | TurboQuant KV 压缩专用（`turboquant_k8v4`/`4bit_nc`/`3bit_nc` 等 kv_cache_dtype）。K+V 打包进单 slot，独立 cache shape。仅 decoder attn_type。 |
| **HPC_ATTN** | `backends/hpc_attn.py` | 基于 Tencent hpc-ops，仅 Hopper（H20/H200），当前限 Hy3 模型，block_size 须 64。 |

### 4.2 MLA 专用
<!-- tags: mla, backend, flashmla, cutlass, prefill -->

MLA（DeepSeek 系列）的 KV cache 存 latent（`kv_c` + `k_pe`），backend 需实现 `forward_mha`（prefill）+ `forward_mqa`（decode）。`MLACommonBackend` 基类在 `mla_attention.py:1422`。

- **FLASHMLA**（`mla/flashmla.py`）：DeepSeek 官方 FlashMLA kernel（`vllm._flashmla_C`），dense 仅 SM90，sparse 支持 SM90+SM100。block_size 固定 64。
- **FLASHINFER_MLA**（`mla/flashinfer_mla.py`）：SM100 首选 MLA decode。
- **CUTLASS_MLA**（`mla/cutlass_mla.py`）：SM100 CUTLASS kernel（`sm100_cutlass_mla_kernel.cu`）。
- **TRITON_MLA**（`mla/triton_mla.py`）：Triton 实现，通用兜底，SM120 首选。
- **FLASH_ATTN_MLA**（`mla/flashattn_mla.py`）：用 vllm_flash_attn 做 MLA。
- **Sparse MLA**：`FLASHMLA_SPARSE`、`FLASHINFER_MLA_SPARSE`、`FLASH_ATTN_MLA_SPARSE` 等，配合 indexer（`mla/indexer.py` 的 `DeepseekV32IndexerBackend`）做 top-k 稀疏。

**MLA prefill 后端**独立选择（`mla/prefill/selector.py`），`MLAPrefillBackendEnum`：`FLASH_ATTN` / `FLASHINFER` / `TRTLLM_RAGGED` / `TOKENSPEED_MLA` / `ROCM_AITER_FA` / `CPU_NATIVE`。优先级（`_get_mla_prefill_backend_priorities`）：SM100 且 DSV3 维度（192/64/256）时 `TRTLLM_RAGGED` 优先；否则 `FLASH_ATTN` 优先。可用 `AttentionConfig.mla_prefill_backend` 显式指定。

---

## 5. vLLM 自研/集成的关键 CUDA kernel（csrc/）
<!-- tags: kernels, cuda, csrc, moe-kernel, quant-kernel -->

CUDA kernel 集中在 `csrc/libtorch_stable/`（libtorch stable ABI，注册到 `torch.ops._C.*`，见 `csrc/libtorch_stable/torch_bindings.cpp` 的 `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops)`）。Python 侧经 `vllm._custom_ops` 调用。

### 5.1 vLLM 自带 FlashAttention 封装（`vllm/vllm_flash_attn/`）
<!-- tags: flash-attn, 封装, fa2, fa3, fa4 -->

`vllm/vllm_flash_attn/flash_attn_interface.py` 是对 Tri Dao flash-attn 的封装，只维护 vLLM 需要的两个入口：`flash_attn_varlen_func`（paged，带 `block_table`/`seqused_k`/`scheduler_metadata`/`q/k/v_descale`/`num_splits`/`s_aux`(sinks)/`mask_mod`/`dynamic_causal`）与 `flash_attn_with_kvcache`。编译产物 `_vllm_fa2_C` / `_vllm_fa3_C`，FA4 走 `vllm.vllm_flash_attn.cute.interface`（CuTE-DSL）。

**FA 版本选择**（`vllm/v1/attention/backends/fa_utils.py::get_flash_attn_version`）：默认 SM90→FA3、SM100→FA4、其余→FA2；可被 `AttentionConfig.flash_attn_version` 覆盖；ALiBi 强制回退 FA2；SM100 上 FA3 自动降为 FA4/FA2。

### 5.2 Attention / MLA kernel
<!-- tags: kernels, mla, cascade, dcp, kv-write -->

- `csrc/libtorch_stable/attention/merge_attn_states.cu` — `merge_attn_states`：cascade attention 合并 prefix + suffix 两段 attention 输出（用 log-sum-exp）。
- `csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu` — SM100 CUTLASS MLA decode（`sm100_cutlass_mla_decode`）。
- `csrc/libtorch_stable/attention/dcp_utils/` — decode context parallelism 的 LSE reduce / KV gather / Q gather（`dcp_direct_a2a_lse_reduce.cu` 等）。注：原 `vllm/v1/attention/ops/dcp_alltoall.py` 已删除，CP/DCP 的 attention ops 在 #52839 中整合进 `csrc` 侧（`VLLM_USE_DIRECT_DCP_A2A/Q_GATHER/KV_GATHER` 控制 direct 路径）。
- `csrc/attention/attention_generic.cuh` + `attention_dtypes.h` — 从 FasterTransformer 移植的通用 paged-attention 模板（dtype 特化 `dtype_{float16,bfloat16,float32,fp8}.cuh`），主要供 ROCm/legacy 路径引用。
- `csrc/rocm/attention.cu` — ROCm 原生 attention kernel。
- **MLA KV 写入**：`concat_and_cache_mla` / `concat_and_cache_mla_grouped` / `concat_and_cache_mla_rope_fused`（`_custom_ops.py:2817/2830/2877`），把 `kv_c`+`k_pe` 拼接送入 MLA cache（可融合 RoPE、支持 FP8）。

### 5.3 KV cache 管理 kernel
<!-- tags: kv-cache, kernels, reshape-and-cache, swap, gather -->

- `csrc/libtorch_stable/cache_kernels.cu` — `reshape_and_cache` / `reshape_and_cache_flash`（写 KV，flash 版 K/V 打包）、`swap_blocks` / `swap_blocks_batch`（preemption 换页）、`cp_gather_cache` / `cp_gather_and_upconvert_fp8_kv_cache`（context-parallel gather）、`gather_and_maybe_dequant_cache`。
- `csrc/libtorch_stable/cache_kernels_fused.cu` — 融合版 KV 写入。
- `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` — NVFP4 KV cache 处理。

### 5.4 MoE kernel
<!-- tags: moe, kernels, fused-moe, triton, router -->

- `vllm/model_executor/layers/fused_moe/fused_moe.py` — **Triton** `fused_moe_kernel`（`@triton.jit`，`fused_moe.py:299`）与 `fused_moe_kernel_gptq_awq`（:65），是默认 fused MoE GEMM；`fused_experts_impl`（:1656）编排。
- `csrc/libtorch_stable/moe/` — `moe_align_sum_kernels.cu`（`moe_align_block_size` + `moe_sum`，token 按 expert 对齐/归约）、`moe_permute_unpermute_op.cu`、`topk_softmax_kernels.cu` / `topk_softplus_sqrt_kernels.cu`（router top-k）、`marlin_moe_wna16/`（Marlin 量化 MoE）、`dsv3_router_gemm_*`（DeepSeek V3 router GEMM）。
- `csrc/libtorch_stable/fp32_router_gemm.cu`、`dsv3_fused_a_gemm.cu` — router 专用 GEMM。
- 量化 MoE 走 `cutlass_moe_mm` / `cutlass_w4a8_moe_mm` / `cutlass_fp4_group_mm` 等 CUTLASS 路径。

### 5.5 量化 kernel（`csrc/libtorch_stable/quantization/`）
<!-- tags: quantization, kernels, marlin, gptq, awq -->

- **FP8/INT8 (w8a8)**：`w8a8/fp8/`、`w8a8/int8/`、`w8a8/cutlass/`；`per_token_group_fp8_quant` / `per_token_group_quant_int8` / `dynamic_scaled_fp8_quant` / `static_scaled_fp8_quant` 等逐 token 量化。
- **Marlin**（`marlin/`）：`marlin_gemm`（W4A16 高速反量化 GEMM）、`marlin_int4_fp8_preprocess`、`gptq_marlin_repack` / `awq_marlin_repack`（权重重排）。
- **GPTQ**（`gptq/`）：`gptq_gemm`（`q_gemm.cu` + `qdq_{2,3,4,8}.cuh` 各 bit 宽反量化）。
- **AWQ**（`awq/`）：`awq_gemm` / `awq_dequantize`。
- **Machete**（`machete/`）：Hopper 混合精度 GEMM（`machete_mm`）。
- **FP4**（`fp4/`、`cutlass_w4a8/`）：NVFP4/MXFP4（`scaled_fp4_quant`、`cutlass_scaled_fp4_mm`、`mxfp4_experts_quant`）。
- **Hadamard**（`hadamard/`）：`hadacore_transform`（旋转，用于某些量化方案）。

### 5.6 其他算子
<!-- tags: kernels, rope, layernorm, activation, sampler -->

- **RoPE**：`csrc/libtorch_stable/pos_encoding_kernels.cu` — `rotary_embedding`（`apply_rotary_embedding`，支持 NeoX 与 interleaved）。
- **LayerNorm**：`csrc/libtorch_stable/layernorm_kernels.cu` — `rms_norm` / `fused_add_rms_norm`；`layernorm_quant_kernels.cu` — 融合量化（`rms_norm_dynamic_per_token_quant` / `rms_norm_static_fp8_quant` / `rms_norm_per_block_quant`）。
- **激活**：`csrc/libtorch_stable/activation_kernels.cu` — `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` / `relu_squared` 及各自量化融合版（`silu_and_mul_quant` / `silu_and_mul_nvfp4_quant` 等）。
- **采样**：`csrc/libtorch_stable/sampler.cu` — 批量采样 kernel；`topk.cu` / `cooperative_topk.cu` / `persistent_topk.cuh` — `topk_softmax` / `topk_sigmoid` / `top_k_per_row_{decode,prefill}`；`apply_repetition_penalties_`。
- **通信**：`custom_all_reduce.cu` / `custom_all_gather_reduce_scatter.cu` / `quickreduce/` — 自定义 all-reduce（小消息低延迟）。
- **SSM/Mamba**：`csrc/libtorch_stable/mamba/selective_scan_fwd.cu`（`selective_scan_fwd`，Mamba1）、`gdn/`（Gated DeltaNet）、`kimi_k3/`（Kimi K3 专用融合 kernel）。
- **模型专用融合**：`fused_qknorm_rope_kernel.cu`（`fused_qk_norm_rope`，QK-norm+RoPE 融合）、`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`、`fused_kimi_k3_mla_*`、`fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`。

---

## 6. Triton kernel 用途
<!-- tags: triton, kernels -->

Triton kernel 分布在三处：

1. **`vllm/v1/attention/ops/`** — attention 相关 Triton 算子：
   - `triton_unified_attention.py`（`kernel_unified_attention`）— TRITON_ATTN 的统一 prefill+decode kernel，支持 FP8/INT8/INT4 per-token-head KV 反量化（`_cast_kv_tile`）。
   - `triton_prefill_attention.py`（`context_attention_fwd`）、`triton_decode_attention.py`（`_fwd_kernel_stage1` 等）— 分离式 prefill/decode kernel。
   - `triton_reshape_and_cache_flash.py` — Triton 版 KV 写入。
   - `triton_merge_attn_states.py` — cascade attention 合并。
   - `triton_fp8_mqa_logits.py`、`int4_per_token_head.py` — FP8/INT4 KV 辅助。
   - `turboquant_soa/`、`triton_turboquant_*.py` — TurboQuant 压缩 KV 的 decode/store kernel。
   - `prefix_prefill.py`、`chunked_prefill_paged_decode.py`（`kernel_paged_attention_2d`）— 特殊 prefill 路径。
   - `mla/sparse_utils.py`、`mla/compressor_utils.py`、`mla/indexer.py` — sparse MLA 的 index 转换/compress/indexer kernel。
2. **`vllm/model_executor/layers/fused_moe/fused_moe.py`** — 默认 fused MoE GEMM（`fused_moe_kernel`）。
3. **`vllm/kernels/triton/`** — `qkv_padded_fp8_quant.py`（`quantize_fp8_pad_head_dim_triton`，QKV 对齐到 padded head dim 的 FP8 量化）。
4. **`vllm/kernels/helion/`** — 基于 Helion（Triton 高层 DSL）的 kernel：`per_token_group_fp8_quant`、`rms_norm_*`、`silu_and_mul_per_block_quant`、`fused_qk_norm_rope` 等，带 `ConfigManager` 做 per-GPU 自动调优配置选择。

辅助工具在 `vllm/triton_utils/`：`allocation.py`（`set_triton_allocator`，Triton 临时内存分配器）、`force_first_config.py`（强制首个 autotune config，避免运行时 autotune 抖动）、`tensor_descriptor.py`（TMA tensor descriptor 开关）。

---

## 7. 如何切换/指定 Attention Backend
<!-- tags: attention, backend, config, env, attention-config -->

### 7.1 CLI / 配置
<!-- tags: cli, 配置, attention-backend, attention-config, flags -->

- **`--attention-backend <NAME>`**（`arg_utils.py:975`）：全局指定 backend，取值即 `AttentionBackendEnum` 名（如 `FLASH_ATTN`、`FLASHINFER`、`TRITON_ATTN`、`FLASHMLA`、`TRITON_MLA`）。显式指定且不合法会直接报错。
- **`--attention-config` / `-ac`**（`arg_utils.py:1660`）：传 `AttentionConfig` 的 JSON/dict，可设任意字段，例如：
  - `--attention-config '{"backend": "FLASHINFER"}'`
  - `--attention-config '{"flash_attn_version": 3}'`
  - `--attention-config '{"use_trtllm_attention": true}'`
  - `--attention-config '{"mla_prefill_backend": "TRTLLM_RAGGED"}'`
  - `--attention-config '{"backend_per_kind": {"mla_attention": "FLASHINFER_MLA", "sliding_window_mla": "TRITON_MLA"}}'`
- **`--mamba-backend`**：SSM/线性层 backend（`MAMBA1`/`MAMBA2`/`GDN_ATTN`/`LINEAR`/`SHORT_CONV`）。
- 注意：`--attention-backend` 与 `attention_config.backend` 不能同时设（`arg_utils.py:2398` 会报错）。

### 7.2 `AttentionConfig` 关键字段（`vllm/config/attention.py`）
<!-- tags: attention-config, 字段, backend, mla-prefill, kv-dtype -->

| 字段 | 说明 |
|---|---|
| `backend` | 全局 backend（`AttentionBackendEnum` 或 None=auto）。 |
| `backend_per_kind` | 按 `KVCacheSpecKind`（`mla_attention`/`sliding_window_mla`/`full_attention`/`sliding_window`/`cross_attention`/`encoder_only_attention`）分别指定 backend，用于混合层模型。 |
| `flash_attn_version` | 强制 FA 版本（2/3/4），仅 FLASH_ATTN backend 有效。 |
| `use_trtllm_attention` | 强制开/关 FlashInfer 的 TRTLLM attention（None=auto）。 |
| `mla_prefill_backend` | MLA prefill 后端（`FLASH_ATTN`/`FLASHINFER`/`TRTLLM_RAGGED`/`TOKENSPEED_MLA`）。 |
| `use_prefill_decode_attention` | True 时 TRITON_ATTN 用分离 prefill/decode kernel 而非 unified。 |
| `flash_attn_max_num_splits_for_cuda_graph` | FA decode cudagraph 的 max num_splits（默认 32）。 |
| `disable_flashinfer_q_quantization` | FP8 KV 时不量化 Q。 |
| `use_prefill_query_quantization` | prefill 时量化 query。 |
| `sparse_mla_force_mqa` | 强制 sparse MLA 全走 `forward_mqa`。 |
| `indexer_kv_dtype` | sparse indexer K cache dtype（`auto`/`fp8`/`mxfp4`/`nvfp4`）。 |
| `flex_attn_block_m/n`、`flex_attn_q/kv_block_size` | FlexAttention 的 Triton tile 尺寸。 |

### 7.3 相关环境变量（`vllm/envs.py`）
<!-- tags: env-vars, 环境变量, kv-layout, flashinfer, rocm -->

- `VLLM_KV_CACHE_LAYOUT`（`NHD`/`HND`）— KV cache 物理布局（`envs.py:1781`）。
- `VLLM_BATCH_INVARIANT` — 批不变模式（影响 backend 选择，如 FlexAttention 默认 block 16；MLA/Mamba 需支持 batch invariance）。
- `VLLM_USE_FLASHINFER_SAMPLER`（默认 True）— 采样用 FlashInfer。
- `VLLM_USE_FLASHINFER_MOE_INT4` — FlashInfer INT4 MoE。
- `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` / `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` — FlashInfer autotune 缓存/跳过。
- `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`（默认 394MB）— FlashInfer workspace。
- `VLLM_FLASHINFER_ALLREDUCE_BACKEND`（`auto`/`trtllm`/`mnnvl`）。
- `VLLM_USE_TRITON_AWQ` — AWQ 用 Triton kernel。
- `VLLM_ROCM_USE_AITER` / `VLLM_ROCM_USE_AITER_MLA` / `VLLM_ROCM_USE_AITER_MHA` / `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` — ROCm AITER kernel 开关。
- 注：旧版 `VLLM_ATTENTION_BACKEND` 环境变量在 v1 已移除，改用 `--attention-backend` / `--attention-config`。

### 7.4 相关 CacheConfig 字段（`vllm/config/cache.py`）
<!-- tags: cache-config, block-size, kv-cache-dtype, 字段 -->

- `block_size`（`--block-size`）— KV 分页大小；须满足所选 backend 的 `get_supported_kernel_block_sizes()`（如 FA 须 %16、FlashMLA 固定 64）。显式设 `--block-size` 可能排除高优先级 backend（见 §2.1 warning）。
- `kv_cache_dtype`（`--kv-cache-dtype`）— `auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2`/`nvfp4`/`int8_per_token_head`/`turboquant_*` 等，直接影响 backend 选择与 kernel 路径。
- `kv_cache_dtype_skip_layers` — 指定层跳过量化 KV（混合 dtype）。

---

## 8. 关键文件清单
<!-- tags: files -->

**抽象与选择**
- `vllm/v1/attention/backend.py` — `AttentionBackend`/`AttentionMetadataBuilder`/`AttentionImpl`/`MLAAttentionImpl`/`CommonAttentionMetadata`/`AttentionCGSupport`。
- `vllm/v1/attention/selector.py` — `get_attn_backend`、`AttentionSelectorConfig`、`get_attn_spec_kind`。
- `vllm/v1/attention/backends/registry.py` — `AttentionBackendEnum`、`MambaAttentionBackendEnum`、`register_backend`。
- `vllm/platforms/cuda.py` — CUDA `get_attn_backend_cls` + `_get_backend_priorities`（:83）。
- `vllm/platforms/rocm.py` / `vllm/platforms/cpu.py` — ROCm/CPU 选择逻辑。
- `vllm/config/attention.py` — `AttentionConfig`。

**各 backend 实现**
- `vllm/v1/attention/backends/flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`flex_attention.py`、`turboquant_attn.py`、`hpc_attn.py`。
- `vllm/v1/attention/backends/mla/` — `flashmla.py`、`flashinfer_mla.py`、`cutlass_mla.py`、`triton_mla.py`、`flashattn_mla.py`、`flashmla_sparse.py`、`indexer.py`、`sparse_utils.py`、`prefill/`（`registry.py`/`selector.py`）。
- `vllm/v1/attention/backends/fa_utils.py` — FA 版本选择。

**模型层入口**
- `vllm/model_executor/layers/attention/attention.py` — `Attention` 层、`unified_attention_with_output`、`unified_kv_cache_update`。
- `vllm/model_executor/layers/attention/mla_attention.py` — `MLAAttention`、`MLACommonBackend`、`forward_impl`（mha/mqa 分路）。

**Triton 算子**
- `vllm/v1/attention/ops/` — `triton_unified_attention.py`、`triton_prefill_attention.py`、`triton_decode_attention.py`、`triton_reshape_and_cache_flash.py`、`triton_merge_attn_states.py`、`flashmla.py`（`vllm._flashmla_C` 封装）。
- `vllm/kernels/triton/`、`vllm/kernels/helion/`、`vllm/triton_utils/`。

**CUDA kernel（csrc/）**
- `csrc/libtorch_stable/torch_bindings.cpp` — 所有 `torch.ops._C.*` 注册。
- `csrc/libtorch_stable/attention/`（`merge_attn_states.cu`、`mla/sm100_cutlass_mla_kernel.cu`、`dcp_utils/`）。
- `csrc/libtorch_stable/cache_kernels.cu`、`cache_kernels_fused.cu`、`nvfp4_kv_cache_kernels.cu`。
- `csrc/libtorch_stable/moe/`、`csrc/libtorch_stable/quantization/{marlin,gptq,awq,w8a8,machete,fp4,hadamard}/`。
- `csrc/libtorch_stable/{pos_encoding_kernels,layernorm_kernels,layernorm_quant_kernels,activation_kernels,sampler,topk}.cu`。
- `csrc/attention/`（FasterTransformer 移植的通用 paged-attention 模板）、`csrc/rocm/attention.cu`。
- `vllm/vllm_flash_attn/flash_attn_interface.py` — FA2/3/4 封装。
