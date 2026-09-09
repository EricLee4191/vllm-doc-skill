# 量化与多硬件平台

> 基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（v1 引擎为默认 active engine）。本文聚焦**架构**与**部署/调优**，不逐行注释。
> 路径均相对仓库根 `/Users/baofeng/baofeng/github/vllm`。

vLLM 的量化体系分两条主线：
1. **量化方法层**（`vllm/model_executor/layers/quantization/`）——每种量化格式一个 `QuantizationConfig` 子类，负责识别 checkpoint、为每层（Linear / MoE / Attention-KV）产出 `QuantizeMethodBase`，method 再调用 **kernel 选择器**（`vllm/model_executor/kernels/linear/`）挑选具体 GEMM kernel。
2. **平台抽象层**（`vllm/platforms/`）——`Platform` 基类把硬件差异（device、dtype、通信后端、attention backend 优先级、显存查询、量化白名单）收敛到 `current_platform` 单例，模型代码只依赖抽象接口。

---

## 1. 量化方法全景
<!-- tags: quantization, fp8, awq, gptq, nvfp4, mxfp8, w8a8, w4a16, 量化 -->

### 1.1 方法注册表
<!-- tags: registry, 注册表, quantization-methods, get-config, oot -->

所有方法名定义在 `vllm/model_executor/layers/quantization/__init__.py:12` 的 `QuantizationMethods` Literal 中，`get_quantization_config(name)`（:108）做 name → `QuantizationConfig` 类的映射：

| `--quantization` 值 | Config 类 | 文件 |
|---|---|---|
| `fp8` | `Fp8Config` | `fp8.py` |
| `awq` / `auto_awq` / `awq_marlin` | `AutoAWQConfig` | `auto_awq.py` |
| `gptq` / `auto_gptq` / `gptq_marlin` | `AutoGPTQConfig` | `auto_gptq.py` |
| `modelopt` / `modelopt_fp4` / `modelopt_mxfp8` / `modelopt_mixed` | `ModelOptFp8Config` 等 | `modelopt.py` |
| `compressed-tensors` | `CompressedTensorsConfig` | `compressed_tensors/` |
| `mxfp4` / `gpt_oss_mxfp4` | `Mxfp4Config` / `GptOssMxfp4Config` | `mxfp4.py` |
| `torchao` | `TorchAOConfig` | `torchao.py` |
| `inc`（Intel Neural Compressor） | `INCConfig` | `inc/` |
| `quark`（NVIDIA Quark） | `QuarkConfig` | `quark/` |
| `humming` | `HummingConfig` | `humming.py` |
| `experts_int8`（旧名，等价 `int8_per_channel` MoE-only） | `ExpertsInt8Config` | `experts_int8.py` |
| `moe_wna16`（W4A16/W8A16 MoE 兜底） | `MoeWNA16Config` | `moe_wna16.py` |
| `online` + 在线量化 shorthand（见 1.4） | `OnlineQuantizationConfig` | `online/` |
| `deepseek_v4_fp8` | `DeepseekV4FP8Config` | `models/deepseek_v4.py` |

`DEPRECATED_QUANTIZATION_METHODS = ["fbgemm_fp8", "fp_quant"]`（`__init__.py:49`），使用需 `--allow-deprecated-quantization`。

**已迁出树（OOT plugin）**：`bitsandbytes`（INT8/INT4，`vllm-bnb-plugin`）和 `GGUF`（`vllm-gguf-plugin`，`vllm serve repo_id:Q4_K_M` 格式）在 v0.26 中不再是内置方法，通过 `register_quantization_config()`（`__init__.py:58`）插件机制注册；`docs/features/quantization/bnb.md`、`gguf.md` 有安装说明。

### 1.2 各方法要点与权衡
<!-- tags: quantization, fp8, awq, gptq, 权衡 -->

**FP8（`fp8.py`，`Fp8Config`，min capability 75）**
- 字段：`is_checkpoint_fp8_serialized`、`activation_scheme ∈ {"static","dynamic"}`、`ignored_layers`、`weight_block_size`（128x128 block-wise，DeepSeek 风格，仅支持 dynamic activation）、`store_dtype`。
- 两种形态：
  - **离线（checkpoint 已 FP8 序列化）**：`Fp8LinearMethod` / `Fp8MoEMethod`。per-tensor 或 per-channel 权重 scale + static/dynamic 激活 scale；block-wise 用 `weight_scale_inv`（128x128）。
  - **在线（BF16 权重加载时量化）**：`online/fp8.py` 的 `Fp8PerTensorOnlineLinearMethod` 等，`QuantizeMethodBase.uses_meta_device=True` 时权重先在 meta device 创建、逐层量化，降低加载峰值显存。
- kernel 选择：`Fp8LinearMethod.create_weights` 调 `init_fp8_linear_kernel(activation_quant_key, weight_quant_key, ...)`（`fp8.py:359`）。dynamic + cutlass 支持时激活用 **per-token** scale（`kFp8DynamicTokenSym`）性能更好；无 FP8 硬件的 GPU 自动回退 **Marlin** weight-only FP8 kernel（`fp8.py:259` 注释）。
- MoE：`select_fp8_moe_backend()` 按 (weight_key, activation_key) 选后端（triton / cutlass / deep_gemm / flashinfer_trtllm 等）。
- ROCm MI300/MI325 用 FNUZ 格式，加载时 `normalize_e4m3fn_to_e4m3fnuz()` 转换（`fp8.py:700`）。
- 精度/速度/显存：W8A8，权重显存减半、GEMM 走 FP8 tensor core（Hopper/Blackwell 2x 吞吐）；精度损失通常 <0.1%（per-tensor 略差于 per-block）。**H100/B200 上首选**。

**AWQ（`auto_awq.py`，`AutoAWQConfig`，min capability 75）**
- 只支持 4-bit（`TYPE_MAP = {4: uint4}`；8-bit AWQ 需走 Marlin 后端 `awq_marlin`）。weight-only（W4A16）：权重 INT4 打包进 int32，激活保持 FP16/BF16。
- `from_config` 读 `w_bit/bits`、`q_group_size/group_size`、`zero_point`、`lm_head`、`modules_to_not_convert`。
- kernel：CUDA 上优先 **Marlin**（`check_marlin_supported` + `check_marlin_supports_layer(allow_tile_padding=True)`，不满足回退 Triton `AutoAWQLinearMethod`）；MoE 优先 `AutoAWQMoEMethod`（Marlin MoE），不满足回退 `MoeWNA16Config`。CPU/XPU 上走 `AutoAWQMarlinLinearMethod`（内部 `choose_mp_linear_kernel` 选 `CPUWNA16LinearKernel` / `XPUwNa16LinearKernel`，加载时把 AWQ 的私有 bit 序转成标准 GPTQ 格式，`_convert_awq_to_standard_format`）。
- 权衡：显存约为 BF16 的 1/4，decode 是 memory-bound 所以提速明显；精度略低于 FP8（4-bit 有损），适合显存受限场景。

**GPTQ / GPTQModel（`auto_gptq.py`，`AutoGPTQConfig`，min capability 60）**
- 支持 4-bit 对称（`uint4b8`）与 8-bit 对称（`uint8b128`）；weight-only（W4A16/W8A16）。
- 额外字段：`desc_act`（act-order）、`dynamic`（GPTQModel 的 per-module 正则覆盖，`"+:"`/`"-:"` 前缀）、`modules_in_block_to_quantize`（autoround 标记）。
- `maybe_update_config` 会读 safetensors metadata 自动推断哪些层真的被量化了（:278）。
- kernel 同 AWQ：Marlin 优先，MoE 回退 `MoeWNA16Config`。
- 与 AWQ 的取舍：GPTQ 4-bit 精度通常略好于 AWQ 4-bit（逐层 Hessian 校准），两者都显著优于 BF16 的显存占用；GPTQ 支持 8-bit 是独有优势。

**ModelOpt（`modelopt.py`，NVIDIA TensorRT-Model-Optimizer 产物）**
- `ModelOptFp8Config`（min capability 80）：FP8 W8A8，支持 per-tensor / per-channel-per-tensor（`ModelOptFp8PcPtLinearMethod`）/ per-block weight-only（`ModelOptFp8PbWoLinearMethod`）三种 linear method；可带 KV cache 量化（`kv_cache_quant_method`）。
- `ModelOptNvFp4Config`（`modelopt_fp4`，min capability 75）：NVFP4（fp4_e2m1 + fp8 block scale，group_size=16），Blackwell 最优显存/吞吐组合；有 W4A16 变体 `ModelOptNvFp4W4A16LinearMethod`。
- `ModelOptMxFp8Config`（`modelopt_mxfp8`，min capability 80）：MXFP8（e8m0 scale，1x32 block），Marlin kernel 支持 SM80+。
- `ModelOptMixedPrecisionConfig`（`modelopt_mixed`）：混合精度（不同层不同 bit）。
- 识别方式：`override_quantization_method` 读 `hf_quant_config.json` 里的 `quant_algo`（FP8/NVFP4/MXFP8）。

**compressed-tensors（`compressed_tensors/`，min capability 70）**
- llm-compressor 的通用格式：一个 config 里按 layer 声明 scheme（`schemes/` 下有 W8A8-FP8、W4A16-INT、W8A16、MXFP8、NVFP4 等），`CompressedTensorsConfig.from_config` 解析 `format`/`config_groups`。
- 是 **KV cache 量化 scale 校准**（per-head scale，llm-compressor 路径）的主要载体；也支持 embedding 量化（`compressed_tensors_embedding.py`）与 MoE（`compressed_tensors_moe/`）。

**MXFP4（`mxfp4.py`，min capability 80）**
- OCP MX 格式：fp4_e2m1 数据 + e8m0 per-1x32 scale。`Mxfp4Config` 的 linear 层目前 fallback 到 `UnquantizedLinearMethod`（:91 注释 "MXFP4 linear layer is not implemented"），**主要价值在 MoE**（`Mxfp4MoEMethod`）。
- `GptOssMxfp4Config`（`gpt_oss_mxfp4`）：GPT-OSS 模型的 MXFP4 MoE 专用路径。

**torchao / INC / Quark / Humming**
- `TorchAOConfig`（min capability 75）：读 HF config 里的 torchao `config_dict`，支持 int8/int4/fp8 等多种 torchao recipe。
- `INCConfig`（min capability 60）：Intel Neural Compressor，XPU/CPU 常见（W8A8、W4A16 等）。
- `QuarkConfig`（min capability 70）：NVIDIA Quark 工具链产物。
- `HummingConfig`（min capability 75）：Neural Magic 的混合精度格式（per-layer 不同量化 schema），linear 走 `HummingLinearKernel`，MoE 走 `HummingMoEMethod`。

**在线量化（online/，`OnlineQuantizationConfig`，min capability 75）**
- 无需预量化 checkpoint：加载 BF16/FP16 权重时逐层量化。`--quantization` 的 shorthand 在 `vllm/config/quantization.py:116` 的 `_ONLINE_SHORTHANDS`：

| shorthand | 权重 recipe | 激活 recipe |
|---|---|---|
| `fp8_per_tensor` | fp8_e4m3 + fp32 per-tensor scale | 同左（Ada/Hopper 上 linear 用 per-token） |
| `fp8_per_block` | fp8 + fp32 per-128x128 scale | fp8 + per-1x128 scale |
| `fp8_per_channel` | fp8 + per-channel scale | dynamic per-token |
| `mxfp8` | fp8 + e8m0 per-1x32 | 同左（W8A8 需 SM100+，否则 W8A16 fallback） |
| `mxfp4` | fp4 + e8m0 per-1x32 | linear 视 backend（可能 BF16），MoE 为 fp4 |
| `int8_per_channel_weight_only` | INT8 per-channel（仅 MoE） | 不量化 |
| `nvfp4_per_token` | NVFP4（仅 MoE，Blackwell + FlashInfer TRTLLM） | dynamic per-token |

- 细粒度控制：`--quantization-config '{"linear":{"weight":"fp8_per_block_static","activation":"fp8_per_token"},"moe":{...},"ignore":[...]}'`，名字来自 `QUANT_KEY_NAMES`（`config/quantization.py:26`）。`resolve_quantization_config()`（:158）合并 shorthand 与显式 config（显式优先）。
- 调度表：`online/base.py:67` 的 `_ONLINE_LINEAR_METHODS` / `_ONLINE_MOE_METHODS` 按 `QuantKey` 分发到具体 method。

**KV cache 量化（与权重量化正交）**
- `CacheConfig.cache_dtype`（`vllm/config/cache.py:88`，CLI `--kv-cache-dtype`）：`auto` / `fp8` / `fp8_e4m3` / `fp8_e5m2` / `fp8_inc` / `fp8_ds_mla` / `int8_per_token_head` / `fp8_per_token_head` / `int4_per_token_head` / `nvfp4` / `nvfp4_4over6` / `turboquant_k8v4` / `turboquant_4bit_nc` / `turboquant_k3v4_nc` / `turboquant_3bit_nc`。
- per-tensor scale 从 checkpoint 加载：`BaseKVCacheMethod`（`quantization/kv_cache.py:42`）在 `Attention` 层上注册 `q_scale/k_scale/v_scale/prob_scale`（`KVCacheScaleParameter`，初始 -1.0 哨兵值）；`Fp8KVCacheMethod`、`ModelOptKVCacheMethod` 等继承它。scale 名字映射由 `QuantizationConfig.get_cache_scale_mapper()`（`base_config.py:195`）统一处理（`.kv_scale` → `.attn.k_scale` 等）。
- per-token-head scale（`*_per_token_head`）在 kernel 写 cache 时动态计算，`BaseKVCacheMethod.process_weights_after_loading` 直接置 1.0 并删除参数（`kv_cache.py:74`）。
- `kv_cache_dtype_skip_layers`（`cache.py:134`）：按层跳过 KV 量化（首尾层保持高精度，`Platform._align_heterogeneous_kv_block_size` 负责 block 对齐）。
- TurboQuant（`quantization/turboquant/`）：Hadamard 旋转 + Lloyd-Max 标量量化 K、均匀量化 V，3-4 bit KV。

### 1.3 权重量化 vs 激活量化（WNA16 vs W8A8）
<!-- tags: wna16, w8a8, 权重量化, 激活量化, 对比 -->

- **Weight-only（W4A16/W8A16，AWQ/GPTQ/Marlin 系）**：权重低比特、激活 16-bit。GEMM 时先 dequant 权重。decode（小 batch、memory-bound）下显存减半/四分之一直接转化为吞吐；prefill（compute-bound）提速有限。精度：4-bit 有可感知损失，8-bit 接近无损。
- **W8A8（FP8/INT8）**：权重和激活都 8-bit，GEMM 全程低精度，prefill+decode 都提速（Hopper/Blackwell FP8 tensor core 2x FLOPS）。需要激活 scale（static 需校准 / dynamic 在线算 amax）。精度通常优于 4-bit weight-only。
- **W4A4/W4A8（NVFP4/MXFP4）**：Blackwell 时代的最优解，显存与算力双收益，但硬件门槛高（SM100+ 或特定 kernel）。

### 1.4 量化如何接入模型
<!-- tags: quantization, 接入, 识别, kernel-selection, 生命周期 -->

1. **识别**：`ModelConfig._verify_quantization()`（`vllm/config/model.py:1245`）读 HF `config.json` 的 `quantization_config.quant_method`，按 `overrides` 优先级列表（:1258，`auto_gptq` > `gptq` > `gptq_marlin` > `auto_awq` > `awq` > `awq_marlin` > `inc` > `moe_wna16` > `modelopt*` > `mxfp8` > `mxfp4` > `gpt_oss_mxfp4` > `deepseek_v4_fp8` > `humming`）逐个调 `override_quantization_method()` 探测；用户 `--quantization` 与 checkpoint 不一致直接报错（:1321）。最后 `current_platform.verify_quantization()` 对照平台白名单（`interface.py:962`）。
2. **每层绑定**：`LinearBase.__init__`（`vllm/model_executor/layers/linear.py:266`）调 `quant_config.get_quant_method(self, prefix)` 得到 `LinearMethodBase`；`RoutedExperts`（MoE）同理得到 `FusedMoEMethodBase`；`Attention` 层得到 `BaseKVCacheMethod` 子类。
3. **生命周期**：`create_weights()`（注册 `weight`/`weight_scale`/`input_scale` 等参数并**选 kernel**）→ 权重加载（`weight_loader` 从 checkpoint 灌入，`packed_modules_mapping` 处理 QKV/gate_up 融合）→ `process_weights_after_loading()`（转置/重打包/shuffle 成 kernel 期望的布局）→ `apply()`（forward 时调 kernel）。
4. **kernel 选择**（`vllm/model_executor/kernels/linear/__init__.py`）：
   - `init_fp8_linear_kernel()`（:666）/ `init_int8_linear_kernel()`（:739）：W8A8 走 `ScaledMMLinearKernel` 族（`scaled_mm/` 下 `cutlass.py`、`deep_gemm.py`、`flashinfer.py`、`marlin.py`、`triton.py`、`pytorch.py`、`aiter.py`、`rocm.py`、`xpu.py`、`cpu.py`、`b12x.py`…），按 `QuantKey`（weight/activation 的 dtype+scale group shape，定义在 `quantization/utils/quant_utils.py:168`）+ 平台 + capability 过滤，`choose_scaled_mm_linear_kernel` 取第一个 `is_supported() and can_implement()` 的。
   - `choose_mp_linear_kernel()`（:775）：weight-only 走 `MPLinearKernel` 族（`mixed_precision/` 下 `marlin.py`、`machete.py`、`exllama.py`、`conch.py`、`triton_w4a16.py`、`rdna3_w4a16.py`、`cpu.py`、`xpu.py`、`zentorch.py`…），按 `_POSSIBLE_KERNELS[platform]` 顺序 + `get_min_capability()` + `can_implement()` 选择。
   - 可用 `--linear-backend` / `--moe-backend` 强制指定（`vllm/config/kernel.py:168` 的 `KernelConfig`，选项清单见该文件 docstring），`VLLM_DISABLED_KERNELS` 环境变量可禁用特定 kernel 类。
   - MoE kernel 由 `vllm/model_executor/layers/fused_moe/oracle/`（如 `oracle/fp8.py` 的 `select_fp8_moe_backend`）按同样的 (quant_key, 平台) 逻辑选择。

---

## 2. 平台抽象（`vllm/platforms/`）
<!-- tags: platform, nvidia, rocm, cpu, hpu, 平台, 硬件探测 -->

### 2.1 Platform 基类（`interface.py:134`）
<!-- tags: platform, 基类, 能力, device-id, config-hooks -->

`Platform` 是"硬件能力 + 默认值"的单一入口，关键成员：

- **类属性**：`_enum`（`PlatformEnum`: CUDA/ROCM/TPU/XPU/CPU/OOT/UNSPECIFIED）、`device_name`/`device_type`、`dispatch_key`（PyTorch dispatch key）、`ray_device_key`、`dist_backend`（NCCL/gloo/xccl）、`device_control_env_var`（`CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK` / …）、`supported_quantization: list[str]`（**量化白名单**，空 = 不限制）、`supported_dtypes`（第一个是 "auto" dtype 的 fallback）。
- **设备 ID 三命名空间**（:285-341）：logical（vLLM local rank）→ physical（NVML 全局 ID）→ visible（`CUDA_VISIBLE_DEVICES` 重映射后的 ordinal）。`device_id_to_physical_device_id` / `logical_device_id_to_visible_device_id` / `visible_device_id_to_physical_device_id` 显式转换，避免 `CUDA_VISIBLE_DEVICES` 陷阱。
- **能力查询**：`get_device_capability()` → `DeviceCapability(major, minor)`（`to_int()` 得 80/89/90/100…）；`has_device_capability(80)` 是量化/kernel 门槛判断的基础；`supports_fp8()` / `supports_mx()` / `is_fp8_fnuz()` / `fp8_dtype()`（OCP vs FNUZ FP8）；`num_compute_units()`（SM/CU/EU 数）。
- **attention backend 选择**：`get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)` 返回 backend 类路径；`get_supported_vit_attn_backends()` 给 ViT 用。
- **config 钩子**（在 `VllmConfig` 初始化时被调用）：
  - `apply_config_platform_defaults(vllm_config)`：平台默认值（如 ROCm 注入 AITER custom ops）。
  - `check_and_update_config(vllm_config)`：兼容性检查/修正（如 CPU 强制 `block_size=128`、XPU 禁用 fusion pass）。
  - `update_block_size_for_backend(vllm_config)`（:609）：按 backend 的 `get_preferred_block_size()` 设 `cache_config.block_size`，并对 hybrid（attention+mamba）与异构 KV dtype 做 page 对齐。
  - `verify_quantization(quant)`（:962）：不在 `supported_quantization` 白名单则报错。
- **其他**：`get_device_total_memory()`、`get_current_memory_usage()`（显存 profiling 用）、`get_device_communicator_cls()`（NCCL/UCX 通信器）、`use_custom_allreduce()`、`inference_mode()`（TPU 回退 `no_grad`）、`is_sleep_mode_available()`（CUDA/ROCm/XPU）、`stateless_init_device_torch_dist_pg()`（无状态初始化 process group，供 Ray 等场景）。

### 2.2 平台探测与 `current_platform`（`platforms/__init__.py`）
<!-- tags: platform, 探测, current-platform, 单例, oot -->

- 内置探测函数：`cuda_platform_plugin()`（pynvml 查 GPU 数，排除 cpu build，Jetson 特判）、`rocm_platform_plugin()`（amdsmi）、`xpu_platform_plugin()`（`torch.xpu.is_available()` + xccl）、`cpu_platform_plugin()`（`VLLM_TARGET_DEVICE=="cpu"` 或 cpu build 或 macOS；AMD Zen + AVX-512 + zentorch 时选 `ZenCpuPlatform`）、`tpu_platform_plugin()`（`VLLM_TPU_USING_PATHWAYS` 或 libtpu）。
- `resolve_current_platform_cls_qualname()`（:219）：`VLLM_TARGET_DEVICE=cpu` 时 CPU 优先；否则跑所有 builtin + OOT 插件（entry point group `PLATFORM_PLUGINS_GROUP`），**只允许一个激活**，否则 RuntimeError；都没有则 `UnspecifiedPlatform`。
- `current_platform` 是模块级 lazy 单例（`__getattr__`，:278），首次访问才解析，保证 OOT 插件先加载。
- OOT 平台通过插件继承 `Platform`，可覆盖 `import_ir_kernels()`、`pre_register_and_update()`（注册自定义量化 config 等）。

### 2.3 各平台要点
<!-- tags: platform, cuda, rocm, cpu, xpu -->

**CUDA（`cuda.py`）**
- `CudaPlatformBase`（:208）：`dist_backend="nccl"`，`device_control_env_var="CUDA_VISIBLE_DEVICES"`，`use_custom_allreduce=True`，`opaque_attention_op=True`，CUDA Graph wrapper（`CUDAGraphWrapper`）。`CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform`（:1028）——NVML 版可无状态查显存/卡名（不初始化 CUDA context）。
- `supported_dtypes`（:237）：capability ≥80 → `[bf16, fp16, fp32]`；60-79 → `[fp16, fp32]`（无 bf16）。
- `supports_fp8()` = capability ≥89（Ada 起）。
- attention backend 优先级 `_get_backend_priorities()`（:83）：
  - 非 MLA：SM100（Blackwell）causal → `[FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]`；其他 → `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]`。
  - MLA：SM100 → `[FLASHINFER_MLA, TOKENSPEED_MLA, CUTLASS_MLA, FLASH_ATTN_MLA, FLASHMLA, TRITON_MLA, *sparse]`（FP8 KV 时 FlashInfer 优先）；SM120 → `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`；其他 → `[FLASH_ATTN_MLA, FLASHMLA, ...]`。
  - 每个候选调 `validate_configuration()` 过滤（如 block_size 不兼容），选优先级最高者；`--attention-backend` 可强制。
- `check_and_update_config`（:312）：`worker_cls` 默认 `vllm.v1.worker.gpu_worker.Worker`；WSL2 + `--cpu-offload-gb` + cudagraph 的 pinned memory 警告。
- 部署要点：FP8 需 SM89+（`torch._scaled_mm` 限制 e4m3fn）；DeepGEMM（`VLLM_USE_DEEP_GEMM`，默认开）用于 Hopper FP8 block GEMM；`VLLM_BATCH_INVARIANT=1` 时 FP8 linear 走 BF16 dequant 路径保证可复现（`fp8.py:426`）。

**ROCm（`rocm.py`，`RocmPlatform` :498）**
- `device_type="cuda"`（HIP 复用 torch.cuda API）、`dispatch_key="CUDA"`、`dist_backend="nccl"`（RCCL）、`device_control_env_var="CUDA_VISIBLE_DEVICES"`（也认 `ROCR_VISIBLE_DEVICES` 的 ray noset 变量）。
- **量化白名单**（:513）：`awq/auto_awq/awq_marlin, gptq/auto_gptq, fp8, deepseek_v4_fp8, compressed-tensors, fbgemm_fp8, inc, quark, mxfp4, mxfp8, torchao, modelopt*, fp8_per_tensor/per_block/per_channel, online, gpt_oss_mxfp4`——**不在名单的量化直接拒绝**（如 humming、moe_wna16）。
- 架构探测：`_GCN_ARCH`（amdsmi 查 gfx 号），`on_cdna()`（gfx9*/gfx1250）、`on_rdna4()`（gfx1200/1201）、`_ON_MI3XX`（gfx942/950）。
- `supports_fp8()` = CDNA 或 RDNA4；`is_fp8_fnuz()` = gfx94（MI300 系用 FNUZ，`fp8_dtype()` 返回 `float8_e4m3fnuz`）；`supports_mx()` = gfx95/gfx1250；`use_custom_allreduce()` 仅 MI300 系（gfx94/95）。
- attention 优先级（:459）：MLA → `[ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]`；普通 → `[ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN, TURBOQUANT]`（按 AITER 可用性裁剪）。
- `apply_config_platform_defaults`（:866）：AITer 开启时自动加 `+quant_fp8`、`+grouped_topk`、`+sparse_attn_indexer` custom ops。
- `verify_quantization`（:940）：AWQ 在 ROCm 上强制 `VLLM_USE_TRITON_AWQ=1`（Marlin 不可用）。
- 部署要点：大量 `VLLM_ROCM_USE_AITER_*` 开关（`VLLM_ROCM_USE_AITER_LINEAR/MOE/MHA/MLA/FP8BMM/FP4BMM/...`，默认多为 True）；`VLLM_ROCM_FP8_PADDING`/`VLLM_ROCM_MOE_PADDING`（FP8 对齐 padding）；DCP/PCP 与 full cudagraph 不兼容会自动降为 PIECEWISE（:911）。

**CPU（`cpu.py`，`CpuPlatform` :43）**
- `dist_backend="gloo"`，`inference_mode()` 回退 `torch.no_grad()`，`is_pin_memory_available()=False`，`support_hybrid_kv_cache()=True`。
- `supported_dtypes`（:52）：按 CPU 架构（x86/ARM/POWERPC/RISCV）与 macOS ARM BF16 特性探测。
- attention：只有 `CPU_ATTN`；MLA 模型在 x86 + AMX tile 时用 `AMX_MLA`，否则 `CPU_MLA`（参考实现，且强制 `block_size=16`，:175；非 AMX MLA 还强制关 chunked prefill 和 prefix caching，:378）。
- `check_and_update_config`（:143）：默认 `block_size=128`（非 32 倍数会警告）；`worker_cls` 默认 `vllm.v1.worker.cpu_worker.CPUWorker`；`VLLM_ENABLE_V1_MULTIPROCESSING=1` 时强制 `mp` executor（OMP 线程绑定需要）；`VLLM_CPU_KVCACHE_SPACE`（GB）可指定 KV cache 空间；自动 `LD_PRELOAD` libgomp/libtcmalloc；AVX-512BF16 时 SSM conv state 用 SD layout。
- 量化：无白名单（全量方法可用），但实际 kernel 由 `choose_mp_linear_kernel` 按平台过滤——AWQ/GPTQ 走 `CPUWNA16LinearKernel`（`mixed_precision/cpu.py:20`，要求 group_size 偶数、input size 32 倍数）；`VLLM_CPU_INT4_W4A8`（默认 True）启用 INT4 W4A8。
- 部署要点：`VLLM_CPU_OMP_THREADS_BIND`（线程绑核，默认 auto）、`VLLM_CPU_NUM_OF_RESERVED_CPU`、`VLLM_CPU_ATTN_SPLIT_KV`（默认 True）；NUMA 拓扑发现（`discover_numa_topology`）供 KV transfer 预留核。

**XPU（`xpu.py`，`XPUPlatform` :103）**
- `dist_backend="xccl"`（oneCCL），`device_control_env_var="ZE_AFFINITY_MASK"`，`ray_device_key="GPU"`。
- **量化白名单**（:113）：`awq, gptq, auto_awq, auto_gptq, inc, fp8, deepseek_v4_fp8, mxfp4, mxfp8, fp8_per_tensor, fp8_per_block, online, gpt_oss_mxfp4, modelopt, compressed-tensors`。
- attention（:142）：turboquant KV → `TURBOQUANT`；sparse → `XPU_MLA_SPARSE`；MLA → `TRITON_MLA`；默认 `FLASH_ATTN`，fp32/mm-prefix 回退 `TRITON_ATTN`。
- `check_and_update_config`（:283）：XPU Graph 实验性（需 `VLLM_XPU_ENABLE_XPU_GRAPH=1` 且 PyTorch 支持，仅单卡）；禁用多个 fusion pass（`fuse_gemm_comms`、`fuse_allreduce_rms`、`fuse_attn_quant` 等）；UVA offload 时关 Inductor static launcher；`worker_cls` 默认 `vllm.v1.worker.xpu_worker.XPUWorker`；强制 `spawn` 多进程、`UCX_MEMTYPE_CACHE=n`、`shutdown_timeout=5`。
- FP8 linear 默认 **W8A16**（weight-only），`--linear-backend xpu` 强制 W8A8，`--linear-backend xpu_woq` 显式 W8A16（`docs/features/quantization/online.md`）。
- GDN 模型 block_size 需 64 倍数（`update_block_size_for_backend`，:387）。

**TPU（`tpu.py`）**
- 仅 10 行：依赖外部 `tpu_inference` 包（`TpuPlatform` 从 `tpu_inference.platforms` 导入）；`VLLM_TPU_USING_PATHWAYS=1` 时走 Pathways 代理（`tpu_inference.platforms.tpu_platform.TpuPlatform`）。`uses_host_device_handling()=True`（`DeviceConfig` 把 device 置 None）。量化/attention 细节都在 OOT 包里。

**DeviceConfig（`vllm/config/device.py`）**
- `device: "auto"`（已 deprecated，自动从 `current_platform.device_type` 推断，`__post_init__` :49）；`device_type` 是 init=False 字段。

---

## 3. 不同硬件部署要点速查
<!-- tags: hardware, deployment, h100, a100, b200, consumer-gpu, 5090 -->

| 场景 | 关键选择 |
|---|---|
| **NVIDIA H100/B200 大模型** | FP8（`fp8` checkpoint 或 `--quantization fp8_per_tensor/fp8_per_block` 在线量化）；B200 上 NVFP4（`modelopt_fp4`）/MXFP4 MoE 更优；`--kv-cache-dtype fp8` 省 KV 显存；attention 默认 FlashAttention/FlashInfer |
| **NVIDIA A100（SM80）** | 无 FP8 硬件：FP8 checkpoint 自动走 Marlin weight-only 路径；首选 AWQ/GPTQ 4-bit 或 INT8；`torch._scaled_mm` FP8 需 SM89+ |
| **显存受限（单卡跑大模型）** | AWQ/GPTQ 4-bit（显存 ~1/4）；bitsandbytes 4-bit（OOT，无需校准）；GGUF Q4_K_M（OOT）；`--gpu-memory-utilization` 调 KV 预算 |
| **AMD MI300X/MI325X（CDNA3）** | FP8 用 FNUZ 格式（自动转换）；AITer kernel 默认开（`VLLM_ROCM_USE_AITER_*`）；custom allreduce 可用；量化白名单见 2.3 |
| **AMD MI355X（CDNA4, gfx950）** | `supports_mx()=True`（MXFP8/MXFP4）；custom allreduce 可用 |
| **AMD RDNA4（gfx1200）** | `supports_fp8()=True`（OCP 格式）；attention 走 AITER unified / ROCm_ATTN |
| **CPU-only** | `VLLM_TARGET_DEVICE=cpu`（或 cpu build）；AWQ/GPTQ 4-bit 走 CPUWNA16；`VLLM_CPU_OMP_THREADS_BIND` 绑核；`VLLM_CPU_KVCACHE_SPACE` 控 KV；MLA 模型注意 block_size=16 限制 |
| **Intel XPU（Gaudi 之外的 Arc/Data Center GPU）** | `--linear-backend xpu`（W8A8）或默认 W8A16；INC 量化 checkpoint 原生支持；`VLLM_XPU_ENABLE_XPU_GRAPH=1` 开实验性 graph；xccl 通信 |
| **TPU** | 装 `tpu_inference`；`VLLM_TPU_USING_PATHWAYS` 走 Pathways；细节在 OOT 包 |

---

## 4. 配置/调优旋钮
<!-- tags: tuning, knobs, quantization, flags -->

**CLI（`vllm/engine/arg_utils.py`）**
- `--quantization / -q`：方法名（含 online shorthand）；`--quantization-config`：JSON 细粒度 spec（`{linear:{weight,activation}, moe:{...}, ignore:[...]}`）；`--allow-deprecated-quantization`。
- `--kv-cache-dtype`：KV cache 精度（fp8 系 / turboquant 系 / per_token_head 系 / nvfp4）。
- `--dtype`：权重/激活 dtype（`auto`/`half`/`bfloat16`/`float32`；AWQ 官方推荐 `half`）。
- `--attention-backend`：强制 attention backend（`AttentionBackendEnum`）；`--linear-backend`、`--moe-backend`：强制 GEMM/MoE kernel 后端（选项清单见 `vllm/config/kernel.py:150-240`）。
- `--gpu-memory-utilization`（默认 0.92）、`--block-size`（KV block，默认 16，平台/backend 会自动调整）。

**环境变量（`vllm/envs.py`，节选）**
- 通用量化：`VLLM_BATCH_INVARIANT`（可复现模式，禁用 Marlin 等）、`VLLM_DISABLED_KERNELS`（逗号分隔的 kernel 类名黑名单）、`VLLM_MARLIN_INPUT_DTYPE`（`int8`/`fp8`）、`VLLM_MARLIN_USE_ATOMIC_ADD`、`VLLM_USE_DEEP_GEMM` / `VLLM_MOE_USE_DEEP_GEMM` / `VLLM_USE_DEEP_GEMM_E8M0`（默认 True，Hopper FP8 block GEMM）、`VLLM_USE_TRITON_AWQ`（ROCm 上 AWQ 自动置 1）、`VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`、`VLLM_HUMMING_*`、`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`。
- 平台选择：`VLLM_TARGET_DEVICE`（`cuda`/`cpu`，显式 CPU 时权威）、`VLLM_TPU_USING_PATHWAYS`。
- ROCm：`VLLM_ROCM_USE_AITER`（总开关）及 `VLLM_ROCM_USE_AITER_{LINEAR,MOE,MHA,MLA,FP8BMM,FP4BMM,UNIFIED_ATTENTION,...}`、`VLLM_ROCM_FP8_PADDING`、`VLLM_ROCM_MOE_PADDING`、`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`。
- CPU：`VLLM_CPU_OMP_THREADS_BIND`、`VLLM_CPU_NUM_OF_RESERVED_CPU`、`VLLM_CPU_KVCACHE_SPACE`（GB）、`VLLM_CPU_ATTN_SPLIT_KV`、`VLLM_CPU_INT4_W4A8`、`VLLM_ZENTORCH_WEIGHT_PREPACK`（Zen CPU）。
- XPU：`VLLM_XPU_ENABLE_XPU_GRAPH`、`VLLM_XPU_USE_SAMPLER_KERNEL`。
- 设备控制：`CUDA_VISIBLE_DEVICES`（CUDA/ROCm 共用）、`ZE_AFFINITY_MASK`（XPU）、`RAY_EXPERIMENTAL_NOSET_*`（Ray 场景）。

**checkpoint 侧（config.json 的 `quantization_config`）**
- 通用键：`quant_method`（识别入口）、`activation_scheme`（FP8 static/dynamic）、`weight_block_size`、`ignored_layers`/`modules_to_not_convert`（跳过层）、`bits`/`group_size`/`desc_act`/`sym`（GPTQ）、`w_bit`/`q_group_size`/`zero_point`（AWQ）、`lm_head`（是否量化 lm_head）。
- 附加文件：GPTQ/AWQ 的 `quantize_config.json`（`get_config_filenames()` 声明）；ModelOpt 的 `hf_quant_config.json`。

---

## 5. 关键文件
<!-- tags: files -->

| 文件 | 作用 |
|---|---|
| `vllm/model_executor/layers/quantization/__init__.py` | 方法注册表、`get_quantization_config`、`register_quantization_config`（OOT 插件入口） |
| `vllm/model_executor/layers/quantization/base_config.py` | `QuantizationConfig` / `QuantizeMethodBase` 抽象、KV scale mapper |
| `vllm/config/quantization.py` | 在线量化 `QuantizationConfigArgs`、`QUANT_KEY_NAMES`、shorthand 解析 |
| `vllm/config/model.py`（:1245 `_verify_quantization`） | 量化方法识别/override/校验 |
| `vllm/config/cache.py`（:19 `CacheDType`） | KV cache 量化 dtype 枚举 |
| `vllm/config/kernel.py`（:168 `KernelConfig`） | `linear_backend` / `moe_backend` 选项 |
| `vllm/config/device.py` | `DeviceConfig`（device 自动推断） |
| `vllm/model_executor/layers/quantization/fp8.py` | FP8 全路径（linear/MoE/KV） |
| `vllm/model_executor/layers/quantization/online/` | 在线量化（fp8/int8/mxfp4/mxfp8/nvfp4） |
| `vllm/model_executor/layers/quantization/{auto_awq,auto_gptq,modelopt,mxfp4,compressed_tensors,torchao,inc,quark,humming,moe_wna16,experts_int8,kv_cache,turboquant}` | 各量化方法实现 |
| `vllm/model_executor/layers/quantization/utils/quant_utils.py` | `QuantKey`/`GroupShape`（量化方案的形式化描述） |
| `vllm/model_executor/kernels/linear/`（`scaled_mm/`、`mixed_precision/`、`mxfp4/`、`mxfp8/`、`nvfp4/`） | W8A8 与 weight-only GEMM kernel 族 + 选择器 |
| `vllm/model_executor/layers/fused_moe/oracle/` | MoE kernel 选择（fp8/int_wna16 等） |
| `vllm/model_executor/layers/linear.py`（:266）、`layers/attention/attention.py` | 量化 method 与层绑定 |
| `vllm/platforms/interface.py` | `Platform` 基类、`DeviceCapability`、设备 ID 映射 |
| `vllm/platforms/__init__.py` | 平台探测、`current_platform` lazy 单例 |
| `vllm/platforms/{cuda,rocm,cpu,xpu,tpu,zen_cpu}.py` | 各平台实现 |
| `vllm/envs.py` | 全部环境变量定义 |
| `docs/features/quantization/`（README、online.md、bnb.md、gguf.md、quantized_kvcache.md、gptqmodel.md…） | 用户文档 |
