# Depth Anything 3 migration evaluation

**Status: evaluation branch only. Production model remains Depth Anything V2.** This branch records repository evidence and provides a reproducible ONNX benchmark harness. It intentionally does not change the shipped model URL or checksum: no project-owned, pinned DA3 ONNX artifact with verified checksum and output-contract tests has yet been established.

## Recommendation

Evaluate **DA3-SMALL first**. The official DA3 model card lists 80M parameters and Apache-2.0 for DA3-SMALL, 120M / Apache-2.0 for DA3-BASE, and 350M / CC BY-NC 4.0 for DA3-LARGE. LARGE is not a reasonable default for a distributable editor without a separate legal review. BASE is a useful challenger if a matched-image evaluation shows that SMALL's quality is insufficient.

A third-party ONNX export repository reports a 96.1 MB DA3-SMALL ONNX file and CPU timings on an Apple M4: median 92.49 ms at 224², 278.13 ms at 392², and 504.01 ms at 504². These are **reference measurements from that repository**, not RapidRAW measurements, and cannot be compared directly to DA2 without running both in the same harness, on the same machine and image set. The current RapidRAW V2 file is 99,373,606 bytes (reported by the model host, roughly 99.4 MB). No DA3 GPU, VRAM, or RapidRAW end-to-end timings are claimed here.

Sources:
- Official DA3 model zoo / licenses / parameter counts: https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards
- Reference ONNX exporter, input/output contract, and M4 CPU table: https://github.com/devin-lai/Depth-Anything-3-Onnx
- Current V2 artifact and SHA-256: https://huggingface.co/CyberTimon/RapidRAW-Models/blob/main/depth_anything_v2_vits.onnx

## Repository dependency map (verified on `main` at 56aa54b1d16afab7f3ad53a32845c093545704b3)

| Concern | File / symbol | Evidence and migration relevance |
|---|---|---|
| Model URL, filename, input size, checksum | `src-tauri/src/ai_processing.rs:58-61` (`DEPTH_URL`, `DEPTH_FILENAME`, `DEPTH_INPUT_SIZE`, `DEPTH_SHA256`) | Replace only after hosting an immutable, verified DA3 export and pinning its SHA-256. |
| Cache path and download verification | `src-tauri/src/ai_processing.rs:316-451` (`get_models_dir`, `download_model`, `verify_sha256`, `download_and_verify_model`) | Stores models under Tauri's app-data `models` directory; downloads to a temporary file, atomically persists it, and verifies SHA-256. Retain this safety contract. |
| Session initialization / in-memory reuse | `src-tauri/src/ai_processing.rs:454-555` (`get_or_init_ai_models`) and `get_or_init_depth_model` | The evaluation branch separates the depth session from the SAM/foreground/sky bundle; a depth request no longer downloads and loads those unrelated models. |
| ONNX Runtime package / loading mode | `src-tauri/Cargo.toml:39` (`ort = 2.0.0-rc.10`, `ndarray`, `load-dynamic`) | Reuse ORT; avoid adding a second inference stack. Exported graph operators must be tested against the actual bundled runtime. |
| Runtime binary per target | `src-tauri/build.rs:71-179` | Selects bundled ORT by OS/architecture and verifies its checksum. This is not an execution-provider matrix; current model inference uses the default session builder. |
| Preprocessing, ONNX run, output normalization | `src-tauri/src/ai_processing.rs:1676-1757` (`run_depth_anything_model`) | Currently resize-to-fit 518², center-paste into a zeroed NCHW tensor after ImageNet normalization, call the first session output, crop away padding, min-max normalize to 8-bit grayscale. DA3's preprocessing and output semantics must be adopted deliberately, not assumed identical. |
| Depth-mask command / cache | `src-tauri/src/ai_commands.rs:145-297` (`generate_ai_depth_mask`, `generate_full_image_depth_map`) | Hashes image path plus geometry adjustments, caches a grayscale depth map, rescales/encodes it for the existing mask pipeline; the full-image depth-map command reuses the same model function. |
| Depth-to-mask postprocessing | `src-tauri/src/mask_generation.rs:908-969`; dispatch at 1311-1313 | Maps the grayscale depth map to min/max/fade/feather parameters. Verify near/far polarity and fade behavior using reference images before changing production. |
| Tauri command registration | `src-tauri/src/lib.rs:2352-2356` | Existing command names can remain stable if the output contract remains compatible. |
| UI invocation and task lifecycle | `src/hooks/useAiMasking.ts:364-390` (`handleGenerateAiDepthMask`) | Invokes `generate_ai_depth_mask`, registers/unregisters the task, and surfaces failures. |
| Depth-range UI | `src/components/panel/right/MasksPanel.tsx`, `src/components/panel/right/Masks.tsx`, `src/components/ui/DepthRangePicker.tsx` | Existing depth mask type and near/far sliders should remain unchanged. Check creation defaults in `src/utils/maskUtils.ts`. |
| Download progress UI state | `src/store/useProcessStore.ts`; download events emitted by `ai_processing.rs:439-442` | Keep event names stable if using the current download progress UI. |

## Evaluation matrix

| Model | Parameters (official card) | License (official card) | ONNX status | Practical assessment |
|---|---:|---|---|---|
| DA3-SMALL | 80M | Apache-2.0 | A community export script targets opset 17 and documents an `image` float tensor and a `depth` output; must validate its exact graph with RapidRAW's bundled ORT. | First candidate. Reference ONNX size is 96.1 MB; reference M4 CPU median is ~504 ms at 504². |
| DA3-BASE | 120M | Apache-2.0 | Same community exporter supports the model name, but a specific pinned binary and ORT smoke-test result are still required. | Quality challenger only. We need same-machine measurements before deciding that its extra cost is justified. |
| DA3-LARGE | 350M | CC BY-NC 4.0 | Same exporter supports the model name; exact target artifact/runtime validation is still required. | Do not ship as a default: non-commercial license is a major distribution constraint and model is substantially larger. |
| Current DA2 ViT-S | Parameter count not asserted in this repository file; model file is 99,373,606 bytes. | Preserve existing model's license/provenance review. | Already loaded by current ORT integration. | Baseline and rollback target. |


**Export-contract caveat:** the reference export README gives a simplified tensor example in the [0, 1] range, but its provided `run_onnx.py` path uses the DA3 `InputProcessor`, whose source applies ImageNet mean/std normalization. Treat this as a contract discrepancy to resolve with a pinned export and fixture-based smoke test; do not infer the Rust input range from the README example alone.

There is no verified cross-platform GPU inference number yet. ONNX export is not proof that every operator runs on every packaged ORT provider. CPU-only is the required baseline; CUDA, DirectML, CoreML, or other acceleration should be measured only where that provider is installed and available.

## Migration plan (do after artifact approval)

1. **Artifact first:** export DA3-SMALL from a pinned official model revision using a documented exporter commit, resolution, opset, graph-optimization setting, and floating-point format. Publish the resulting ONNX file to RapidRAW's controlled model host. Record exact bytes, SHA-256, license/attribution, input/output names, input dimensions, and a known-good sample output. Do not use a moving `main` URL or an unpinned third-party binary.
2. **ORT smoke test:** use the exact Rust `ort` version and bundled ORT runtime on CPU at minimum. Assert input/output names, dtype, shape, finite output, and nonconstant output. Test a portrait, landscape, tiny image, very high-resolution photo, and corrupted/missing model.
3. **Preprocessing adapter:** isolate DA3-specific resize-to-patch-multiple and ImageNet normalization from the DA2 518² padded path. DA3 reference preprocessing constrains dimensions to multiples of 14; 504 is a convenient square target. Do not copy the existing 518² assumptions into DA3.
4. **Output adapter:** explicitly define the meaning and polarity of DA3 depth, finite-value handling, min-max normalization, and resize-back behavior. Verify that existing depth sliders still select the intended near/far regions. Do not infer polarity from a rendered colormap.
5. **Cache safety:** use a DA3-specific filename and checksum. Include the model/version and preprocessing revision in the in-memory cache key (along with current image/geometry inputs) so maps from the old model cannot be reused accidentally. Keep downloads temporary/atomic and checksum-verified.
6. **Failure behavior:** propagate model download, checksum, session-creation, tensor-shape, and inference errors to the existing UI error path. The application must continue to run without loading DA3 until a local depth mask is requested; CPU inference remains supported.
7. **Parity testing:** compare output maps and mask UI behavior on a fixed test image suite, including transforms/crops/rotations, cache reuse and invalidation, cancellation, model-download failure, and missing model.
8. **Decision gate:** ship DA3-SMALL only if blinded depth-mask usefulness improves and performance/memory remain acceptable. Evaluate BASE only when SMALL quality is insufficient. If benefit is not repeatable, retain V2.

## Exact rollback

- Keep V2 constants, URL, SHA-256, and `run_depth_anything_model` intact until the DA3 candidate passes all gates.
- Use a separate DA3 model filename and cache version. Rollback then changes the selected adapter/configuration to V2 without renaming files or re-downloading a DA3 file as V2.
- Revert the migration commit/PR if it changes current mask polarity, cache correctness, or supported CPU targets.

## Benchmark protocol

Use `bench/depth_compare.py` (added on this branch) with the same images, machine, runtime/provider, and iteration count. It records model file size, ONNX session-load time, preprocessing time, inference latency (median/p95/stdev), process RSS, and optional per-process NVIDIA VRAM if `pynvml` is available. It writes normalized preview maps and raw float maps for visual/quantitative follow-up.

Example:

```bash
python -m pip install numpy pillow onnxruntime psutil
python bench/depth_compare.py \
  --v2-model /path/to/depth_anything_v2_vits.onnx \
  --da3-model /path/to/DA3-SMALL-504.onnx \
  --images /path/to/representative-photos \
  --provider CPUExecutionProvider \
  --runs 10 \
  --output-dir bench/out/da3-small-cpu
```

For a CUDA run, install a compatible `onnxruntime-gpu` package and use `--provider CUDAExecutionProvider`. Do not compare numbers across machines or provider configurations. The script's session load time is **model-session initialization**, not the entire RapidRAW app startup. App startup impact requires a separate packaged-app cold-start measurement. Depth quality is not reducible to latency: inspect paired maps blind and, where suitable ground truth exists, evaluate on a documented depth dataset with an appropriate scale-invariant metric. No such RapidRAW quality benchmark is included in this branch because no approved image/ground-truth corpus was supplied.

## PR / commit proposal

Suggested commits:
- `docs(ai): document DA3 depth migration evaluation and gates`
- `bench(ai): add repeatable V2 versus DA3 ONNX benchmark`

Suggested PR title: **Evaluate Depth Anything 3 for local depth masking**

PR summary: maps the live depth-mask pipeline, identifies DA3-SMALL as the initial candidate on model-size/license grounds, adds a reproducible ONNX benchmark harness, and defines artifact, ORT, mask-polarity, performance, and rollback gates. This PR does not change production inference until a project-controlled, checksum-pinned ONNX artifact and platform-specific smoke-test results are available.

## Results status at authoring time

- Repository analysis: performed against the public tree at commit `56aa54b1d16afab7f3ad53a32845c093545704b3`.
- DA3 model metadata: verified from the official README model-card table.
- Existing DA2 artifact size and SHA: verified from RapidRAW's Hugging Face model-host page.
- Benchmark code: provided, including provider auto-probing and a bounded CPU thread count; not executed in this environment against actual ONNX files/hardware.
- Immediate CPU/memory improvements added on the evaluation branch: depth-only model initialization; contiguous NCHW writes; avoid cloning the full ONNX output tensor before postprocessing; bound the depth session's default intra-op thread count to `max(1, min(logical_cores - 1, 8))` so older CPUs reserve capacity for the editor and high-core-count systems do not spawn an oversized pool.
- Added Rust unit tests for the thread-budget policy (zero/one/two/four logical cores and an eight-thread cap). These tests are committed but could not be executed in this environment because Rust/Cargo is not installed and repository GitHub Actions has no run/status records for this draft branch.
- Depth model is now loaded independently; the inference input loop uses contiguous memory, and the full ONNX output no longer gets cloned before postprocessing.
- DA3 GPU/VRAM and RapidRAW-app startup numbers: not measured.
- Depth-quality improvement: plausible from the official DA3 paper/model card, but not established for RapidRAW's depth-mask use case. The DA3 production migration is therefore **not yet justified**.


## Hardware strategy: older PCs through modern accelerators

The project should maintain a **CPU-first compatibility baseline** and treat hardware acceleration as an optional, tested capability. The current production runtime configuration (the bundled ONNX Runtime dynamic library, `ort` rc.10, and default session builders) does not yet register or package a provider-specific runtime. Merely adding provider names in Rust will not enable them; the compatible provider plugin and its dependent libraries must also be shipped or installed.

| Hardware class | Preferred candidate to evaluate | Fallback / caveats |
|---|---|---|
| Very old Intel/AMD CPU, no usable GPU | ONNX Runtime CPU provider; bounded intra-op threads and lower DA3 input profile if the export is dynamic | Verify the bundled ORT binary's CPU instruction-set baseline on the oldest supported machine. A GPU must never be mandatory. |
| Modern Intel CPU / Intel iGPU / Intel NPU | OpenVINO EP with `device_type=AUTO`, after validating the exact provider and DA3 graph | CPU EP if OpenVINO isn't installed, cannot initialize, or is slower. OpenVINO requires a provider package compatible with RapidRAW's ORT version. |
| NVIDIA discrete GPU, older or newer | CUDA EP first when its compatible runtime is present; consider TensorRT only as a separately measured option | CUDA versions, driver, GPU capability, and provider DLL dependencies must match. Keep CPU as the final fallback; don't require TensorRT for older cards. |
| AMD integrated/discrete GPU on Windows | DirectML EP on supported DirectX 12 devices; benchmark versus CPU | Shared-memory iGPUs can lose time to transfers or compete with the editor for memory; auto mode must be allowed to choose CPU. |
| AMD GPU on Linux | MIGraphX / ROCm only for explicitly supported combinations and packaged builds | Do not make ROCm a universal AMD requirement. Older and unsupported GPUs use CPU. |
| Apple Silicon / supported macOS | CoreML EP | CPU fallback; measure rather than assume ANE/GPU acceleration helps this model. |
| Any machine whose accelerator EP is missing or fails | CPU EP | Session creation/inference errors from the accelerated path should trigger a CPU retry, with the fallback recorded in logs. |

These are provider candidates, not a claim that the current RapidRAW package already contains or activates every EP. ONNX Runtime documents CPU as the portable default and provider priority/fallback; the `ort` crate requires the matching Cargo feature **and** a runtime/provider build that actually includes the provider. See https://onnxruntime.ai/docs/execution-providers/ and https://docs.rs/ort/latest/ort/session/builder/struct.SessionBuilder.html.

### Device profiles for DA3

After the export has passed dynamic-shape validation, benchmark these multiples of the DA3 patch size (14):

| Profile | Longest-side processing target | Intended use |
|---|---:|---|
| Quality | 504 px | Best detail; default on capable hardware |
| Balanced | 392 px | Older CPUs, low-power laptops, or an accelerator that does not outperform CPU |
| Performance | 280 px | Very slow/low-memory systems where interactive use matters more than fine mask edges |

This is a proposed profile matrix, **not yet the app's automatic hardware policy**. Start with 504 and 392 for matched-photo comparisons. Ship automatic profile selection only after recording depth-mask quality and latency on representative old CPU-only, Intel iGPU, AMD iGPU/dGPU, and NVIDIA dGPU systems. Fixed-shape ONNX exports cannot switch these resolutions; the chosen model must support the intended dimensions, and all input dimensions must comply with DA3's patch-size requirements.

### What the current branch already optimizes

- `get_or_init_depth_model` loads/verifies just the depth model for depth-mask work, rather than forcing download/session allocation for SAM, foreground, and sky models. The same shared initialization lock prevents duplicate concurrent model initialization.
- `run_depth_anything_model` fills contiguous NCHW input storage through a flat slice and reads the ONNX output through a borrowed view, avoiding a second full output-array allocation. The output length is validated before indexing.
- `bench/depth_compare.py --provider auto` probes installed accelerators in a platform-preferred order, records the attempted/selected provider, retains CPU fallback, and allows explicit `--intra-threads` tests. It records CPU architecture, logical processor count, and system RAM. NVIDIA VRAM sampling is optional; iGPU shared-memory attribution is not portable.
- The thread-budget unit tests are present but unexecuted here. These changes have **not** been benchmarked on physical hardware in this environment. Their expected benefit must be verified with the included harness.

### Release acceptance matrix

At minimum, run a fixed image suite on: (1) the oldest supported CPU without a discrete GPU, (2) a recent Intel iGPU system, (3) an AMD iGPU or DirectML-capable system, (4) an NVIDIA system with supported CUDA, and (5) a system where the requested EP is intentionally unavailable. For each, record cold model-load time, warmed inference median/p95, peak RSS, accelerator memory where observable, depth-mask quality, CPU fallback success, and the actual ORT provider placement from profiling. Test AC/battery mode and a low-memory system. Do not call an EP "active" just because registration succeeded: provider registration and actual graph-node placement are different facts.
