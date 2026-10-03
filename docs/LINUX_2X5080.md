# Two RTX 5080s on Linux

This branch is eddoursul/custom `25e86c7` with the changes needed to run UD-Q4_K_XL on Linux with two 16 GB RTX 5080s. The expert tier design is unchanged. A comparison with upstream v0.1.38 on the same machine is in [docs/2x5080-q4.md](https://github.com/kim-haneol/Strata/blob/q4-diagnostics/docs/2x5080-q4.md) on the `q4-diagnostics` branch.

Machine: Ryzen 9 9900X, 185 GiB RAM, two RTX 5080 16 GB, Rocky Linux 10.1. Physical GPU 0 links at PCIe 5.0 x8, GPU 1 at x4. The config lists the cards as `[1, 0]`, so GPU 1 is the main device (all layers, INT8 K/V) and GPU 0, on the wider link, is the second expert tier.

Which card is main mattered most. My first config used physical GPU 0 (x8) as main and GPU 1 (x4) as the expert tier. Both runs used the same binary and 64K context, with the GPU roles reversed. The first run also had `--stats` enabled.

| One API run each | GPU 0 (x8) main | GPU 1 (x4) main, x8 expert tier |
| --- | ---: | ---: |
| 4K prefill | 972.2 | 1539.7 |
| 32K prefill | 2703.7 | 3614.3 |
| 4K prompt, 256-token decode | 56.1 | 72.3 |
| 300-token code decode | 69.4 | 82.1 |

I put the expert tier on the wider link because it exchanges data with the main card at every layer. The swap moves the link and the physical card together, so these runs don't separate the two; both cards are the same model. Decode here is free generation, and the two code answers differ. The extracted numbers, config differences and binary hash are in [`docs/gpu-order-20260930.json`](https://github.com/kim-haneol/Strata/blob/linux-2x5080/docs/gpu-order-20260930.json).

`examples/ud-q4-k-xl-2x5080-linux.json` is the config. Settings that differ from the fork's Q4 config:

| Setting | Value |
| --- | --- |
| `--second-gpu-prefetch-mb` | 8 |
| `--pcie-frac` | 0.2 |
| `--pool-workers` | 10 |
| reserve on each GPU | 1024 MiB |
| `--max-context` | 262144 |

Prefetch and PCIe share came from a fixed-text sweep at 64K context. Every row emitted the same 300 tokens from the same 54-token prompt:

| Variant | Decode tok/s |
| --- | ---: |
| prefetch 12 MiB, PCIe share 0 (fork default) | 90.18 |
| prefetch 6 MiB, PCIe share 0 | 92.25 |
| prefetch 8 MiB, PCIe share 0 | 93.74 |
| prefetch 8 MiB, PCIe share 0.20 | 94.21 |
| prefetch 8 MiB, PCIe share 0.35 | 81.36 |

A 512 MiB reserve on the second GPU (93.82) and 11 pool workers (93.60) were not worth keeping.

At 64K context the API service passed exact arithmetic and an exact 32K retrieval check. 32K prefill ran at 3616.7 tok/s. Three consecutive 300-token code requests decoded at 84.6, 108.3 and 117.5 tok/s.

With `--max-context 262144` the engine starts with 1477 main and 4813 second-GPU experts and passed the same checks (32K prefill 3540.9 tok/s). I have not tested prompts near 262K.

Code changes, by commit:

- GCC build fixes, including an unsequenced `off[j1++]` read in prompt expert batching.
- The Linux CPU pool groups SMT siblings by physical core from sysfs; the prompt path's PLE reader restores the process CPU mask.
- The stale round-328 expert-cache warning is replaced, as upstream did in 0.1.8.
- The server no longer adds `--layer-split` when `--second-gpu` is set.
- The QSA top-k host check no longer counts two reserved rows, so the 262,144-cell arena starts. The parity test now covers that boundary (82 cases pass).
- Opt-in diagnostic flags (`--spec-follow` with `--serve`, `--diag-*`), off by default.

`tools/local-build` has the build, pack, validation and profiling scripts I used. They point at paths and earlier run directories on my machine, so they need those set up before they run elsewhere.

Code changes, scripts and this note were done with help from Codex (GPT) and Claude.
