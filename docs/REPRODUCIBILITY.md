# Reproducing and validating Splatmorph

Splatmorph has two verification surfaces:

1. a CPU-only unit suite for transport and endpoint/loop mathematics; and
2. a manual NVIDIA GPU smoke test for Apple SHARP inference and `gsplat`
   rendering.

The CPU suite runs in GitHub Actions. The GPU path downloads or consumes Apple's
research model and is intentionally not run by public CI.

## Environment

For the full pipeline, use Python 3.13 and an NVIDIA GPU with a PyTorch-supported
CUDA toolchain. Apple SHARP can perform prediction on CPU, CUDA, or MPS, but
Splatmorph's renderer requires CUDA.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./vendor/ml-sharp
python -m pip install \
  -e packages/splatmorph-transport \
  -e packages/splatmorph-fields \
  -e packages/splatmorph-engine \
  -e 'packages/splatmorph[dev]'
```

Installing or using SHARP constitutes acceptance of Apple's accompanying
licenses. Its model is limited to non-commercial scientific research and
academic development; read `vendor/ml-sharp/LICENSE_MODEL` before downloading
or using the checkpoint.

## CPU checks

These checks do not need SHARP, its model, or a GPU:

```bash
ruff check packages tests
pytest -q
```

The tests exercise the log-domain transport solver, bidirectional barycentric
maps, and the mathematical loop/endpoint invariants on synthetic data. They do
not establish image quality or prove that the external SHARP/gsplat stack works
on a particular machine.

## GPU end-to-end smoke test

First confirm that the runtime and CLI are visible:

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
sharp --help
splatmorph --help
```

`torch.cuda.is_available()` must print `True` for rendering. Then choose two
ordinary RGB images and run a deliberately short strict render:

```bash
splatmorph two A.png B.png \
  --frames 24 \
  --strict \
  --cache artifacts/gaussians \
  --out artifacts/smoke-loop.mp4
```

On first use, SHARP downloads Apple's checkpoint unless `--weights` points to a
checkpoint obtained under Apple's model license. A successful smoke test must:

- exit with status 0;
- report passing solver, loop-closure, and endpoint-purity gates;
- create `artifacts/smoke-loop.mp4`; and
- leave reusable `.ply` files in `artifacts/gaussians`.

The first `gsplat` invocation may compile or initialize CUDA kernels and can be
substantially slower than subsequent runs. Performance and quality depend on
the input images, GPU, PyTorch/CUDA versions, and checkpoint; record those when
publishing benchmark or PSNR results.

## Chain validation

After the two-image smoke test passes, validate the higher-level membership,
ordering, pacing, and boundary gates with a folder containing at least three
images:

```bash
splatmorph chain ./images \
  --strict \
  --cache artifacts/gaussians \
  --out artifacts/chain.mp4
```

Install the optional LPIPS dependency with `python -m pip install
'packages/splatmorph[lpips]'` if perceptual membership filtering is required.
Without it, the CLI reports that filtration was skipped.
