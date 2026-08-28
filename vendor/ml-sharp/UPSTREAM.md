# SHARP vendor provenance

This directory is an **unmodified source snapshot** of Apple's official SHARP
research repository. It is vendored so that Splatmorph installations use a
known-compatible runtime without depending on an unpinned network checkout.

- Upstream repository: <https://github.com/apple/ml-sharp>
- Upstream commit: `1eaa046834b81852261262b41b0919f5c1efdd2e`
- Upstream commit date: 2025-12-18
- Upstream commit subject: `Fix #5 and #6.`
- Snapshot verified against upstream: 2026-08-27

No SHARP source modifications are maintained in this directory. Splatmorph's
own implementation lives under `packages/`.

The source in this directory remains Copyright (C) Apple Inc. and is governed
by [`LICENSE`](LICENSE). The SHARP model and weights are not included in this
repository; when downloaded or supplied separately, they are governed by
[`LICENSE_MODEL`](LICENSE_MODEL). Third-party notices from upstream are retained
in [`ACKNOWLEDGEMENTS`](ACKNOWLEDGEMENTS).

If this snapshot is updated, record the new full upstream commit hash and verify
the complete directory against a clean checkout before publishing it.
