"""splatmorph CLI.

  splatmorph two A.png B.png -o loop.mp4          # one looping morph
  splatmorph chain ./images -o chain.mp4          # the full pipeline
  splatmorph chain ./images --wave                # peak-trough anti-pairing
                                                  # with auto-mirrored troughs
"""

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def _write(out: Path, frames, fps: int, gif: bool):
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=fps, codec="libx264",
                pixelformat="yuv420p", macro_block_size=8,
                output_params=["-threads", "2"])
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / fps:.1f}s loop)")
    if gif:
        small = frames[::2, ::2, ::2]
        gp = out.with_suffix(".gif")
        iio.imwrite(gp, small, duration=2 * 1000 / fps, loop=0)
        print(f"wrote {gp}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="splatmorph",
        description="Two images in, one seamlessly-looping 3D Gaussian morph out.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--out", type=Path, default=None)
    common.add_argument("--fps", type=int, default=30)
    common.add_argument("--gif", action="store_true", help="also write a GIF")
    common.add_argument("--cache", type=Path, default=Path("gaussians"),
                        help="SHARP splat cache directory")
    common.add_argument("--weights", type=Path, default=None,
                        help="SHARP checkpoint (default: sharp CLI's own)")
    common.add_argument("--wave", action="store_true",
                        help="peak-trough anti-pairing (crests collapse into troughs)")
    common.add_argument("--strict", action="store_true",
                        help="exit nonzero if any verification gate fails")

    p2 = sub.add_parser("two", parents=[common],
                        help="loop between two images (A -> B -> A, exact period)")
    p2.add_argument("image_a", type=Path)
    p2.add_argument("image_b", type=Path)
    p2.add_argument("--frames", type=int, default=96)

    pc = sub.add_parser("chain", parents=[common],
                        help="filter, order, and morph a whole folder into one grand loop")
    pc.add_argument("folder", type=Path)
    pc.add_argument("--frames-per-seg", type=int, default=64)
    pc.add_argument("--no-mirror", action="store_true",
                    help="with --wave: do not auto-add mirrored counterparts")

    args = ap.parse_args(argv)
    from . import pipeline                          # lazy: torch import cost

    if args.cmd == "two":
        out = args.out or Path("out/loop.mp4")
        frames, gates = pipeline.render_two(
            args.image_a, args.image_b, args.cache, frames=args.frames,
            anti=args.wave, weights=args.weights)
    else:
        images = sorted(p for p in args.folder.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if len(images) < 3:
            sys.exit("splatmorph: chain needs at least 3 images")
        out = args.out or Path("out/chain.mp4")
        frames, gates, order = pipeline.render_chain(
            images, args.cache, frames_per_seg=args.frames_per_seg,
            anti=args.wave, auto_mirror=args.wave and not args.no_mirror,
            weights=args.weights)

    print("== verification gates")
    ok = pipeline.verify(gates)
    _write(out, np.asarray(frames), args.fps, args.gif)
    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
