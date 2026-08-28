"""Scene loading: image -> SHARP splats -> fields -> transport features."""

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps

from splatmorph_fields import analyze, curl_field
from splatmorph_transport import coarse_features
from splatmorph_transport.transport import get_device


def ensure_gaussians(image_path: Path, cache_dir: Path, weights: Path | None = None) -> Path:
    """Run SHARP once per image, cached by stem. Requires the `sharp` CLI
    (Apple's model, research license — see the repository README)."""
    ply = cache_dir / f"{image_path.stem}.ply"
    if ply.exists():
        return ply
    cache_dir.mkdir(parents=True, exist_ok=True)
    # prefer the sharp CLI installed alongside this interpreter (same venv),
    # falling back to PATH
    local = Path(sys.executable).parent / "sharp"
    sharp_bin = str(local) if local.exists() else (shutil.which("sharp") or "sharp")
    cmd = [sharp_bin, "predict", "-i", str(image_path), "-o", str(cache_dir), "--no-render"]
    if weights is not None:
        cmd += ["-c", str(weights)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("splatmorph: the `sharp` CLI is not installed. "
                 "Install the vendored Apple SHARP snapshot: "
                 "pip install ./vendor/ml-sharp")
    if not ply.exists():
        raise FileNotFoundError(f"SHARP did not produce {ply}")
    return ply


def mirror_image(image_path: Path, out_dir: Path) -> Path:
    """Vertically mirrored copy — gives every crest a trough at its own
    position, the anti-pairing counterpart set."""
    out = out_dir / f"{image_path.stem}_flip{image_path.suffix}"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        ImageOps.flip(Image.open(image_path).convert("RGB")).save(out)
    return out


class Scene:
    """One image's full stack: splats, fields, curl, transport features."""

    def __init__(self, image_path: Path, cache_dir: Path, weights: Path | None = None,
                 device=None):
        from sharp.utils.gaussians import Gaussians3D, load_ply   # lazy: needs sharp

        self.device = device or get_device()
        self.name = image_path.name
        self.image_path = Path(image_path)
        ply = ensure_gaussians(self.image_path, cache_dir, weights)
        self.elev, self.D, self.span, self.roi = analyze(ply, self.image_path)
        gs, self.meta = load_ply(ply)
        self.gs = Gaussians3D(*[t.to(self.device) for t in gs])
        self.col_grid = self.gs.colors[0].reshape(2, 768, 768, 3)[0].cpu().numpy()
        self.curl = curl_field(self.elev)
        self.feats = coarse_features(self.elev, self.col_grid, self.roi,
                                     device=self.device)
