"""
Analyze per-pixel Gaussian weight distribution to test the sparse-influence assumption.

For each target camera view, re-projects the Gaussians via EWA splatting and computes
per-pixel alpha-compositing weights:

  w_i = alpha_i * T_i,  T_i = prod_{j<i}(1 - alpha_j)

Reports:
  - max weight per pixel: near 1.0 means one Gaussian dominates (sparse influence holds)
  - effective Gaussian count per pixel = 1 / sum(w^2): near 1.0 means sparse influence
  - accumulated alpha per pixel: coverage of the pixel by the Gaussian cloud

Saves per-view heatmaps and aggregated histograms to the output directory.

Usage:
    python -m src.scripts.analyze_gaussian_weights \\
        outputs/.../scene_gaussians.ply \\
        [--cameras PATH]       # default: inferred from PLY path (same stem convention)
        [--image-size W H]     # default: inferred from nearby depth/ images, else 512 384
        [--stride 8]           # pixel sampling stride; larger is faster but coarser
        [--sigma-cull 3.0]     # Gaussian bounding-box radius in sigmas
        [--out DIR]            # output dir (default: same directory as PLY)

Memory note: peak RAM per batch is roughly 8 * G * B * 4 bytes where B is the pixel
batch size (auto-tuned to ~200 MB). For very large G (>500k), use --stride 16.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_ply(path: Path):
    """Return positions (G,3), covariances (G,3,3), opacities (G,)."""
    v = PlyData.read(path)["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))).ravel()

    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    )
    # PLY stores quaternion as (w, x, y, z); scipy expects (x, y, z, w)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(
        np.float32
    )
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)

    S = scales[:, :, None] * np.eye(3, dtype=np.float32)[None]
    RS = R @ S
    covariances = (RS @ RS.transpose(0, 2, 1)).astype(np.float32)
    return positions, covariances, opacities


def infer_image_size(cameras_path: Path) -> tuple[int, int]:
    """Try to read H x W from depth images saved alongside; fall back to 512x384."""
    depth_dir = cameras_path.parent / "depth"
    if depth_dir.exists():
        pngs = sorted(depth_dir.glob("*.png"))
        if pngs:
            from PIL import Image

            img = Image.open(pngs[0])
            print(
                f"  Image size inferred from {pngs[0].name}: {img.width} x {img.height}"
            )
            return img.width, img.height
    print(
        "  Warning: could not infer image size; using 512x384. Pass --image-size W H to override."
    )
    return 512, 384


# ---------------------------------------------------------------------------
# EWA projection
# ---------------------------------------------------------------------------


def project_gaussians(
    positions: np.ndarray,  # (G, 3) float32
    covariances: np.ndarray,  # (G, 3, 3) float32
    opacities: np.ndarray,  # (G,) float32
    c2w: np.ndarray,  # (4, 4) float64  camera-to-world
    K_norm: np.ndarray,  # (3, 3) float64  normalized intrinsics
    W: int,
    H: int,
    near: float = 0.1,
    sigma_cull: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project Gaussians into pixel space via EWA splatting.

    Intrinsics are stored in normalized form: multiply row 0 by W and row 1 by H
    to get pixel-space K (same convention as decoder_splatting_gsplat.py:66-67).

    Returns (means2d, cov2d_inv, depths, opacities, radii) for valid Gaussians only,
    all float32.
    """
    K = K_norm.astype(np.float64).copy()
    K[0] *= W
    K[1] *= H
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    w2c = np.linalg.inv(c2w.astype(np.float64))
    R_cam = w2c[:3, :3]  # (3, 3)
    t_cam = w2c[:3, 3]  # (3,)

    pos64 = positions.astype(np.float64)
    p_cam = (R_cam @ pos64.T).T + t_cam  # (G, 3)
    z = p_cam[:, 2]
    valid = z > near

    u = np.where(valid, fx * p_cam[:, 0] / z + cx, 0.0)
    v = np.where(valid, fy * p_cam[:, 1] / z + cy, 0.0)

    # Jacobian of perspective projection at each camera-space point
    z2 = np.where(valid, z**2, 1.0)
    J = np.zeros((len(positions), 2, 3), dtype=np.float64)
    J[:, 0, 0] = np.where(valid, fx / z, 0.0)
    J[:, 0, 2] = np.where(valid, -fx * p_cam[:, 0] / z2, 0.0)
    J[:, 1, 1] = np.where(valid, fy / z, 0.0)
    J[:, 1, 2] = np.where(valid, -fy * p_cam[:, 1] / z2, 0.0)

    # 2D covariance: cov2d = J @ (R_cam @ cov3d @ R_cam^T) @ J^T
    cov64 = covariances.astype(np.float64)
    cov_cam = np.einsum("ij,gjk,lk->gil", R_cam, cov64, R_cam)  # (G, 3, 3)
    cov2d = np.einsum("gij,gjk,glk->gil", J, cov_cam, J)  # (G, 2, 2)

    # Low-pass anti-aliasing filter (same as 3DGS paper)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # Analytic inverse of 2x2 matrix
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    valid &= det > 1e-10

    safe_det = np.where(det > 1e-10, det, 1.0)
    cov2d_inv = np.zeros_like(cov2d)
    cov2d_inv[:, 0, 0] = cov2d[:, 1, 1] / safe_det
    cov2d_inv[:, 0, 1] = -cov2d[:, 0, 1] / safe_det
    cov2d_inv[:, 1, 0] = -cov2d[:, 1, 0] / safe_det
    cov2d_inv[:, 1, 1] = cov2d[:, 0, 0] / safe_det

    # Bounding-box radius: sigma_cull * sqrt(largest eigenvalue of cov2d)
    trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    diff = cov2d[:, 0, 0] - cov2d[:, 1, 1]
    disc = np.sqrt(np.maximum(0.0, diff**2 + 4 * cov2d[:, 0, 1] ** 2))
    lambda_max = (trace + disc) / 2.0
    radii = sigma_cull * np.sqrt(np.maximum(0.0, lambda_max))

    # Cull Gaussians entirely outside the image
    valid &= (u + radii >= 0) & (u - radii < W) & (v + radii >= 0) & (v - radii < H)

    idx = np.where(valid)[0]
    means2d = np.stack([u[idx], v[idx]], axis=1).astype(np.float32)
    return (
        means2d,
        cov2d_inv[idx].astype(np.float32),
        z[idx].astype(np.float32),
        opacities[idx],
        radii[idx].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Per-pixel weight computation
# ---------------------------------------------------------------------------


def analyze_view(
    means2d: np.ndarray,  # (G, 2) float32 — already culled to this view
    cov2d_inv: np.ndarray,  # (G, 2, 2) float32
    depths: np.ndarray,  # (G,) float32
    opacities: np.ndarray,  # (G,) float32
    radii: np.ndarray,  # (G,) float32
    W: int,
    H: int,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-sampled-pixel metrics as 2D maps (ny, nx):
      max_weight, eff_count, total_alpha
    """
    G = len(means2d)
    px_vals = np.arange(stride // 2, W, stride, dtype=np.float32)
    py_vals = np.arange(stride // 2, H, stride, dtype=np.float32)
    nx, ny = len(px_vals), len(py_vals)
    P = nx * ny

    if G == 0:
        return np.zeros((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx))

    # Depth-sort once; all per-pixel computations share this order
    order = np.argsort(depths)
    means2d = means2d[order]
    cov2d_inv = cov2d_inv[order]
    opacities = opacities[order]
    radii = radii[order]

    # Pixel sample grid: rows = y (axis 0), cols = x (axis 1)
    px_grid, py_grid = np.meshgrid(px_vals, py_vals)  # each (ny, nx)
    px_flat = px_grid.ravel()  # (P,)
    py_flat = py_grid.ravel()  # (P,)

    # Auto-tune batch size: target ~200 MB for the (G, B, 2) array
    B = max(1, int(200e6 / (G * 2 * 4)))
    B = min(B, P)

    max_w_flat = np.zeros(P, dtype=np.float32)
    eff_flat = np.zeros(P, dtype=np.float32)
    talpha_flat = np.zeros(P, dtype=np.float32)

    for p0 in range(0, P, B):
        p1 = min(p0 + B, P)
        px = px_flat[p0:p1]  # (b,)
        py = py_flat[p0:p1]
        b = len(px)

        # Displacement from each Gaussian center to each sample pixel: (G, b, 2)
        d = means2d[:, None, :] - np.stack([px, py], axis=1)[None, :, :]

        # Rough bounding-box cull before computing the full quadratic form
        in_bbox = (np.abs(d[:, :, 0]) <= radii[:, None]) & (
            np.abs(d[:, :, 1]) <= radii[:, None]
        )  # (G, b) bool

        # Mahalanobis power: -0.5 * d^T Cinv d  per (Gaussian, pixel)
        Cd = np.einsum("gij,gbj->gbi", cov2d_inv, d)  # (G, b, 2)
        power = -0.5 * np.einsum("gbi,gbi->gb", d, Cd)  # (G, b)

        # Only Gaussians inside bbox with non-negligible contribution
        contrib = in_bbox & (power > -8.0)  # exp(-8) ~ 3e-4

        alpha = np.where(
            contrib,
            np.minimum(0.99, opacities[:, None] * np.exp(np.clip(power, -500.0, 0.0))),
            0.0,
        ).astype(
            np.float32
        )  # (G, b)

        # Alpha compositing weights via vectorized cumprod:
        #   T[g, :] = transmittance BEFORE Gaussian g
        T = np.ones((G, b), dtype=np.float32)
        if G > 1:
            np.cumprod(1.0 - alpha[:-1], axis=0, out=T[1:])
        weights = alpha * T  # (G, b)

        max_w_flat[p0:p1] = weights.max(axis=0)
        sum_w2 = (weights**2).sum(axis=0)
        talpha_flat[p0:p1] = weights.sum(axis=0)
        eff_flat[p0:p1] = np.where(sum_w2 > 1e-8, 1.0 / sum_w2, 0.0)

    return (
        max_w_flat.reshape(ny, nx),
        eff_flat.reshape(ny, nx),
        talpha_flat.reshape(ny, nx),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def save_view_histogram(
    max_w: np.ndarray, eff: np.ndarray, covered: np.ndarray, vi: int, path
):
    """Two-panel histogram for a single view (covered pixels only)."""
    EFF_CLIP = 100
    BIN_SIZE = 2  # shared bin size: 2 units for eff count, 0.02 for max weight [0,1]

    mw = max_w[covered].ravel()
    ef = eff[covered].ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), tight_layout=True)
    fig.suptitle(f"View {vi}  ({len(mw):,} covered pixels)", fontsize=10)

    mw_bins = np.arange(0, 1 + BIN_SIZE / EFF_CLIP, BIN_SIZE / EFF_CLIP)
    axes[0].hist(
        mw, bins=mw_bins, range=(0, 1), color="tomato", edgecolor="white", linewidth=0.3
    )
    med = float(np.median(mw))
    axes[0].axvline(med, color="k", linestyle="--", label=f"median={med:.3f}")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Max weight per pixel")
    axes[0].set_ylabel("Pixel count")
    axes[0].set_title("Max weight")
    axes[0].legend(fontsize=8)

    eff_bins = np.arange(0, EFF_CLIP + BIN_SIZE, BIN_SIZE)
    axes[1].hist(
        np.clip(ef, 0, EFF_CLIP),
        bins=eff_bins,
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    med_e = float(np.median(ef))
    axes[1].axvline(
        min(med_e, EFF_CLIP),
        color="k",
        linestyle="--",
        label=f"median={med_e:.1f}",
    )
    axes[1].set_xlim(0, EFF_CLIP)
    axes[1].set_xlabel("Effective Gaussian count per pixel")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Effective Gaussian count")
    axes[1].legend(fontsize=8)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_summary_histograms(all_max_w, all_eff, n_views, path):
    EFF_CLIP = 100
    BIN_SIZE = 2  # shared bin size: 2 units for eff count, 0.02 for max weight [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    fig.suptitle(
        f"{n_views} views  |  {len(all_max_w):,} pixels (accumulated alpha > 0.1)",
        fontsize=10,
    )

    mw_bins = np.arange(0, 1 + BIN_SIZE / EFF_CLIP, BIN_SIZE / EFF_CLIP)
    axes[0].hist(
        all_max_w,
        bins=mw_bins,
        range=(0, 1),
        color="tomato",
        edgecolor="white",
        linewidth=0.3,
    )
    med = float(np.median(all_max_w))
    axes[0].axvline(med, color="k", linestyle="--", label=f"median = {med:.3f}")
    axes[0].set_xlabel("Max weight per pixel")
    axes[0].set_ylabel("Pixel count")
    axes[0].set_title("Max weight per pixel")
    axes[0].legend()

    eff_bins = np.arange(0, EFF_CLIP + BIN_SIZE, BIN_SIZE)
    axes[1].hist(
        np.clip(all_eff, 0, EFF_CLIP),
        bins=eff_bins,
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    med_e = float(np.median(all_eff))
    axes[1].axvline(
        min(med_e, EFF_CLIP), color="k", linestyle="--", label=f"median = {med_e:.2f}"
    )
    axes[1].set_xlabel("Effective Gaussian count per pixel")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Effective Gaussian count per pixel")
    axes[1].legend()

    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ply", help="Path to _gaussians.ply")
    parser.add_argument(
        "--cameras", help="Path to _cameras.npz (inferred from PLY if omitted)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Image width and height in pixels",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Pixel sampling stride (default: 8; larger = faster but coarser)",
    )
    parser.add_argument(
        "--sigma-cull",
        type=float,
        default=3.0,
        help="Gaussian bounding-box radius in sigmas (default: 3.0)",
    )
    parser.add_argument(
        "--out", help="Output directory (default: same directory as PLY)"
    )
    args = parser.parse_args()

    ply_path = Path(args.ply)

    if args.cameras:
        cam_path = Path(args.cameras)
    else:
        stem = ply_path.stem
        cam_path = ply_path.with_name(stem.split("_gaussians")[0] + "_cameras.npz")

    if not cam_path.exists():
        sys.exit(f"Camera file not found: {cam_path}\nPass --cameras PATH explicitly.")

    out_dir = Path(args.out) if args.out else ply_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gaussians from {ply_path} ...")
    positions, covariances, opacities = load_ply(ply_path)
    print(f"  {len(positions):,} Gaussians")

    print(f"Loading cameras from {cam_path} ...")
    cam = np.load(cam_path)
    extrinsics = cam["target_extrinsics"]  # (V, 4, 4) c2w
    intrinsics = cam["target_intrinsics"]  # (V, 3, 3) normalized

    if args.image_size:
        W, H = args.image_size
        print(f"  Image size: {W} x {H} (from --image-size)")
    else:
        W, H = infer_image_size(cam_path)
    print(
        f"  Stride: {args.stride}  ->  {len(range(0, W, args.stride))} x {len(range(0, H, args.stride))} sample pixels per view"
    )

    all_max_w = []
    all_eff = []

    for vi, (c2w, K_norm) in enumerate(zip(extrinsics, intrinsics)):
        print(f"\nView {vi + 1}/{len(extrinsics)}")

        means2d, cov2d_inv, depths, ops_v, radii = project_gaussians(
            positions,
            covariances,
            opacities,
            c2w,
            K_norm,
            W,
            H,
            sigma_cull=args.sigma_cull,
        )
        print(f"  {len(means2d):,} Gaussians in frustum")

        max_w, eff, talpha = analyze_view(
            means2d, cov2d_inv, depths, ops_v, radii, W, H, stride=args.stride
        )

        covered = talpha > 0.1
        n_cov = int(covered.sum())
        print(f"  Covered pixels (accumulated alpha > 0.1): {n_cov:,} / {max_w.size:,}")
        if n_cov > 0:
            print(
                f"  Max weight   mean={max_w[covered].mean():.3f}  median={np.median(max_w[covered]):.3f}"
            )
            print(
                f"  Effective count    mean={eff[covered].mean():.2f}  median={np.median(eff[covered]):.2f}"
            )
            all_max_w.append(max_w[covered].ravel())
            all_eff.append(eff[covered].ravel())
            save_view_histogram(
                max_w, eff, covered, vi, out_dir / f"view_{vi:03d}_histogram.png"
            )

    if not all_max_w:
        print("\nNo covered pixels found; no summary generated.")
        return

    all_max_w_cat = np.concatenate(all_max_w)
    all_eff_cat = np.concatenate(all_eff)

    # Summary histogram over all views pooled
    save_summary_histograms(
        all_max_w_cat, all_eff_cat, len(all_max_w), out_dir / "summary_histograms.png"
    )

    stats_path = out_dir / "summary_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Gaussians: {len(positions):,}\n")
        f.write(f"Views analyzed: {len(extrinsics)}\n")
        f.write(f"Covered pixels (alpha>0.1): {len(all_max_w_cat):,}\n\n")
        f.write("Max weight per pixel  (1.0 = one Gaussian dominates):\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            f.write(f"  p{p:02d}: {np.percentile(all_max_w_cat, p):.4f}\n")
        f.write("\nEffective Gaussian count per pixel  (1.0 = perfectly sparse):\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            f.write(f"  p{p:02d}: {np.percentile(all_eff_cat, p):.2f}\n")
        frac_sparse = float((all_max_w_cat > 0.9).mean())
        frac_single = float((all_eff_cat < 1.5).mean())
        f.write(
            f"\nFraction of pixels with max_weight > 0.9: {frac_sparse:.3f}  ({100*frac_sparse:.1f}%)\n"
        )
        f.write(
            f"Fraction of pixels with eff_count  < 1.5: {frac_single:.3f}  ({100*frac_single:.1f}%)\n"
        )

    print(f"\nDone. Results saved to {out_dir}/")

    # Print summary to terminal
    with open(stats_path, encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
