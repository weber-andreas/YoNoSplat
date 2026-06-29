"""
View a 3DGS PLY file in the browser via viser (works on headless machines over SSH).

Usage:
    python -m src.scripts.view_gaussians <path/to/scene_gaussians.ply> [--port 8080]

If a _cameras.npz file exists alongside the PLY (saved by save_debug_info), context
and target camera frustums are shown automatically.

On your local machine, forward the port:
    ssh -L 8080:localhost:8080 user@server

Then open http://localhost:8080 in your browser.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from plyfile import PlyData
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def load_ply(path: str):
    v = PlyData.read(path)["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # DC SH band -> base RGB color
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    # Sigmoid activation on raw opacity
    opacities = (
        (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32))))
        .reshape(-1, 1)
        .astype(np.float32)
    )

    # Scales stored as log(scale)
    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    )

    # Quaternion (w, x, y, z) -> rotation matrix
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)

    # 3D covariance: R @ S^2 @ R^T
    S = scales[:, :, None] * np.eye(3, dtype=np.float32)[None]
    RS = R @ S
    covariances = (RS @ RS.transpose(0, 2, 1)).astype(np.float32)

    return positions, covariances, colors, opacities


def opencv_to_opengl(positions: np.ndarray, covariances: np.ndarray):
    # Flip Y and Z to convert from OpenCV (Y-down, Z-forward)
    # to OpenGL convention (Y-up, Z-backward) that viser uses.
    positions = positions * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    F = np.diag(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    covariances = (F @ covariances @ F).astype(np.float32)
    return positions, covariances


def center_scene(positions: np.ndarray, covariances: np.ndarray):
    center = np.median(positions, axis=0)
    positions = positions - center
    scale = float(np.abs(positions).max())
    positions = positions / scale
    covariances = covariances / (scale**2)
    return positions, covariances, center, scale


def transform_c2w(
    c2w: np.ndarray, center: np.ndarray | None, scale: float | None
) -> np.ndarray:
    c2w = c2w.copy().astype(np.float32)
    F = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    c2w[:, :3, :] *= F[:, None]
    if center is not None and scale is not None:
        c2w[:, :3, 3] = (c2w[:, :3, 3] - center) / scale
    return c2w


def add_camera_frustums(
    server: viser.ViserServer,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    scale: float = 0.05,
) -> list:
    handles = []
    for i, (mat, K) in enumerate(zip(c2w, intrinsics)):
        fy = float(K[1, 1])
        fx = float(K[0, 0])
        fov_y = 2.0 * np.arctan(0.5 / fy)
        wxyz = Rotation.from_matrix(mat[:3, :3]).as_quat()[[3, 0, 1, 2]]
        handles.append(
            server.scene.add_camera_frustum(
                f"/{label}/{i:04d}",
                fov=fov_y,
                aspect=fx / fy,
                scale=scale,
                color=color,
                wxyz=wxyz.astype(np.float32),
                position=mat[:3, 3].astype(np.float32),
            )
        )
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Path to _gaussians.ply file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--no-center", action="store_true", help="Skip centering/normalizing the scene"
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=10.0,
        help="Multiply scene size after centering",
    )
    parser.add_argument(
        "--frustum-scale", type=float, default=0.15, help="Size of camera frustums"
    )
    args = parser.parse_args()

    print(f"Loading {args.ply} ...")
    positions, covariances, colors, opacities = load_ply(args.ply)
    print(f"  {len(positions):,} Gaussians loaded")

    positions, covariances = opencv_to_opengl(positions, covariances)

    center, scale = None, None
    if not args.no_center:
        positions, covariances, center, scale = center_scene(positions, covariances)
    if args.scene_scale != 1.0:
        positions = positions * args.scene_scale
        covariances = covariances * (args.scene_scale**2)

    server = viser.ViserServer(port=args.port, verbose=False)

    server.scene.add_gaussian_splats(
        "/scene",
        centers=positions,
        covariances=covariances,
        rgbs=colors,
        opacities=opacities,
    )

    cameras_path = Path(args.ply).with_name(
        Path(args.ply).stem.split("_gaussians")[0] + "_cameras.npz"
    )
    if cameras_path.exists():
        cam = np.load(cameras_path)
        target_c2w = transform_c2w(cam["target_extrinsics"], center, scale)
        context_c2w = transform_c2w(cam["context_extrinsics"], center, scale)
        if args.scene_scale != 1.0:
            target_c2w[:, :3, 3] *= args.scene_scale
            context_c2w[:, :3, 3] *= args.scene_scale
        target_handles = add_camera_frustums(
            server,
            target_c2w,
            cam["target_intrinsics"],
            label="target",
            color=(255, 80, 80),
            scale=args.frustum_scale,
        )
        context_handles = add_camera_frustums(
            server,
            context_c2w,
            cam["context_intrinsics"],
            label="context",
            color=(80, 80, 255),
            scale=args.frustum_scale * 0.5,
        )
        print(
            f"  {len(target_c2w)} target cameras (red), {len(context_c2w)} context cameras (blue)"
        )

        show_target = server.gui.add_checkbox("Show target cameras", initial_value=True)
        show_context = server.gui.add_checkbox(
            "Show context cameras", initial_value=True
        )

        @show_target.on_update
        def _(_) -> None:
            for h in target_handles:
                h.visible = show_target.value

        @show_context.on_update
        def _(_) -> None:
            for h in context_handles:
                h.visible = show_context.value

        view_dropdown = server.gui.add_dropdown(
            "Go to target view",
            options=["(free)"] + [str(i) for i in range(len(target_c2w))],
            initial_value="(free)",
        )

        @view_dropdown.on_update
        def _(_) -> None:
            if view_dropdown.value == "(free)":
                return
            idx = int(view_dropdown.value)
            mat = target_c2w[idx]
            pos = mat[:3, 3].astype(np.float32)
            wxyz = (
                Rotation.from_matrix(mat[:3, :3])
                .as_quat()[[3, 0, 1, 2]]
                .astype(np.float32)
            )
            for client in server.get_clients().values():
                client.camera.position = pos
                client.camera.wxyz = wxyz

    else:
        print(
            f"  No cameras file found at {cameras_path} — run with test.save_debug_info=true to generate it"
        )

    print("\nViewer ready. Forward the port from your local machine:")
    print(f"  ssh -L {args.port}:localhost:{args.port} <user>@<server>")
    print(f"Then open: http://localhost:{args.port}\n")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
