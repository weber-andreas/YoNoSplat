from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


OCCAMLGS_SH_DEGREE = 3
OCCAMLGS_NUM_REST = 3 * (OCCAMLGS_SH_DEGREE + 1) ** 2 - 3  # 45


def export_occamlgs_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    """Export a PLY compatible with OccamLGS gaussian_model.py load_ply.

    Differences from export_ply:
    - opacity is inverted back to logit space (OccamLGS applies sigmoid internally)
    - f_rest_* is padded with zeros to match sh_degree=3 (45 coefficients)
    """
    # Rotations: scipy uses (x,y,z,w), 3DGS / OccamLGS uses (w,x,y,z).
    rotations_np = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations_np = R.from_matrix(rotations_np).as_quat()
    x, y, z, w = rearrange(rotations_np, "g xyzw -> xyzw g")
    rotations_np = np.stack((w, x, y, z), axis=-1)

    f_dc = harmonics[..., 0].detach().cpu().contiguous().numpy()

    # Pad higher-order SH coefficients with zeros; YoNoSplat only predicts the DC band.
    n = means.shape[0]
    f_rest = np.zeros((n, OCCAMLGS_NUM_REST), dtype=np.float32)

    # OccamLGS stores opacity as a raw logit (pre-sigmoid).
    opacities_logit = torch.logit(opacities.clamp(1e-6, 1 - 1e-6))

    dtype_full = [(attr, "f4") for attr in construct_list_of_attributes(OCCAMLGS_NUM_REST)]
    elements = np.empty(n, dtype=dtype_full)
    attributes = np.concatenate(
        [
            means.detach().cpu().numpy(),
            np.zeros((n, 3), dtype=np.float32),  # normals (unused)
            f_dc,
            f_rest,
            opacities_logit[..., None].detach().cpu().numpy(),
            scales.log().detach().cpu().numpy(),
            rotations_np,
        ],
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
