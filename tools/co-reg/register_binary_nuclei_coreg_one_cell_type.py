from pathlib import Path
import re

import numpy as np
import pyvips

"""
Register WSI binary nuclei map for one cell type to an H&E image that have already been co-registered to a Visium image
"""
def ensure_3x3_affine(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape == (2, 3):
        out = np.eye(3, dtype=np.float64)
        out[:2, :] = matrix
        matrix = out
    elif matrix.shape != (3, 3):
        raise ValueError(f"Expected 2x3 or 3x3 matrix, got {matrix.shape}")

    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("Matrix is not affine.")

    return matrix


def get_image_dimensions(image_path: str | Path) -> tuple[int, int]:
    image = pyvips.Image.new_from_file(str(image_path), access="sequential")
    return image.width, image.height


def get_cell_name(binary_map_path: str | Path) -> str:
    """whole_{cell}_image_complete.png -> {cell}"""
    stem = Path(binary_map_path).stem
    match = re.fullmatch(r"whole_(.+)_image_complete", stem)
    return match.group(1) if match else stem

def load_final_matrix(final_matrix_path: str | Path) -> np.ndarray:
    """
    final_matrix.npy from your co-registration already contains:
      scale -> manual rotation/translation -> fine residual

    Do NOT multiply scale_matrix.npy or manual_matrix.npy into it again.
    """
    matrix = ensure_3x3_affine(np.load(final_matrix_path))

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Final matrix contains NaN/Inf values.")

    if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError("Final matrix is singular.")

    return matrix

def build_binary_to_fixed_matrix(
    binary_width: int,
    binary_height: int,
    coreg_moving_width: int,
    coreg_moving_height: int,
    final_matrix: np.ndarray,
) -> np.ndarray:
    """
    Coordinate chain:

        full-resolution binary nuclei map
            -> coordinates of the exact moving H&E used in co-registration
            -> final_matrix.npy
            -> fixed H&E
    """
    binary_to_moving = np.array(
        [
            [coreg_moving_width / binary_width, 0.0, 0.0],
            [0.0, coreg_moving_height / binary_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    registration_matrix = (
        ensure_3x3_affine(final_matrix)
        @ binary_to_moving
    )

    print("\nBinary -> co-reg moving matrix:")
    print(binary_to_moving)

    print("\nComplete binary -> fixed matrix:")
    print(registration_matrix)

    return registration_matrix

def inspect_transformed_bounds(
    matrix: np.ndarray,
    source_width: int,
    source_height: int,
    fixed_width: int,
    fixed_height: int,
) -> None:
    corners = np.array(
        [
            [0, 0, 1],
            [source_width - 1, 0, 1],
            [0, source_height - 1, 1],
            [source_width - 1, source_height - 1, 1],
        ],
        dtype=np.float64,
    ).T

    transformed = (ensure_3x3_affine(matrix) @ corners)[:2].T

    names = ["top-left", "top-right", "bottom-left", "bottom-right"]

    print("\nTransformed source corners:")
    for name, (x, y) in zip(names, transformed):
        print(f"  {name:<12} x={x:10.2f}, y={y:10.2f}")

    print(
        f"\nFixed canvas: 0..{fixed_width - 1} x "
        f"0..{fixed_height - 1}"
    )

def load_binary_vips(binary_map_path: str | Path) -> pyvips.Image:
    image = pyvips.Image.new_from_file(
        str(binary_map_path),
        access="random",
    )

    if image.bands > 1:
        image = image.extract_band(0)

    return (image > 0).ifthenelse(255, 0).cast("uchar")


def register_binary_map(
    binary_map_path: str | Path,
    registration_matrix: np.ndarray,
    fixed_width: int,
    fixed_height: int,
) -> pyvips.Image:
    binary = load_binary_vips(binary_map_path)
    matrix = ensure_3x3_affine(registration_matrix)

    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    tx = float(matrix[0, 2])
    c = float(matrix[1, 0])
    d = float(matrix[1, 1])
    ty = float(matrix[1, 2])

    print("\nApplying affine:")
    print(f"[{a:.10f}  {b:.10f}  {tx:.10f}]")
    print(f"[{c:.10f}  {d:.10f}  {ty:.10f}]")

    nearest = pyvips.Interpolate.new("nearest")

    registered = binary.affine(
        [a, b, c, d],
        interpolate=nearest,
        odx=tx,
        ody=ty,
        oarea=[0, 0, int(fixed_width), int(fixed_height)],
        background=[0],
        extend="background",
    )

    return (registered > 0).ifthenelse(255, 0).cast("uchar")


def save_vips_image(image: pyvips.Image, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".png":
        image.pngsave(str(output_path), compression=6)
    elif suffix in {".tif", ".tiff"}:
        image.tiffsave(
            str(output_path),
            compression="deflate",
            tile=True,
            tile_width=512,
            tile_height=512,
            bigtiff=True,
            pyramid=False,
        )
    else:
        image.write_to_file(str(output_path))

def register_single_binary_map(
    binary_map_path: str | Path,
    fixed_image_path: str | Path,
    coreg_moving_image_path: str | Path,
    final_matrix_path: str | Path,
    save_dir: str | Path,
    output_extension: str = ".png",
) -> Path:
    """
    Register one explicitly specified binary map.

    Example:
        whole_connective_image_complete.png
    """
    binary_map_path = Path(binary_map_path)
    fixed_image_path = Path(fixed_image_path)
    coreg_moving_image_path = Path(coreg_moving_image_path)
    final_matrix_path = Path(final_matrix_path)
    save_dir = Path(save_dir)

    for path, label in [
        (binary_map_path, "binary map"),
        (fixed_image_path, "fixed image"),
        (coreg_moving_image_path, "co-registration moving image"),
        (final_matrix_path, "final matrix"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}:\n{path}")

    if not output_extension.startswith("."):
        output_extension = "." + output_extension

    cell_name = get_cell_name(binary_map_path)

    binary_width, binary_height = get_image_dimensions(binary_map_path)
    moving_width, moving_height = get_image_dimensions(coreg_moving_image_path)
    fixed_width, fixed_height = get_image_dimensions(fixed_image_path)

    print("=" * 70)
    print(f"CELL TYPE: {cell_name}")
    print("=" * 70)
    print(f"Binary map:        {binary_width:,} x {binary_height:,}")
    print(f"Co-reg moving H&E: {moving_width:,} x {moving_height:,}")
    print(f"Fixed H&E:         {fixed_width:,} x {fixed_height:,}")

    final_matrix = load_final_matrix(final_matrix_path)

    registration_matrix = build_binary_to_fixed_matrix(
        binary_width=binary_width,
        binary_height=binary_height,
        coreg_moving_width=moving_width,
        coreg_moving_height=moving_height,
        final_matrix=final_matrix,
    )

    inspect_transformed_bounds(
        matrix=registration_matrix,
        source_width=binary_width,
        source_height=binary_height,
        fixed_width=fixed_width,
        fixed_height=fixed_height,
    )

    registered = register_binary_map(
        binary_map_path=binary_map_path,
        registration_matrix=registration_matrix,
        fixed_width=fixed_width,
        fixed_height=fixed_height,
    )

    output_path = (
        save_dir
        / f"{cell_name}_registered_binary{output_extension}"
    )

    save_vips_image(registered, output_path)

    print(f"\nSaved:\n{output_path}")
    return output_path

if __name__ == "__main__":

    biopsy_number = "116"
    cell = "epithelial"
    # --------------------------------------------------------
    # DIRECT PATH TO ONE BINARY MAP
    # --------------------------------------------------------

    binary_map_path = Path(
        "/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/new/"
        f"Func{biopsy_number}_ST_HE_40x_BF_01/"
        f"wsi/{cell}/"
        f"whole_{cell}_image_complete.png"
    )

    # --------------------------------------------------------
    # FIXED H&E
    # --------------------------------------------------------

    fixed_image_path = Path(
        "/media/jenny/Expansion1/Prostata_Vilde/"
        "Co-reg_images_10x/"
        f"Func{biopsy_number}_HE.tif"
    )

    # --------------------------------------------------------
    # EXACT MOVING H&E USED TO GENERATE final_matrix.npy
    # --------------------------------------------------------

    coreg_moving_image_path = Path(
        "/media/jenny/Expansion/HE/"
        "1_3_40x/images/"
        f"Func{biopsy_number}_ST_HE_40x_BF_01_"
        "dowsampled_1_3.png"
    )

    # --------------------------------------------------------
    # FINAL CO-REGISTRATION MATRIX
    # --------------------------------------------------------

    final_matrix_path = Path(
        "/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/new/"
        f"Func{biopsy_number}_ST_HE_40x_BF_01/"
        "co_registration/"
        "final_matrix.npy"
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    save_dir = Path(
        "/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/new/"
        f"Func{biopsy_number}_ST_HE_40x_BF_01/"
        "co_registration/"
        "registered_nuclei_maps/"
    )

    output_extension = ".png"

    register_single_binary_map(
        binary_map_path=binary_map_path,
        fixed_image_path=fixed_image_path,
        coreg_moving_image_path=coreg_moving_image_path,
        final_matrix_path=final_matrix_path,
        save_dir=save_dir,
        output_extension=output_extension,
    )