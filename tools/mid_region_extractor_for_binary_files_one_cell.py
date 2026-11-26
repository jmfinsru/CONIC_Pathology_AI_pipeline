from pathlib import Path
from math import ceil

import numpy as np
import pyvips
from natsort import natsorted

"""

Reconstruct a binary WSI from binary patches for one chosen cell type

"""

def find_png_files(type_dir: str | Path):

    type_dir = Path(type_dir)
    type_name = type_dir.name

    print(f"Type directory: {type_dir}")

    image_list = natsorted(
        type_dir.rglob("*.png")
    )

    if not image_list:
        raise FileNotFoundError(
            f"No PNG files found in {type_dir}"
        )

    print(
        f"Found {len(image_list)} PNG files "
        f"for type '{type_name}'"
    )

    return type_name, image_list


def create_wsi_single_folder(
    type_dir: Path,
    root_wsi_dir: Path,
    output_root_dir: Path,
    sample : str,
):

    type_dir = Path(type_dir)
    if not type_dir.is_dir():
        raise NotADirectoryError(f"Cell-type directory not found: {type_dir}")
    root_wsi_dir = Path(root_wsi_dir)
    output_root_dir = Path(output_root_dir)

    # ---------------------------------------------------------
    # Find corresponding original WSI
    # ---------------------------------------------------------

    wsi_path = next(
        root_wsi_dir.rglob(
            f"{sample}_ST_HE_40x_BF_01.btf" # Change name based on original WSI filename
        ),
        None
    )

    if wsi_path is None:
        raise FileNotFoundError(
            f"No WSI found matching:\n"
            f"{sample}_ST_HE_40x_BF_01.btf\n" # Change name based on original WSI filename
            f"inside {root_wsi_dir}"
        )

    print(f"WSI: {wsi_path}")

    wsi_image = pyvips.Image.new_from_file(
        str(wsi_path)
    )

    print(
        f"WSI dimensions: "
        f"{wsi_image.width} x {wsi_image.height}"
    )

    # Process only the selected cell-type directory
    type_name, image_list = find_png_files(type_dir)

    print(f"Processing type: {type_name}")

    # -----------------------------------------------------
    # Output
    #
    # output_root/
    #     epithelial/
    #         whole_epithelial_image_complete.png
    #     lymphocyte/
    #         whole_lymphocyte_image_complete.png
    # -----------------------------------------------------

    patch_save_path = (
        output_root_dir
        / type_name
    )

    patch_save_path.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        patch_save_path
        / f"whole_{type_name}_image_complete.png" # Change name based on filename
    )

    print(f"Output: {output_path}")

    if output_path.exists():

        print(
            f"Output already exists. Skipping:\n"
            f"{output_path}"
        )

        return

    # =====================================================
    # RECONSTRUCTION SETTINGS
    # =====================================================

    missing_pixels = 50

    x = 50
    y = 50

    width = 1948
    height = 1948

    number_x = ceil(
        wsi_image.width / width
    )

    number_y = ceil(
        wsi_image.height / height
    )

    expected_patches = (
        number_x * number_y
    )

    print(
        f"Patch grid: {number_x} x {number_y}"
    )

    print(
        f"Expected patches: {expected_patches}"
    )

    print(
        f"Found patches: {len(image_list)}"
    )

    if len(image_list) != expected_patches:

        print(
            f"WARNING: Expected {expected_patches} "
            f"patches but found {len(image_list)}."
        )

    # =====================================================
    # LOAD AND CROP PATCHES
    # =====================================================

    region_list = []

    compare_idx = 2
    compare_region = None

    for i, image_path in enumerate(image_list):

        image = pyvips.Image.new_from_file(
            str(image_path)
        )

        region = image.crop(
            x,
            y,
            width,
            height
        )

        region_list.append(region)

        if i == compare_idx:

            print(
                f"Comparison patch: {image_path}"
            )

            compare_region = region.numpy()

    if not region_list:
        return

    # =====================================================
    # CREATE EMPTY WSI
    # =====================================================

    first_region = region_list[0]

    bands = first_region.bands

    print(f"Input bands: {bands}")

    canvas_width = (
        first_region.width
        * number_x
        + missing_pixels
    )

    canvas_height = (
        first_region.height
        * number_y
        + missing_pixels
    )

    new_image = pyvips.Image.black(
        canvas_width,
        canvas_height,
        bands=bands
    )

    # =====================================================
    # INSERT PATCHES
    # =====================================================

    for idx, img in enumerate(region_list):

        # Keep binary masks single-channel
        if bands == 1 and img.bands > 1:
            img = img.extract_band(0)

        elif bands == 3 and img.bands == 4:
            img = img.extract_band(
                0,
                n=3
            )

        x_position = (
            (idx % number_x)
            * width
            + missing_pixels
        )

        y_position = (
            (idx // number_x)
            * height
            + missing_pixels
        )

        new_image = new_image.insert(
            img,
            x_position,
            y_position
        )

    # =====================================================
    # CROP TO EXACT WSI DIMENSIONS
    # =====================================================

    new_image_cropped = new_image.crop(
        0,
        0,
        wsi_image.width,
        wsi_image.height
    )

    print(
        f"Output dimensions: "
        f"{new_image_cropped.width} x "
        f"{new_image_cropped.height}"
    )

    # =====================================================
    # CHECK ONE PATCH
    # =====================================================

    if (
        compare_region is not None
        and compare_idx < len(region_list)
    ):

        x0 = (
            missing_pixels
            + (compare_idx % number_x)
            * width
        )

        y0 = (
            missing_pixels
            + (compare_idx // number_x)
            * height
        )

        merged_region = (
            new_image_cropped.crop(
                x0,
                y0,
                width,
                height
            )
            .numpy()
        )

        print(
            "compare_region shape:",
            compare_region.shape
        )

        print(
            "merged_region shape:",
            merged_region.shape
        )

        print(
            "exact equal:",
            np.array_equal(
                merged_region,
                compare_region
            )
        )

        diff = (
            merged_region.astype(np.int16)
            -
            compare_region.astype(np.int16)
        )

        print(
            "max abs diff:",
            np.abs(diff).max()
        )

        if diff.ndim == 3:

            different_pixels = np.any(
                diff != 0,
                axis=2
            ).sum()

        else:

            different_pixels = (
                diff != 0
            ).sum()

        print(
            "different pixels:",
            different_pixels
        )

    # =====================================================
    # SAVE
    # =====================================================

    print(
        f"Saving:\n{output_path}"
    )

    new_image_cropped.write_to_file(
        str(output_path)
    )

    print(
        f"Finished type: {type_name}"
    )


if __name__ == "__main__":

    # =========================================================
    # INPUT
    #
    # Structure:
    #
    # binary_maps_separated_nuclei/
    #     epithelial/
    #         *.png
    #     lymphocyte/
    #         *.png
    #     neutrophil/
    #         *.png
    #     plasma/
    #         *.png
    #     eosinophil/
    #         *.png
    #     connective/
    #         *.png
    # =========================================================
    
    # Sample name and selected cell type
    sample = "Func044"
    cell_type = "dead"  # Change to the desired cell type

    type_dir = Path(
        f"/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/new/{sample}_ST_HE_40x_BF_01/binary_maps_separated_nuclei/{cell_type}/"
    )

    # Folder containing:
    #
    # {sample}_ST_HE_40x_BF_01.btf
    root_wsi_dir = Path(
        "/media/jenny/Expansion/HE/40x/images/"
    )

    # Output root
    output_root_dir = Path(
        f"/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/new/{sample}_ST_HE_40x_BF_01/wsi/"
    )

    # =========================================================
    # RUN
    # =========================================================

    create_wsi_single_folder(
        type_dir=type_dir,
        root_wsi_dir=root_wsi_dir,
        output_root_dir=output_root_dir,
        sample = sample,
    )