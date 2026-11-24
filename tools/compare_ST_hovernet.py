import os
import numpy as np
import pandas as pd
import pyvips
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

"""

Compare Spatial Transcriptomics data with HoverNet output

"""
# ============================================================
# MATLAB-COMPATIBLE ROUNDING
# ============================================================

def matlab_round(x):
    """
    MATLAB-compatible rounding.

    MATLAB rounds halfway values away from zero.
    """

    x = np.asarray(x)

    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ============================================================
# LOAD AND DOWNSCALE HE IMAGE USING PYVIPS
# ============================================================

def load_he_pyvips(
    he_path,
    scale_factor=0.1,
    mask_path=None
):
    """
    Load and downscale a large H&E image using pyvips.

    The full-resolution image is not converted into
    a NumPy array.

    Parameters
    ----------
    he_path : str
        Path to the registered H&E image.

    scale_factor : float
        Image scaling factor for visualization.

    mask_path : str or None
        Optional binary tissue mask. Pixels outside
        the mask are made white.

    Returns
    -------
    he_small : ndarray
        Downscaled RGB H&E image.

    original_shape : tuple
        Original image dimensions (H, W).
    """

    if not os.path.isfile(he_path):
        raise FileNotFoundError(
            f"H&E image not found: {he_path}"
        )

    # --------------------------------------------------------
    # LOAD IMAGE WITH PYVIPS
    # --------------------------------------------------------

    he = pyvips.Image.new_from_file(
        he_path,
        access="sequential"
    )

    H = he.height
    W = he.width

    print(f"  Original H&E size: {W} x {H}")

    # --------------------------------------------------------
    # ENSURE RGB
    # --------------------------------------------------------

    if he.bands == 1:

        he = he.bandjoin([he, he])

    elif he.bands > 3:

        he = he.extract_band(0, n=3)

    # --------------------------------------------------------
    # OPTIONAL TISSUE MASK
    # --------------------------------------------------------

    if mask_path is not None:

        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Tissue mask not found: {mask_path}"
            )

        mask = pyvips.Image.new_from_file(
            mask_path,
            access="sequential"
        )

        if mask.bands > 1:
            mask = mask.extract_band(0)

        if mask.width != W or mask.height != H:

            raise ValueError(
                "H&E image and tissue mask have "
                "different dimensions. The mask must "
                "be aligned before applying it."
            )

        # White outside the tissue
        mask = mask > 0

        he = mask.ifthenelse(
            he,
            [255, 255, 255]
        )

    # --------------------------------------------------------
    # DOWNSCALE USING PYVIPS
    # --------------------------------------------------------

    he_small = he.resize(
        scale_factor,
        kernel="lanczos3"
    )

    # Convert to 8-bit RGB for plotting.
    # Assumes the source H&E uses 8-bit intensity values.
    he_small = he_small.cast("uchar")

    # --------------------------------------------------------
    # CONVERT ONLY DOWNSCALED IMAGE TO NUMPY
    # --------------------------------------------------------

    he_np = np.frombuffer(
        he_small.write_to_memory(),
        dtype=np.uint8
    ).reshape(
        he_small.height,
        he_small.width,
        he_small.bands
    ).copy()

    print(
        f"  Downscaled H&E size: "
        f"{he_small.width} x {he_small.height}"
    )

    return he_np, (H, W)


# ============================================================
# GET COLOR LIMITS
# ============================================================

def get_color_limits(values, shared_scale=False):
    """
    Return suitable color limits for the heatmap.
    """

    if shared_scale:
        return (0, 1)

    values = np.asarray(values, dtype=float)

    valid = values[np.isfinite(values)]

    if len(valid) == 0:
        return (0, 1)

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    if vmin == vmax:

        if vmin == 0:
            return (0, 1)

        delta = abs(vmin) * 0.1

        if delta == 0:
            delta = 0.1

        return (
            vmin - delta,
            vmax + delta
        )

    return (vmin, vmax)


# ============================================================
# PLOT SPOT HEATMAP
# ============================================================

def plot_circle_heatmap(
    ax,
    HE,
    x,
    y,
    spot_radius,
    values,
    original_shape,
    color_limits,
    title,
    colorbar_label,
    alpha=1,
    matlab_coordinates=True
):
    """
    Plot circular ST spots over an H&E image.

    Both ST and HoverNet use identical spot positions.

    The circles represent the original ST spot radius,
    while HoverNet fractions were calculated using
    square sampling regions.
    """

    H_full, W_full = original_shape
    H_small, W_small = HE.shape[:2]

    scale_x = W_small / W_full
    scale_y = H_small / H_full

    ax.imshow(HE)

    cmap = plt.get_cmap("turbo")

    norm = Normalize(
        vmin=color_limits[0],
        vmax=color_limits[1]
    )

    circles = []
    valid_values = []

    for j in range(len(x)):

        if (
            not np.isfinite(values[j])
            or not np.isfinite(x[j])
            or not np.isfinite(y[j])
        ):
            continue

        # Match MATLAB rounding
        cx = int(matlab_round(x[j]))
        cy = int(matlab_round(y[j]))

        # Convert MATLAB to Python indexing
        if matlab_coordinates:
            cx -= 1
            cy -= 1

        # Scale coordinates to downsampled H&E
        cx_scaled = (cx + 0.5) * scale_x - 0.5
        cy_scaled = (cy + 0.5) * scale_y - 0.5

        # Scale spot radius
        radius_scaled = spot_radius * scale_x

        # Create circle
        circle = Circle(
            (cx_scaled, cy_scaled),
            radius=radius_scaled
        )

        circles.append(circle)
        valid_values.append(values[j])

    collection = PatchCollection(
        circles,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        edgecolor="none"
    )

    collection.set_array(
        np.asarray(valid_values, dtype=float)
    )

    ax.add_collection(collection)

    ax.set_xlim(-0.5, W_small - 0.5)
    ax.set_ylim(H_small - 0.5, -0.5)

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    cb = plt.colorbar(
        collection,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.04
    )

    cb.set_label(
        colorbar_label,
        fontsize=12
    )

    cb.ax.tick_params(labelsize=11)

    return ax

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # DIRECTORIES
    # ========================================================

    DIR0 = (
        "/media/jenny/Expansion/jenny_funcprost/"
    )

    # Co-registered ST coordinates
    DIR_SPATIAL = os.path.join(
        DIR0,
        "conic/results/New/Fra_Vilde/ST_data/Co-reg_coordinates/"
    )

    # HoverNet quantification files from Script 1
    DIR_DATA = os.path.join(
        DIR0,
        "pannuke/results/zhang_original_weights/New/quantification_jenny/data/"
    )

    # Spatial transcriptomics cell-type abundance files
    # CHANGE THIS to your actual ST data directory
    DIR_ST = "/media/jenny/Expansion/jenny_funcprost/conic/results/New/Fra_Vilde/ST_data/SpaCET/Data/excel/"

    # Registered H&E images
    # CHANGE THIS to your actual H&E image directory
    DIR_HE = "/media/jenny/Expansion1/Prostata_Vilde/Co-reg_images_10x/"

    # Optional H&E tissue masks
    # Set to None if you do not want to use tissue masks
    DIR_MASK = None

    # Save comparison figures
    DIR_SAVE = os.path.join(
        DIR0,
        "pannuke/results/zhang_original_weights/New/quantification_jenny/figures/ST_HoverNet/"
    )

    os.makedirs(
        DIR_SAVE,
        exist_ok=True
    )

    # ========================================================
    # SETTINGS
    # ========================================================

    NAME_LIST = [
        "Func050",
        "Func116"
    ]

    # ST cell type : HoverNet cell type
    COMPARISONS = {
        "CAF": "connective",
        "Malignant": "neoplastic"
    }

    # H&E downscaling
    SCALE_FACTOR = 0.1

    # Spot transparency
    SPOT_ALPHA = 0.85

    # Same color limits for ST and HoverNet
    # Set True only if both represent fractions [0, 1].
    SHARED_COLOR_SCALE = False

    # MATLAB-style co-registered coordinates
    MATLAB_COORDINATES = True

    # CSV filename patterns
    ST_FILENAME = "{name}_SpaCET_deconvolution.xlsx"

    QUANT_FILENAME = "{name}_Conic_SpotQuantification.csv"

    # H&E and optional mask filename patterns
    HE_FILENAME = "{name}_HE.tif"

    MASK_FILENAME = "{name} HE_Mask.tif"

    # ========================================================
    # LOAD SPOT RADII
    # ========================================================

    radius_table = pd.read_csv(
        os.path.join(
            DIR_SPATIAL,
            "spot_radius_coreg.csv"
        )
    )

    # ========================================================
    # PROCESS ALL BIOPSIES
    # ========================================================

    for name in NAME_LIST:

        print("\n" + "=" * 60)
        print(f"Processing {name}")
        print("=" * 60)

        # ----------------------------------------------------
        # 1. LOAD HOVERNET QUANTIFICATION
        # ----------------------------------------------------

        quant_path = os.path.join(
            DIR_DATA,
            QUANT_FILENAME.format(name=name)
        )

        if not os.path.isfile(quant_path):
            raise FileNotFoundError(
                f"Quantification file not found:\n{quant_path}"
            )

        quant = pd.read_csv(quant_path)

        quant["barcode"] = (
            quant["barcode"]
            .astype(str)
            .str.strip()
        )

        if not quant["barcode"].is_unique:
            raise ValueError(
                f"Duplicate HoverNet barcodes: {name}"
            )

        print(
            f"  HoverNet spots: {len(quant)}"
        )

        # ----------------------------------------------------
        # 2. LOAD ST CELL-TYPE ABUNDANCE
        # ----------------------------------------------------

        st_path = os.path.join(
            DIR_ST,
            ST_FILENAME.format(name=name)
        )

        if not os.path.isfile(st_path):
            raise FileNotFoundError(
                f"ST file not found:\n{st_path}"
            )

        ST = pd.read_excel(st_path, engine = "calamine")

        ST["barcode"] = (
            ST["barcode"]
            .astype(str)
            .str.strip()
        )

        if not ST["barcode"].is_unique:
            raise ValueError(
                f"Duplicate ST barcodes: {name}"
            )

        print(
            f"  ST spots: {len(ST)}"
        )

        # ----------------------------------------------------
        # 3. MATCH BY BARCODE
        # ----------------------------------------------------

        # Keep HoverNet coordinates.
        # Add ST cell-type values to the same spots.

        merged = quant.merge(
            ST,
            on="barcode",
            how="left",
            validate="one_to_one",
            suffixes=("_HoverNet", "_ST"),
            indicator=True
        )

        matched = (
            merged["_merge"] == "both"
        ).sum()

        print(
            f"  Matched barcodes: "
            f"{matched}/{len(merged)}"
        )

        merged = merged.drop(
            columns="_merge"
        )

        # ----------------------------------------------------
        # 4. GET SPOT RADIUS
        # ----------------------------------------------------

        radius_row = radius_table.loc[
            radius_table["nameList"].astype(str) == name,
            "spot_radius_coreg"
        ]

        if radius_row.empty:
            raise ValueError(
                f"Missing spot radius: {name}"
            )

        spot_radius = float(
            radius_row.iloc[0]
        )

        # EXACT MATLAB SQUARE CALCULATION
        half_side = int(
            matlab_round(
                spot_radius + 0.3 * spot_radius
            )
        )

        square_size = 2 * half_side + 1

        print(
            f"  Spot radius: {spot_radius:.2f}"
        )

        print(
            f"  Square size: "
            f"{square_size} x {square_size} pixels"
        )

        # ----------------------------------------------------
        # 5. LOAD HE WITH PYVIPS
        # ----------------------------------------------------

        he_path = os.path.join(
            DIR_HE,
            HE_FILENAME.format(name=name)
        )

        if DIR_MASK is not None:

            mask_path = os.path.join(
                DIR_MASK,
                MASK_FILENAME.format(name=name)
            )

        else:

            mask_path = None

        HE, original_shape = load_he_pyvips(
            he_path=he_path,
            scale_factor=SCALE_FACTOR,
            mask_path=mask_path
        )

        # ----------------------------------------------------
        # 6. EXTRACT SPOT COORDINATES
        # ----------------------------------------------------

        x = merged[
            "x_coreg"
        ].to_numpy(dtype=float)

        y = merged[
            "y_coreg"
        ].to_numpy(dtype=float)

        # ----------------------------------------------------
        # 7. GENERATE COMPARISON FIGURES
        # ----------------------------------------------------

        for st_type, hovernet_type in COMPARISONS.items():

            print(
                f"\n  Comparing {st_type} "
                f"vs {hovernet_type}"
            )

            # Check columns
            if st_type not in merged.columns:
                raise KeyError(
                    f"Missing ST column: {st_type}"
                )

            if hovernet_type not in merged.columns:
                raise KeyError(
                    f"Missing HoverNet column: {hovernet_type}"
                )

            # ------------------------------------------------
            # ST VALUES
            # ------------------------------------------------

            st_values = pd.to_numeric(
                merged[st_type],
                errors="coerce"
            ).to_numpy(dtype=float)

            # ------------------------------------------------
            # HOVERNET VALUES
            # ------------------------------------------------

            hovernet_values = pd.to_numeric(
                merged[hovernet_type],
                errors="coerce"
            ).to_numpy(dtype=float)

            # ------------------------------------------------
            # COLOR LIMITS
            # ------------------------------------------------

            clim_st = get_color_limits(
                st_values,
                shared_scale=SHARED_COLOR_SCALE
            )

            clim_hovernet = get_color_limits(
                hovernet_values,
                shared_scale=SHARED_COLOR_SCALE
            )

            # HoverNet values are fractions.
            if not SHARED_COLOR_SCALE:
                clim_hovernet = (0, 1)

            # ------------------------------------------------
            # CREATE FIGURE
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(16, 8),
                facecolor="white"
            )

            # ------------------------------------------------
            # ST HEATMAP
            # ------------------------------------------------

            plot_circle_heatmap(
            ax=axes[0],
            HE=HE,
            x=x,
            y=y,
            spot_radius=spot_radius,
            values=st_values,
            original_shape=original_shape,
            color_limits=clim_st,
            title=f"ST - {st_type}",
            colorbar_label="ST abundance",
            alpha=SPOT_ALPHA,
            matlab_coordinates=MATLAB_COORDINATES
        )

            # ------------------------------------------------
            # HOVERNET HEATMAP
            # ------------------------------------------------

            plot_circle_heatmap(
            ax=axes[1],
            HE=HE,
            x=x,
            y=y,
            spot_radius=spot_radius,
            values=hovernet_values,
            original_shape=original_shape,
            color_limits=clim_hovernet,
            title=f"HoverNet - {hovernet_type}",
            colorbar_label="Cell fraction",
            alpha=SPOT_ALPHA,
            matlab_coordinates=MATLAB_COORDINATES
        )

            # ------------------------------------------------
            # FIGURE TITLE
            # ------------------------------------------------

            fig.suptitle(
                f"{name}: {st_type} vs {hovernet_type}",
                fontsize=19,
                fontweight="bold"
            )

            plt.tight_layout()

            # ------------------------------------------------
            # SAVE FIGURE
            # ------------------------------------------------

            output_path = os.path.join(
                DIR_SAVE,
                f"{name}_{st_type}_vs_{hovernet_type}.png"
            )

            fig.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white"
            )

            plt.close(fig)

            print(
                f"  Saved figure: {output_path}"
            )

            # ------------------------------------------------
            # SAVE MATCHED DATA AS CSV
            # ------------------------------------------------

            comparison_data = pd.DataFrame({

                "barcode": merged["barcode"],

                "x_coreg": x,

                "y_coreg": y,

                f"ST_{st_type}": st_values,

                f"HoverNet_{hovernet_type}": hovernet_values

            })

            csv_path = os.path.join(
                DIR_SAVE,
                f"{name}_{st_type}_vs_{hovernet_type}.csv"
            )

            comparison_data.to_csv(
                csv_path,
                index=False
            )

            print(
                f"  Saved data: {csv_path}"
            )

        # Release image array before next biopsy
        del HE

    print("\nFinished all comparisons.")