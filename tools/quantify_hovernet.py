import os
import numpy as np
import pandas as pd
import pyvips

from scipy.ndimage import label

"""
Quantify results from HoVerNet by calculating cell fractions for the same spatial spots as used in the Spatial Transcrips data
"""

# ============================================================
# DIRECTORIES
# ============================================================

DIR0 = "/media/jenny/Expansion/jenny_funcprost/"

DIR_SPATIAL = os.path.join(
    DIR0,
    "conic/results/New/Fra_Vilde/ST_data/Co-reg_coordinates/"
)

DIR_BW = os.path.join(
    DIR0,
    "pannuke/results/zhang_original_weights/New/"
)

DIR_SAVE = os.path.join(
    DIR0,
    "pannuke/results/zhang_original_weights/New/quantification_jenny/data/"
)

os.makedirs(DIR_SAVE, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NAME_LIST = [
    "Func050",
    "Func116"
]

CELL_TYPES = [
    "epithelial",
    "connective",
    "dead",
    "inflammatory",
    "neoplastic",
]


# ============================================================
# MATLAB-COMPATIBLE ROUNDING
# ============================================================

def matlab_round(x):
    """
    MATLAB-compatible rounding.
    Halfway values are rounded away from zero.
    """

    x = np.asarray(x)

    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ============================================================
# FUNCTION: CONVERT SMALL PYVIPS REGION TO NUMPY
# ============================================================

def vips_to_numpy(image):
    """
    Convert a small pyvips image region into a NumPy array.

    This function should only be called on cropped regions,
    not full-resolution whole-slide images.
    """

    # Ensure single channel
    if image.bands > 1:
        image = image.extract_band(0)

    # Convert to uint8
    image = image.cast("uchar")

    # Convert region to NumPy
    array = np.frombuffer(
        image.write_to_memory(),
        dtype=np.uint8
    ).reshape(
        image.height,
        image.width
    ).copy()

    return array


# ============================================================
# FUNCTION: COUNT FRACTIONS WITHIN SQUARE SPOTS
# ============================================================

def count_fraction(
    BW1,
    BW2,
    x,
    y,
    spot_radius
):

    """
    Python / pyvips equivalent of MATLAB funcCountFraction.

    BW1:
        pyvips binary map for one cell type.

    BW2:
        pyvips total binary cell map.

    x, y:
        MATLAB-style co-registered spot coordinates.

    spot_radius:
        Spot radius in registered image pixels.

    Returns:
        Cell fractions per spot.

    Each spot uses a square region defined by:
        half_side = round(spot_radius + 0.3*spot_radius)
    """

    fractions = np.full(
        len(x),
        np.nan,
        dtype=float
    )

    half_side = int(
        matlab_round(
            spot_radius + 0.3 * spot_radius
        )
    )

    H = BW1.height
    W = BW1.width

    # MATLAB 2D connected-component counting
    # uses 8-connectivity by default.
    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    for j in range(len(x)):

        if (
            not np.isfinite(x[j])
            or not np.isfinite(y[j])
        ):
            continue

        # ----------------------------------------------------
        # SPOT COORDINATES
        # ----------------------------------------------------

        # Convert MATLAB indexing to Python indexing
        cx = int(matlab_round(x[j])) - 1
        cy = int(matlab_round(y[j])) - 1

        # ----------------------------------------------------
        # SQUARE BOUNDARIES
        # ----------------------------------------------------

        x1 = max(
            cx - half_side,
            0
        )

        x2 = min(
            cx + half_side + 1,
            W
        )

        y1 = max(
            cy - half_side,
            0
        )

        y2 = min(
            cy + half_side + 1,
            H
        )

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            continue

        # ----------------------------------------------------
        # EXTRACT SMALL REGIONS USING PYVIPS
        # ----------------------------------------------------

        BW1_crop = BW1.crop(
            x1,
            y1,
            width,
            height
        )

        BW2_crop = BW2.crop(
            x1,
            y1,
            width,
            height
        )

        # ----------------------------------------------------
        # CONVERT CROPS TO NUMPY
        # ----------------------------------------------------

        BW1_np = vips_to_numpy(BW1_crop) > 0
        BW2_np = vips_to_numpy(BW2_crop) > 0

        # ----------------------------------------------------
        # COUNT CONNECTED COMPONENTS
        # ----------------------------------------------------

        _, n_positive = label(
            BW1_np,
            structure=structure
        )

        _, n_total = label(
            BW2_np,
            structure=structure
        )

        # ----------------------------------------------------
        # CALCULATE FRACTION
        # ----------------------------------------------------

        if n_total > 0:

            fractions[j] = (
                n_positive / n_total
            )

    return fractions


# ============================================================
# LOAD SPOT RADII
# ============================================================

radius_table = pd.read_csv(
    os.path.join(
        DIR_SPATIAL,
        "spot_radius_coreg.csv"
    )
)


# ============================================================
# PROCESS ALL SAMPLES
# ============================================================

for name in NAME_LIST:

    print(f"\nProcessing {name}...")

    # --------------------------------------------------------
    # LOAD SPOT COORDINATES
    # --------------------------------------------------------

    spots = pd.read_csv(
        os.path.join(
            DIR_SPATIAL,
            f"{name}_Spot_Coordinates_CoReg.csv"
        )
    )

    x = spots["x_coreg"].to_numpy(
        dtype=float
    )

    y = spots["y_coreg"].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # GET SPOT RADIUS
    # --------------------------------------------------------

    radius_row = radius_table.loc[
        radius_table["nameList"].astype(str) == name,
        "spot_radius_coreg"
    ]

    if radius_row.empty:
        raise ValueError(
            f"Missing spot radius for {name}"
        )

    spot_radius = float(
        radius_row.iloc[0]
    )

    half_side = int(
        matlab_round(
            spot_radius + 0.3 * spot_radius
        )
    )

    print(
        f"Spot radius: {spot_radius:.2f} pixels"
    )

    print(
        f"Square size: "
        f"{2 * half_side + 1} x "
        f"{2 * half_side + 1} pixels"
    )

    print(
        f"Number of spots: {len(spots)}"
    )

    # --------------------------------------------------------
    # DEFINE OUTPUT TABLE
    # --------------------------------------------------------

    result = spots[
        [
            "barcode",
            "x_coreg",
            "y_coreg"
        ]
    ].copy()

    # --------------------------------------------------------
    # BINARY MAP DIRECTORY
    # --------------------------------------------------------

    bw_folder = os.path.join(
        DIR_BW,
        f"{name}_ST_HE_40x_BF_01",
        "co_registration",
        "registered_nuclei_maps"
    )

    # --------------------------------------------------------
    # LOAD BINARY MAPS WITH PYVIPS
    # --------------------------------------------------------

    binary_maps = {}

    for cell_type in CELL_TYPES:

        path = os.path.join(
            bw_folder,
            f"{cell_type}_registered_binary.png"
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing binary map: {path}"
            )

        BW = pyvips.Image.new_from_file(
            path,
            access="random"
        )

        # Extract first channel if necessary
        if BW.bands > 1:
            BW = BW.extract_band(0)

        # Ensure binary values 0 and 255
        BW = (BW > 0).cast("uchar") * 255

        binary_maps[cell_type] = BW

        print(
            f"Loaded {cell_type}: "
            f"{BW.width} x {BW.height}"
        )

    # --------------------------------------------------------
    # CHECK IMAGE DIMENSIONS
    # --------------------------------------------------------

    first_map = binary_maps[CELL_TYPES[0]]

    W = first_map.width
    H = first_map.height

    for cell_type, BW in binary_maps.items():

        if BW.width != W or BW.height != H:

            raise ValueError(
                f"Image dimensions differ for "
                f"{cell_type} in {name}"
            )

    # --------------------------------------------------------
    # CREATE TOTAL BINARY MAP
    # --------------------------------------------------------

    print("Creating total cell map...")

    BW_Total = binary_maps[CELL_TYPES[0]]

    for cell_type in CELL_TYPES[1:]:

        BW_Total = BW_Total | binary_maps[cell_type]

    # The total map remains a lazy pyvips image.
    # It is not converted to a full NumPy array.

    # --------------------------------------------------------
    # CALCULATE CELL FRACTIONS
    # --------------------------------------------------------

    for cell_type in CELL_TYPES:

        print(
            f"  Counting {cell_type}..."
        )

        BW = binary_maps[cell_type]

        fractions = count_fraction(
            BW1=BW,
            BW2=BW_Total,
            x=x,
            y=y,
            spot_radius=spot_radius
        )

        result[cell_type] = fractions

        n_valid = np.isfinite(
            fractions
        ).sum()

        print(
            f"    Valid spots: "
            f"{n_valid}/{len(fractions)}"
        )

    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    output_path = os.path.join(
        DIR_SAVE,
        f"{name}_Conic_SpotQuantification.csv"
    )

    result.to_csv(
    output_path,
    index=False
    )

    print(
        f"Saved: {output_path}"
    )

    # Release references between biopsies
    del binary_maps
    del BW_Total


print("\nFinished quantifying binary maps.")