import os
import gc
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Co-register H&E image to Visium image
"""
# ============================================================
# USER SETTINGS
# ============================================================
fixed_path = "/media/jenny/Expansion1/Prostata_Vilde/Co-reg_images_10x/Func050_HE.tif" #Fixed image path
moving_path = "/media/jenny/Expansion/HE/3_4_20x/images/Func050_ST_HE_20x_BF_01_dowsampled_three_fourth.png" #Moving image path

output_dir = "/media/jenny/Expansion/jenny_funcprost/conic/results/correct_nuclei/Func050_ST_HE_20x_BF_01/testtest_co_registration/"


os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

scaled_path = os.path.join(
    output_dir,
    "01_scaled.png"
)

manual_path = os.path.join(
    output_dir,
    "02_manual_registered.png"
)

manual_overlay_path = os.path.join(
    output_dir,
    "03_manual_overlay.png"
)

fine_path = os.path.join(
    output_dir,
    "04_fine_registered.png"
)

fine_overlay_path = os.path.join(
    output_dir,
    "05_fine_overlay.png"
)

diagnostics_path = os.path.join(
    output_dir,
    "registration_diagnostics.json"
)

scale_matrix_path = os.path.join(
    output_dir,
    "scale_matrix.npy"
)

manual_matrix_path = os.path.join(
    output_dir,
    "manual_matrix.npy"
)

fine_residual_matrix_path = os.path.join(
    output_dir,
    "fine_residual_matrix.npy"
)

final_matrix_path = os.path.join(
    output_dir,
    "final_matrix.npy"
)


# ============================================================
# MANUAL ALIGNMENT PARAMETERS
# ============================================================

# # Func 044
# manual_angle = -10

# manual_tx = -80
# manual_ty = 0

# # Func 043
# manual_angle = 5

# manual_tx = -30
# manual_ty = 0

# Func 050
manual_angle = -47

manual_tx = -100
manual_ty = 0

# # Func 116
# manual_angle = 8

# manual_tx = -100
# manual_ty = 0



# ============================================================
# REGISTRATION SETTINGS
# ============================================================

# Fine registration is performed only on a thumbnail.
#
# 1200-2000 is generally sufficient when the manual alignment
# is already reasonably good.
MAX_THUMBNAIL_DIM = 1600


# ------------------------------------------------------------
# Maximum fine correction allowed
#
# These prevent affine ECC from producing crazy distortions.
# ------------------------------------------------------------

MAX_FINE_ROTATION_DEG = 4.0

# Maximum change from scale = 1.0.
#
# 0.05 means approximately:
#
#     0.95 - 1.05
#
MAX_FINE_SCALE_CHANGE = 0.05


# Maximum departure from a 90 degree angle between transformed
# X and Y axes.
#
# This limits shear.
MAX_FINE_SHEAR_DEG = 4.0


# Maximum residual translation as fraction of image diagonal
MAX_FINE_TRANSLATION_FRACTION = 0.05


# ------------------------------------------------------------
# Acceptance requirements
# ------------------------------------------------------------

# Fine alignment must improve NCC by at least this amount
MIN_IMPROVEMENT = 0.003


# Do not accept an affine transform if it destroys too much
# of the original tissue overlap.
#
# 0.85 = retain at least 85% of initial overlap
MIN_OVERLAP_RETENTION = 0.85


# ============================================================
# DISPLAY
# ============================================================

SHOW_THUMBNAIL_PREVIEWS = True


# ============================================================
# BASIC MATRIX FUNCTIONS
# ============================================================

def to_homogeneous(M):
    """
    Convert a 2x3 affine matrix into a 3x3 homogeneous matrix.
    """

    M3 = np.eye(
        3,
        dtype=np.float64
    )

    M3[:2, :] = M

    return M3


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_bgr(path, image):
    """
    Save OpenCV BGR image and verify that writing succeeded.
    """

    print(
        f"Saving:\n{path}"
    )

    success = cv2.imwrite(
        path,
        image
    )

    if not success:
        raise IOError(
            f"Could not save image:\n{path}"
        )


# ============================================================
# THUMBNAIL
# ============================================================

def make_thumbnail(
    image,
    max_dimension=1600
):
    """
    Make a low-memory thumbnail while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_dimension / max(height, width)
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    thumbnail = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return thumbnail


# ============================================================
# H&E REGISTRATION CHANNEL
# ============================================================

def he_registration_channel(bgr):
    """
    Low-memory structural channel for H&E registration.

    Uses the inverted green channel.

    Hematoxylin has strong contrast in the green channel,
    making it useful for registration without requiring
    memory-intensive full HED stain deconvolution.

    IMPORTANT:
    This function is only used on thumbnails.
    """

    green = bgr[:, :, 1]

    channel = (
        255 - green
    ).astype(np.uint8)

    # Improve local tissue contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    channel = clahe.apply(
        channel
    )

    # Slight blur emphasizes tissue architecture rather than
    # individual nuclei.
    channel = cv2.GaussianBlur(
        channel,
        (0, 0),
        sigmaX=1.2,
        sigmaY=1.2
    )

    return (
        channel.astype(np.float32) / 255.0
    )


# ============================================================
# TISSUE MASK
# ============================================================

def tissue_mask(bgr):
    """
    Generate approximate tissue mask for H&E.

    Removes:
        - white glass/background
        - black invalid pixels

    Used only on thumbnails.
    """

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue = (
        (saturation > 12)
        |
        (value < 235)
    )

    # Remove completely black pixels/padding
    valid = (
        np.max(
            bgr,
            axis=2
        ) > 10
    )

    tissue &= valid

    mask = (
        tissue.astype(np.uint8) * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# ============================================================
# NORMALIZED CROSS CORRELATION
# ============================================================

def masked_ncc(
    fixed,
    moving,
    mask
):
    """
    Normalized cross correlation inside tissue overlap.

    Higher = better.
    """

    valid = mask > 0

    if np.count_nonzero(valid) < 500:
        return -1.0

    a = fixed[valid].astype(
        np.float32
    )

    b = moving[valid].astype(
        np.float32
    )

    a -= a.mean()
    b -= b.mean()

    denominator = np.sqrt(
        np.sum(a * a)
        *
        np.sum(b * b)
    )

    if denominator < 1e-10:
        return -1.0

    return float(
        np.sum(a * b)
        /
        denominator
    )


# ============================================================
# SCALE TRANSFORMATION
# ============================================================

def build_scale_transform(
    fixed_shape,
    moving_shape
):
    """
    Scale original moving image to fixed-image dimensions.

    WARNING:
    This assumes differences in image dimensions correspond to
    differences in scale.

    If you know scanner microns/pixel, physical pixel size should
    ideally be used instead.
    """

    fixed_h, fixed_w = fixed_shape[:2]
    moving_h, moving_w = moving_shape[:2]

    scale_x = (
        fixed_w / moving_w
    )

    scale_y = (
        fixed_h / moving_h
    )

    S = np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )

    return S


# ============================================================
# MANUAL TRANSFORMATION
# ============================================================

def build_manual_transform(
    fixed_shape,
    moving_shape,
    angle,
    tx,
    ty
):
    """
    Construct transformation:

        original moving
              ↓
            scale
              ↓
           rotation
              ↓
         translation
              ↓
         fixed canvas

    The image itself is NOT resampled here.
    """

    fixed_h, fixed_w = fixed_shape[:2]

    scale_matrix = build_scale_transform(
        fixed_shape,
        moving_shape
    )

    center = (
        fixed_w / 2.0,
        fixed_h / 2.0
    )

    rotation_2x3 = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotation_2x3[0, 2] += tx
    rotation_2x3[1, 2] += ty

    rotation_matrix = to_homogeneous(
        rotation_2x3
    )

    # First scaling, then rotation/translation
    manual_matrix = (
        rotation_matrix
        @
        scale_matrix
    )

    return manual_matrix


# ============================================================
# ANALYZE AFFINE TRANSFORMATION
# ============================================================

def analyze_affine(M):
    """
    Extract interpretable information from affine correction.

    The fine transform can contain:

        rotation
        scale X
        scale Y
        shear
        translation

    Singular values are also calculated because they provide
    a robust measure of total stretching/compression.
    """

    A = M[:2, :2]

    tx = float(
        M[0, 2]
    )

    ty = float(
        M[1, 2]
    )

    # --------------------------------------------------------
    # Determinant
    # --------------------------------------------------------

    determinant = float(
        np.linalg.det(A)
    )

    # --------------------------------------------------------
    # Column norms:
    # approximate X/Y scale
    # --------------------------------------------------------

    column_x = A[:, 0]
    column_y = A[:, 1]

    scale_x = float(
        np.linalg.norm(column_x)
    )

    scale_y = float(
        np.linalg.norm(column_y)
    )

    # --------------------------------------------------------
    # Shear
    #
    # Orthogonal axes = 90 degrees.
    # --------------------------------------------------------

    denom = (
        np.linalg.norm(column_x)
        *
        np.linalg.norm(column_y)
    )

    if denom > 1e-12:

        cos_angle = (
            np.dot(
                column_x,
                column_y
            )
            /
            denom
        )

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0
        )

        axis_angle = np.degrees(
            np.arccos(cos_angle)
        )

        shear_deg = abs(
            90.0 - axis_angle
        )

    else:

        axis_angle = np.nan
        shear_deg = np.inf

    # --------------------------------------------------------
    # Polar decomposition for robust rotation estimate
    # --------------------------------------------------------

    U, singular_values, Vt = np.linalg.svd(
        A
    )

    R = (
        U @ Vt
    )

    if np.linalg.det(R) < 0:

        U[:, -1] *= -1

        R = (
            U @ Vt
        )

    rotation_deg = np.degrees(
        np.arctan2(
            R[1, 0],
            R[0, 0]
        )
    )

    translation_px = float(
        np.hypot(
            tx,
            ty
        )
    )

    return {
        "rotation_deg":
            float(rotation_deg),

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "principal_scale_1":
            float(singular_values[0]),

        "principal_scale_2":
            float(singular_values[1]),

        "shear_deg":
            float(shear_deg),

        "axis_angle_deg":
            float(axis_angle),

        "tx":
            tx,

        "ty":
            ty,

        "translation_px":
            translation_px,

        "determinant":
            determinant
    }


# ============================================================
# PREVIEW
# ============================================================

def show_thumbnail_overlay(
    fixed_thumb,
    moving_thumb,
    title
):
    """
    Show registration preview without making a full-resolution
    overlay.
    """

    overlay = cv2.addWeighted(
        fixed_thumb,
        0.5,
        moving_thumb,
        0.5,
        0
    )

    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    fig = plt.figure(
        figsize=(11, 8)
    )

    plt.imshow(
        overlay_rgb
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()
    plt.show()

    plt.close(fig)

    del overlay
    del overlay_rgb

    gc.collect()


# ============================================================
# AFFINE FINE REGISTRATION
# ============================================================

def estimate_fine_affine(
    fixed_thumb,
    moving_manual_thumb,
    full_fixed_shape,
    thumbnail_to_full_matrix,
    full_to_thumbnail_matrix,
    max_rotation_deg=4.0,
    max_scale_change=0.05,
    max_shear_deg=4.0,
    max_translation_fraction=0.05,
    min_improvement=0.003,
    min_overlap_retention=0.85
):
    """
    Estimate an AFFINE residual correction using ECC.

    Unlike rigid registration, this permits:

        translation
        rotation
        scaling X
        scaling Y
        shear

    The correction is only accepted if both:

        1. registration quality improves
        2. affine distortion remains anatomically plausible
    """

    thumb_h, thumb_w = fixed_thumb.shape[:2]

    fixed_h, fixed_w = full_fixed_shape[:2]

    # --------------------------------------------------------
    # Registration channels
    # --------------------------------------------------------

    fixed_reg = he_registration_channel(
        fixed_thumb
    )

    moving_reg = he_registration_channel(
        moving_manual_thumb
    )

    # --------------------------------------------------------
    # Tissue masks
    # --------------------------------------------------------

    fixed_mask = tissue_mask(
        fixed_thumb
    )

    moving_mask = tissue_mask(
        moving_manual_thumb
    )

    initial_overlap = cv2.bitwise_and(
        fixed_mask,
        moving_mask
    )

    # Remove uncertain mask boundaries
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    initial_overlap = cv2.erode(
        initial_overlap,
        kernel
    )

    initial_overlap_pixels = np.count_nonzero(
        initial_overlap
    )

    # --------------------------------------------------------
    # Suppress blank background
    # --------------------------------------------------------

    fixed_reg[
        fixed_mask == 0
    ] = 0

    moving_reg[
        moving_mask == 0
    ] = 0

    # --------------------------------------------------------
    # Initial manual alignment score
    # --------------------------------------------------------

    initial_score = masked_ncc(
        fixed_reg,
        moving_reg,
        initial_overlap
    )

    print(
        "\nManual registration NCC:",
        initial_score
    )

    # --------------------------------------------------------
    # Identity starting point
    # --------------------------------------------------------

    warp_ecc = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )

    # --------------------------------------------------------
    # Full affine ECC
    # --------------------------------------------------------

    try:

        ecc_score, warp_ecc = cv2.findTransformECC(
            templateImage=fixed_reg,
            inputImage=moving_reg,
            warpMatrix=warp_ecc,

            # THIS is the important change
            motionType=cv2.MOTION_AFFINE,

            criteria=criteria,
            inputMask=initial_overlap,
            gaussFiltSize=5
        )

    except cv2.error as e:

        diagnostics = {
            "accepted": False,
            "reason": "ECC affine registration failed",
            "initial_score": float(initial_score)
        }

        print(
            "\nECC affine registration failed:"
        )

        print(e)

        identity = np.eye(
            3,
            dtype=np.float64
        )

        return (
            identity,
            diagnostics
        )

    # --------------------------------------------------------
    # ECC returns a matrix normally applied using
    # WARP_INVERSE_MAP.
    #
    # Convert it to conventional:
    #
    # manual moving -> fixed
    # --------------------------------------------------------

    fine_thumb_2x3 = cv2.invertAffineTransform(
        warp_ecc
    )

    fine_thumb = to_homogeneous(
        fine_thumb_2x3
    )

    # --------------------------------------------------------
    # Convert residual transformation from thumbnail
    # coordinates back into full-resolution fixed coordinates.
    #
    # F_full = D^-1 * F_thumbnail * D
    # --------------------------------------------------------

    fine_full = (
        thumbnail_to_full_matrix
        @
        fine_thumb
        @
        full_to_thumbnail_matrix
    )

    # --------------------------------------------------------
    # Analyze proposed affine transform
    # --------------------------------------------------------

    affine_info = analyze_affine(
        fine_full
    )

    # --------------------------------------------------------
    # Test candidate on thumbnail
    # --------------------------------------------------------

    candidate_reg = cv2.warpAffine(
        moving_reg,
        fine_thumb[:2, :],
        (thumb_w, thumb_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    candidate_mask = cv2.warpAffine(
        moving_mask,
        fine_thumb[:2, :],
        (thumb_w, thumb_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    final_overlap = cv2.bitwise_and(
        fixed_mask,
        candidate_mask
    )

    final_overlap = cv2.erode(
        final_overlap,
        kernel
    )

    final_overlap_pixels = np.count_nonzero(
        final_overlap
    )

    final_score = masked_ncc(
        fixed_reg,
        candidate_reg,
        final_overlap
    )

    improvement = (
        final_score
        -
        initial_score
    )

    # --------------------------------------------------------
    # Overlap retention
    # --------------------------------------------------------

    if initial_overlap_pixels > 0:

        overlap_retention = (
            final_overlap_pixels
            /
            initial_overlap_pixels
        )

    else:

        overlap_retention = 0.0

    # --------------------------------------------------------
    # Full-resolution translation threshold
    # --------------------------------------------------------

    image_diagonal = np.hypot(
        fixed_w,
        fixed_h
    )

    max_translation = (
        max_translation_fraction
        *
        image_diagonal
    )

    # --------------------------------------------------------
    # Acceptance test
    # --------------------------------------------------------

    accepted = True
    rejection_reasons = []

    # Reflection / collapse is never acceptable
    if affine_info["determinant"] <= 0:

        accepted = False

        rejection_reasons.append(
            "negative/zero affine determinant"
        )

    # Rotation
    if (
        abs(
            affine_info["rotation_deg"]
        )
        >
        max_rotation_deg
    ):

        accepted = False

        rejection_reasons.append(
            "rotation correction too large"
        )

    # X scale
    if (
        abs(
            affine_info["scale_x"] - 1.0
        )
        >
        max_scale_change
    ):

        accepted = False

        rejection_reasons.append(
            "X scaling correction too large"
        )

    # Y scale
    if (
        abs(
            affine_info["scale_y"] - 1.0
        )
        >
        max_scale_change
    ):

        accepted = False

        rejection_reasons.append(
            "Y scaling correction too large"
        )

    # Principal stretch/compression
    for value in [
        affine_info["principal_scale_1"],
        affine_info["principal_scale_2"]
    ]:

        if (
            abs(value - 1.0)
            >
            max_scale_change
        ):

            accepted = False

            rejection_reasons.append(
                "overall affine scaling too large"
            )

            break

    # Shear
    if (
        affine_info["shear_deg"]
        >
        max_shear_deg
    ):

        accepted = False

        rejection_reasons.append(
            "shear correction too large"
        )

    # Translation
    if (
        affine_info["translation_px"]
        >
        max_translation
    ):

        accepted = False

        rejection_reasons.append(
            "translation correction too large"
        )

    # Similarity improvement
    if (
        improvement
        <
        min_improvement
    ):

        accepted = False

        rejection_reasons.append(
            "registration score did not improve enough"
        )

    # Tissue overlap
    if (
        overlap_retention
        <
        min_overlap_retention
    ):

        accepted = False

        rejection_reasons.append(
            "too much tissue overlap was lost"
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = {
        "accepted":
            bool(accepted),

        "initial_score":
            float(initial_score),

        "final_score":
            float(final_score),

        "improvement":
            float(improvement),

        "ecc_score":
            float(ecc_score),

        "overlap_retention":
            float(overlap_retention),

        **affine_info
    }

    if accepted:

        diagnostics["reason"] = (
            "Affine fine registration accepted"
        )

    else:

        diagnostics["reason"] = (
            "; ".join(
                rejection_reasons
            )
        )

        # Reject transform entirely
        fine_full = np.eye(
            3,
            dtype=np.float64
        )

    # Clean thumbnail arrays
    del fixed_reg
    del moving_reg
    del fixed_mask
    del moving_mask
    del initial_overlap
    del candidate_reg
    del candidate_mask
    del final_overlap

    gc.collect()

    return (
        fine_full,
        diagnostics
    )


# ============================================================
# MAIN REGISTRATION
# ============================================================

def register_he_affine_low_memory(
    fixed_path,
    moving_path
):

    # ========================================================
    # 1. READ FIXED IMAGE ONLY TO OBTAIN SIZE + THUMBNAIL
    # ========================================================

    print(
        "\nLoading fixed image..."
    )

    fixed_bgr = cv2.imread(
        fixed_path,
        cv2.IMREAD_COLOR
    )

    if fixed_bgr is None:

        raise FileNotFoundError(
            f"Could not read fixed image:\n"
            f"{fixed_path}"
        )

    fixed_shape = fixed_bgr.shape

    fixed_h, fixed_w = fixed_shape[:2]

    print(
        "Fixed:",
        fixed_w,
        "x",
        fixed_h
    )

    fixed_thumb = make_thumbnail(
        fixed_bgr,
        MAX_THUMBNAIL_DIM
    )

    thumb_h, thumb_w = (
        fixed_thumb.shape[:2]
    )

    print(
        "Thumbnail:",
        thumb_w,
        "x",
        thumb_h
    )

    # --------------------------------------------------------
    # Coordinate transformation:
    #
    # fixed full resolution -> fixed thumbnail
    # --------------------------------------------------------

    thumb_scale_x = (
        thumb_w / fixed_w
    )

    thumb_scale_y = (
        thumb_h / fixed_h
    )

    full_to_thumb = np.array(
        [
            [
                thumb_scale_x,
                0.0,
                0.0
            ],
            [
                0.0,
                thumb_scale_y,
                0.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ],
        dtype=np.float64
    )

    thumb_to_full = np.linalg.inv(
        full_to_thumb
    )

    # We no longer need the full fixed image during registration
    del fixed_bgr

    gc.collect()

    # ========================================================
    # 2. LOAD MOVING FULL IMAGE
    # ========================================================

    print(
        "\nLoading moving image..."
    )

    moving_bgr = cv2.imread(
        moving_path,
        cv2.IMREAD_COLOR
    )

    if moving_bgr is None:

        raise FileNotFoundError(
            f"Could not read moving image:\n"
            f"{moving_path}"
        )

    moving_shape = moving_bgr.shape

    moving_h, moving_w = (
        moving_shape[:2]
    )

    print(
        "Moving:",
        moving_w,
        "x",
        moving_h
    )

    # ========================================================
    # 3. SCALE TRANSFORM
    # ========================================================

    scale_matrix = build_scale_transform(
        fixed_shape,
        moving_shape
    )

    print(
        "\nScale transform:"
    )

    print(
        scale_matrix
    )

    # ========================================================
    # 4. MANUAL TRANSFORM
    # ========================================================

    manual_matrix = build_manual_transform(
        fixed_shape=fixed_shape,
        moving_shape=moving_shape,
        angle=manual_angle,
        tx=manual_tx,
        ty=manual_ty
    )

    print(
        "\nManual transform:"
    )

    print(
        manual_matrix
    )

    # ========================================================
    # 5. MANUAL THUMBNAIL DIRECTLY FROM ORIGINAL MOVING IMAGE
    # ========================================================

    manual_thumb_matrix = (
        full_to_thumb
        @
        manual_matrix
    )

    moving_manual_thumb = cv2.warpAffine(
        moving_bgr,
        manual_thumb_matrix[:2, :],
        (thumb_w, thumb_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    if SHOW_THUMBNAIL_PREVIEWS:

        show_thumbnail_overlay(
            fixed_thumb,
            moving_manual_thumb,
            "Manual registration"
        )

    # ========================================================
    # 6. AFFINE FINE REGISTRATION
    # ========================================================

    print(
        "\nEstimating affine fine registration..."
    )

    fine_residual_matrix, diagnostics = (
        estimate_fine_affine(
            fixed_thumb=fixed_thumb,

            moving_manual_thumb=(
                moving_manual_thumb
            ),

            full_fixed_shape=(
                fixed_shape
            ),

            thumbnail_to_full_matrix=(
                thumb_to_full
            ),

            full_to_thumbnail_matrix=(
                full_to_thumb
            ),

            max_rotation_deg=(
                MAX_FINE_ROTATION_DEG
            ),

            max_scale_change=(
                MAX_FINE_SCALE_CHANGE
            ),

            max_shear_deg=(
                MAX_FINE_SHEAR_DEG
            ),

            max_translation_fraction=(
                MAX_FINE_TRANSLATION_FRACTION
            ),

            min_improvement=(
                MIN_IMPROVEMENT
            ),

            min_overlap_retention=(
                MIN_OVERLAP_RETENTION
            )
        )
    )

    # ========================================================
    # 7. COMPOSE FINAL TRANSFORMATION
    # ========================================================

    if diagnostics["accepted"]:

        final_matrix = (
            fine_residual_matrix
            @
            manual_matrix
        )

    else:

        # Fine registration rejected.
        # Keep manual result unchanged.
        final_matrix = (
            manual_matrix.copy()
        )

    print(
        "\nFine registration accepted:",
        diagnostics["accepted"]
    )

    print(
        "Reason:",
        diagnostics["reason"]
    )

    print(
        "\nFine residual transform:"
    )

    print(
        fine_residual_matrix
    )

    print(
        "\nFinal transform:"
    )

    print(
        final_matrix
    )

    # ========================================================
    # SAVE TRANSFORMATION MATRICES
    # ========================================================

    np.save(
        scale_matrix_path,
        scale_matrix
    )

    np.save(
        manual_matrix_path,
        manual_matrix
    )

    np.save(
        fine_residual_matrix_path,
        fine_residual_matrix
    )

    np.save(
        final_matrix_path,
        final_matrix
    )

    print(
        "\nTransformation matrices saved:"
    )

    print(
        "Scale:",
        scale_matrix_path
    )

    print(
        "Manual:",
        manual_matrix_path
    )

    print(
        "Fine residual:",
        fine_residual_matrix_path
    )

    print(
        "Final:",
        final_matrix_path
    )

    # ========================================================
    # 8. FINE THUMBNAIL PREVIEW
    # ========================================================

    final_thumb_matrix = (
        full_to_thumb
        @
        final_matrix
    )

    moving_fine_thumb = cv2.warpAffine(
        moving_bgr,
        final_thumb_matrix[:2, :],
        (thumb_w, thumb_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    if SHOW_THUMBNAIL_PREVIEWS:

        if diagnostics["accepted"]:

            title = (
                "Affine fine registration - accepted"
            )

        else:

            title = (
                "Affine fine registration rejected - "
                "manual retained"
            )

        show_thumbnail_overlay(
            fixed_thumb,
            moving_fine_thumb,
            title
        )

    # Thumbnail data no longer needed
    del fixed_thumb
    del moving_manual_thumb
    del moving_fine_thumb

    gc.collect()

    # ========================================================
    # 9. SAVE SCALED IMAGE
    #
    # Full-resolution arrays are generated ONE AT A TIME.
    # ========================================================

    print(
        "\nCreating scaled image..."
    )

    scaled_full = cv2.warpAffine(
        moving_bgr,
        scale_matrix[:2, :],
        (fixed_w, fixed_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    save_bgr(
        scaled_path,
        scaled_full
    )

    del scaled_full

    gc.collect()

    # ========================================================
    # 10. SAVE MANUAL IMAGE
    # ========================================================

    print(
        "\nCreating manual registration..."
    )

    manual_full = cv2.warpAffine(
        moving_bgr,
        manual_matrix[:2, :],
        (fixed_w, fixed_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    save_bgr(
        manual_path,
        manual_full
    )

    del manual_full

    gc.collect()

    # ========================================================
    # 11. SAVE FINE IMAGE
    # ========================================================

    print(
        "\nCreating final fine registration..."
    )

    fine_full = cv2.warpAffine(
        moving_bgr,
        final_matrix[:2, :],
        (fixed_w, fixed_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    save_bgr(
        fine_path,
        fine_full
    )

    del fine_full

    # Original moving image no longer required
    del moving_bgr

    gc.collect()

    # ========================================================
    # 12. CREATE OVERLAYS
    #
    # Reload images from disk to keep peak RAM low.
    # ========================================================

    print(
        "\nCreating overlays..."
    )

    fixed_bgr = cv2.imread(
        fixed_path,
        cv2.IMREAD_COLOR
    )

    if fixed_bgr is None:

        raise FileNotFoundError(
            f"Could not reload fixed image:\n"
            f"{fixed_path}"
        )

    # --------------------------------------------------------
    # Manual overlay
    # --------------------------------------------------------

    manual_full = cv2.imread(
        manual_path,
        cv2.IMREAD_COLOR
    )

    if manual_full is None:

        raise FileNotFoundError(
            f"Could not reload:\n"
            f"{manual_path}"
        )

    # IMPORTANT:
    #
    # dst=manual_full means OpenCV overwrites manual_full
    # instead of allocating a second full-size overlay array.
    cv2.addWeighted(
        fixed_bgr,
        0.5,
        manual_full,
        0.5,
        0,
        dst=manual_full
    )

    save_bgr(
        manual_overlay_path,
        manual_full
    )

    del manual_full

    gc.collect()

    # --------------------------------------------------------
    # Fine overlay
    # --------------------------------------------------------

    fine_full = cv2.imread(
        fine_path,
        cv2.IMREAD_COLOR
    )

    if fine_full is None:

        raise FileNotFoundError(
            f"Could not reload:\n"
            f"{fine_path}"
        )

    cv2.addWeighted(
        fixed_bgr,
        0.5,
        fine_full,
        0.5,
        0,
        dst=fine_full
    )

    save_bgr(
        fine_overlay_path,
        fine_full
    )

    del fine_full
    del fixed_bgr

    gc.collect()

    # ========================================================
    # 13. SAVE DIAGNOSTICS
    # ========================================================

    # Convert matrices to regular lists so JSON can store them.
    diagnostics["scale_matrix"] = (
        scale_matrix.tolist()
    )

    diagnostics["manual_matrix"] = (
        manual_matrix.tolist()
    )

    diagnostics["fine_residual_matrix"] = (
        fine_residual_matrix.tolist()
    )

    diagnostics["final_matrix"] = (
        final_matrix.tolist()
    )

    with open(
        diagnostics_path,
        "w"
    ) as file:

        json.dump(
            diagnostics,
            file,
            indent=4
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "REGISTRATION FINISHED"
    )

    print(
        "========================================"
    )

    print(
        "\nFine registration accepted:",
        diagnostics["accepted"]
    )

    print(
        "\nReason:"
    )

    print(
        diagnostics["reason"]
    )

    print(
        "\nScores:"
    )

    print(
        "Manual NCC:",
        diagnostics["initial_score"]
    )

    print(
        "Fine NCC:",
        diagnostics.get(
            "final_score"
        )
    )

    print(
        "Improvement:",
        diagnostics.get(
            "improvement"
        )
    )

    print(
        "\nFine transformation:"
    )

    print(
        "Rotation:",
        diagnostics.get(
            "rotation_deg"
        ),
        "degrees"
    )

    print(
        "Scale X:",
        diagnostics.get(
            "scale_x"
        )
    )

    print(
        "Scale Y:",
        diagnostics.get(
            "scale_y"
        )
    )

    print(
        "Shear:",
        diagnostics.get(
            "shear_deg"
        ),
        "degrees"
    )

    print(
        "Translation X:",
        diagnostics.get(
            "tx"
        ),
        "pixels"
    )

    print(
        "Translation Y:",
        diagnostics.get(
            "ty"
        ),
        "pixels"
    )

    print(
        "\nSaved:"
    )

    print(
        scaled_path
    )

    print(
        manual_path
    )

    print(
        manual_overlay_path
    )

    print(
        fine_path
    )

    print(
        fine_overlay_path
    )

    print(
        diagnostics_path
    )

    return diagnostics


# ============================================================
# RUN
# ============================================================

diagnostics = register_he_affine_low_memory(
    fixed_path=fixed_path,
    moving_path=moving_path
)