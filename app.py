
import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageColor, ImageOps
from scipy import ndimage


st.set_page_config(
    page_title="Halo Remover",
    page_icon="✨",
    layout="wide",
)

st.title("Halo Remover")
st.caption(
    "Specialized cleanup for transparent watercolor and clipart PNGs. "
    "The app works only on a narrow outer edge and protects the interior artwork."
)


# ---------------------------------------------------------
# Image helpers
# ---------------------------------------------------------
def to_rgba(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGBA")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def composite(image: Image.Image, color: str) -> Image.Image:
    rgba = to_rgba(image)
    background = Image.new(
        "RGBA",
        rgba.size,
        ImageColor.getrgb(color) + (255,),
    )
    return Image.alpha_composite(background, rgba).convert("RGB")


def edge_mask_from_alpha(alpha: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
    - visible mask
    - narrow inward edge band measured from transparency
    """
    visible = alpha > 0
    distance_inside = ndimage.distance_transform_edt(visible)
    edge_band = visible & (distance_inside <= max(1, width))
    return visible, edge_band


def nearest_interior_colors(
    rgb: np.ndarray,
    alpha: np.ndarray,
    edge_width: int,
) -> np.ndarray:
    """
    Find a nearby clean interior color for every pixel.
    The seed is placed several pixels inside the visible object.
    """
    visible = alpha > 0
    distance_inside = ndimage.distance_transform_edt(visible)

    seed_distance = max(3, edge_width + 2)
    seed = visible & (distance_inside >= seed_distance) & (alpha >= 160)

    if not np.any(seed):
        seed = visible & (distance_inside >= max(2, edge_width + 1))
    if not np.any(seed):
        seed = visible & (alpha >= 220)
    if not np.any(seed):
        seed = visible

    _, nearest = ndimage.distance_transform_edt(
        ~seed,
        return_indices=True,
    )
    return rgb[nearest[0], nearest[1]]


def decontaminate_partial_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    contamination_rgb: np.ndarray,
    edge_band: np.ndarray,
    amount: float,
) -> np.ndarray:
    """
    Recover foreground color F from compositing equation:
        observed = alpha * F + (1-alpha) * background

    This is applied only to semi-transparent pixels in the outer edge band.
    """
    a = np.clip(alpha / 255.0, 0.0, 1.0)
    safe_a = np.maximum(a, 0.05)[..., None]
    bg = contamination_rgb.reshape(1, 1, 3).astype(np.float32)

    recovered = (rgb - (1.0 - a[..., None]) * bg) / safe_a
    recovered = np.clip(recovered, 0.0, 255.0)

    partial = edge_band & (alpha > 0) & (alpha < 255)
    weight = partial.astype(np.float32) * (1.0 - a) * amount
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    return rgb * (1.0 - weight) + recovered * weight


def repair_solid_fringe(
    rgb: np.ndarray,
    alpha: np.ndarray,
    contamination_rgb: np.ndarray,
    edge_band: np.ndarray,
    inner_rgb: np.ndarray,
    amount: float,
    color_tolerance: float,
) -> np.ndarray:
    """
    Repair opaque or nearly opaque fringe pixels.

    A pixel is considered suspicious only when:
    - it is in the narrow outer edge band,
    - it is relatively close to the former background color,
    - it differs from the nearby interior motif color.

    This prevents the whole pale motif from being altered.
    """
    contamination = contamination_rgb.reshape(1, 1, 3).astype(np.float32)

    dist_to_bg = np.sqrt(np.sum((rgb - contamination) ** 2, axis=2))
    dist_to_inner = np.sqrt(np.sum((rgb - inner_rgb) ** 2, axis=2))

    # Near-background score: 1 when close to contamination color.
    near_bg = np.clip(
        1.0 - dist_to_bg / max(1.0, color_tolerance),
        0.0,
        1.0,
    )

    # Require a real difference from the local interior color.
    differs_from_inner = np.clip(
        (dist_to_inner - 4.0) / 55.0,
        0.0,
        1.0,
    )

    # Fully opaque fringe can exist after a poor background remover.
    opaque_factor = np.clip((alpha - 80.0) / 175.0, 0.0, 1.0)
    semi_factor = np.clip((255.0 - alpha) / 180.0, 0.0, 1.0)
    alpha_factor = np.maximum(0.55 * opaque_factor, semi_factor)

    weight = (
        edge_band.astype(np.float32)
        * near_bg
        * differs_from_inner
        * alpha_factor
        * amount
    )
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    return rgb * (1.0 - weight) + inner_rgb * weight


def clean_weak_outer_pixels(
    alpha: np.ndarray,
    edge_band: np.ndarray,
    amount: float,
) -> np.ndarray:
    """
    Remove only extremely weak outer remnants.
    This is intentionally conservative.
    """
    cleaned = alpha.copy()
    threshold = int(round(2 + amount * 16))
    weak = edge_band & (cleaned > 0) & (cleaned <= threshold)
    cleaned[weak] = 0
    return cleaned


def process_halo(
    image: Image.Image,
    halo_type: str,
    strength: str,
    halo_width: int,
    contamination_hex: str,
    repair_opaque: bool,
    weak_pixel_cleanup: bool,
) -> Image.Image:
    rgba = np.array(to_rgba(image), dtype=np.uint8)

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32)

    strength_values = {
        "Off": 0.0,
        "Gentle": 0.35,
        "Medium": 0.65,
        "Strong": 1.0,
    }
    amount = strength_values[strength]

    if amount == 0.0 or not np.any(alpha > 0):
        return Image.fromarray(rgba, mode="RGBA")

    contamination_rgb = np.array(
        ImageColor.getrgb(contamination_hex),
        dtype=np.float32,
    )

    visible, edge_band = edge_mask_from_alpha(alpha, halo_width)
    inner_rgb = nearest_interior_colors(rgb, alpha, halo_width)

    # For dark halos, users may still choose a custom former background color.
    if halo_type == "Dark halo" and contamination_hex.upper() == "#FFFFFF":
        contamination_rgb = np.array([32.0, 32.0, 32.0], dtype=np.float32)

    # 1. Correct semi-transparent edge pixels using alpha decontamination.
    rgb = decontaminate_partial_alpha(
        rgb=rgb,
        alpha=alpha,
        contamination_rgb=contamination_rgb,
        edge_band=edge_band,
        amount=amount,
    )

    # 2. Optionally repair solid fringe pixels by borrowing local interior color.
    if repair_opaque:
        tolerance = 125.0 if halo_type == "Light halo" else 105.0
        rgb = repair_solid_fringe(
            rgb=rgb,
            alpha=alpha,
            contamination_rgb=contamination_rgb,
            edge_band=edge_band,
            inner_rgb=inner_rgb,
            amount=amount,
            color_tolerance=tolerance,
        )

    # 3. Only delete nearly invisible outer leftovers.
    if weak_pixel_cleanup:
        alpha = clean_weak_outer_pixels(
            alpha=alpha,
            edge_band=edge_band,
            amount=amount,
        )

    # Fully transparent pixels should not carry white/gray RGB residue.
    rgb[alpha <= 0] = 0

    result = np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ]
    )
    return Image.fromarray(result, mode="RGBA")


def alpha_preview(image: Image.Image) -> Image.Image:
    alpha = np.array(to_rgba(image))[..., 3]
    return Image.fromarray(alpha.astype(np.uint8), mode="L")


def edge_preview(image: Image.Image, width: int) -> Image.Image:
    rgba = np.array(to_rgba(image))
    alpha = rgba[..., 3].astype(np.float32)
    _, edge = edge_mask_from_alpha(alpha, width)
    preview = np.zeros((*edge.shape, 3), dtype=np.uint8)
    preview[edge] = 255
    return Image.fromarray(preview, mode="RGB")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Halo cleanup")

halo_type = st.sidebar.radio(
    "Halo type",
    ["Light halo", "Dark halo"],
)

strength = st.sidebar.select_slider(
    "Cleanup strength",
    options=["Off", "Gentle", "Medium", "Strong"],
    value="Medium",
)

halo_width = st.sidebar.slider(
    "Halo width",
    min_value=1,
    max_value=6,
    value=2,
    help="Only this narrow outer edge band is changed.",
)

default_contamination = "#FFFFFF" if halo_type == "Light halo" else "#202020"
contamination_color = st.sidebar.color_picker(
    "Former background color",
    default_contamination,
    help="Usually white for a light halo.",
)

repair_opaque = st.sidebar.checkbox(
    "Repair solid white/gray fringe",
    value=True,
)

weak_pixel_cleanup = st.sidebar.checkbox(
    "Remove extremely weak outer pixels",
    value=True,
)

preview_choice = st.sidebar.selectbox(
    "Preview background",
    ["White", "Light gray", "Dark", "Custom"],
)

preview_map = {
    "White": "#FFFFFF",
    "Light gray": "#C8C8C8",
    "Dark": "#1E1E1E",
}

if preview_choice == "Custom":
    preview_color = st.sidebar.color_picker(
        "Custom preview color",
        "#9FB3C8",
    )
else:
    preview_color = preview_map[preview_choice]

debug_mode = st.sidebar.checkbox(
    "Developer / Debug mode",
    value=False,
)


# ---------------------------------------------------------
# Upload and processing
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more transparent PNG files",
    type=["png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload a transparent PNG to begin.")
    st.stop()

results = []

for index, uploaded in enumerate(uploaded_files):
    try:
        source = Image.open(uploaded)
        source = ImageOps.exif_transpose(source)

        has_alpha = source.mode in ("RGBA", "LA") or (
            source.mode == "P" and "transparency" in source.info
        )

        if not has_alpha:
            st.error(
                f"{uploaded.name}: This PNG has no transparency. "
                "Please upload the transparent source file."
            )
            continue

        result = process_halo(
            image=source,
            halo_type=halo_type,
            strength=strength,
            halo_width=halo_width,
            contamination_hex=contamination_color,
            repair_opaque=repair_opaque,
            weak_pixel_cleanup=weak_pixel_cleanup,
        )

        output_name = f"{Path(uploaded.name).stem}_halo_clean.png"
        results.append((output_name, result))

        st.subheader(uploaded.name)

        before_col, after_col = st.columns(2)

        with before_col:
            st.markdown("**Before**")
            st.image(
                composite(source, preview_color),
                use_container_width=True,
            )

        with after_col:
            st.markdown("**After**")
            st.image(
                composite(result, preview_color),
                use_container_width=True,
            )

        st.download_button(
            "Download cleaned PNG",
            data=png_bytes(result),
            file_name=output_name,
            mime="image/png",
            key=f"download_{index}",
        )

        with st.expander("Check on four backgrounds"):
            columns = st.columns(4)
            backgrounds = [
                ("White", "#FFFFFF"),
                ("Gray", "#BEBEBE"),
                ("Dark", "#202020"),
                ("Color", "#91A9BD"),
            ]

            for column, (label, color) in zip(columns, backgrounds):
                with column:
                    st.caption(label)
                    st.image(
                        composite(result, color),
                        use_container_width=True,
                    )

        if debug_mode:
            rgba = np.array(to_rgba(source))
            alpha = rgba[..., 3]

            transparent_count = int(np.sum(alpha == 0))
            partial_count = int(np.sum((alpha > 0) & (alpha < 255)))
            opaque_count = int(np.sum(alpha == 255))

            st.markdown("### Debug information")
            st.write(
                {
                    "File format": source.format or "PNG",
                    "Image mode": source.mode,
                    "Dimensions": f"{source.width} × {source.height}",
                    "Transparent pixels": transparent_count,
                    "Partially transparent pixels": partial_count,
                    "Opaque pixels": opaque_count,
                }
            )

            debug_cols = st.columns(2)
            with debug_cols[0]:
                st.markdown("**Alpha channel**")
                st.image(alpha_preview(source), use_container_width=True)

            with debug_cols[1]:
                st.markdown("**Detected edge band**")
                st.image(
                    edge_preview(source, halo_width),
                    use_container_width=True,
                )

        st.divider()

    except Exception as exc:
        st.error(f"Could not process {uploaded.name}: {exc}")

if len(results) > 1:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename, image in results:
            archive.writestr(filename, png_bytes(image))

    st.download_button(
        "Download all cleaned PNGs as ZIP",
        data=zip_buffer.getvalue(),
        file_name="halo_cleaned_pngs.zip",
        mime="application/zip",
    )

st.caption(
    "Recommended starting point for white or beige halos: "
    "Light halo · Medium · Width 2 · Repair solid fringe enabled."
)
