
import io
import zipfile
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageColor, ImageOps
from scipy import ndimage


st.set_page_config(
    page_title="Background & Halo Remover",
    page_icon="🪄",
    layout="wide",
)

st.title("Background & Halo Remover")
st.caption(
    "Conservative background removal for watercolor and clipart motifs. "
    "The app is designed to preserve fine details and soft transparent edges."
)


# -----------------------------
# Utility functions
# -----------------------------
def pil_to_rgba(image: Image.Image) -> Image.Image:
    return image.convert("RGBA")


def estimate_background_color(rgb: np.ndarray, border_width: int = 8) -> np.ndarray:
    """Estimate the background color from a robust median of border pixels."""
    h, w, _ = rgb.shape
    bw = max(1, min(border_width, h // 4, w // 4))

    border = np.concatenate(
        [
            rgb[:bw, :, :].reshape(-1, 3),
            rgb[-bw:, :, :].reshape(-1, 3),
            rgb[:, :bw, :].reshape(-1, 3),
            rgb[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    return np.median(border.astype(np.float32), axis=0)


def color_distance(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Euclidean RGB color distance."""
    diff = rgb.astype(np.float32) - bg.reshape(1, 1, 3)
    return np.sqrt(np.sum(diff * diff, axis=2))


def connected_to_border(mask: np.ndarray) -> np.ndarray:
    """
    Keep only mask regions connected to the image border.
    This protects enclosed light areas inside the motif.
    """
    h, w = mask.shape
    seeds = np.zeros_like(mask, dtype=bool)
    seeds[0, :] = mask[0, :]
    seeds[-1, :] = mask[-1, :]
    seeds[:, 0] = mask[:, 0]
    seeds[:, -1] = mask[:, -1]

    return ndimage.binary_propagation(seeds, mask=mask)


def remove_small_holes(mask: np.ndarray, max_hole_size: int = 64) -> np.ndarray:
    """Fill only very small holes to protect delicate enclosed details."""
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    labels, count = ndimage.label(holes)

    result = mask.copy()
    for label_id in range(1, count + 1):
        region = labels == label_id
        if int(region.sum()) <= max_hole_size:
            result[region] = True
    return result


def decontaminate_colors(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Approximate color decontamination:
    Recover foreground colors from a background-composited image.
    Only affects partially transparent pixels.
    """
    a = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)
    safe_a = np.maximum(a, 0.08)[..., None]
    bgf = bg.reshape(1, 1, 3).astype(np.float32)

    recovered = (rgb.astype(np.float32) - (1.0 - a[..., None]) * bgf) / safe_a
    recovered = np.clip(recovered, 0, 255)

    edge_weight = np.clip((1.0 - a) * strength, 0.0, 1.0)[..., None]
    mixed = rgb.astype(np.float32) * (1.0 - edge_weight) + recovered * edge_weight
    return np.clip(mixed, 0, 255).astype(np.uint8)


def process_background_removal(
    image: Image.Image,
    tolerance: int,
    softness: int,
    edge_preserve: int,
    manual_bg_hex: str | None,
    decontamination: float,
    edge_trim_px: int,
) -> Image.Image:
    rgba = np.array(pil_to_rgba(image), dtype=np.uint8)
    rgb = rgba[..., :3]
    original_alpha = rgba[..., 3].astype(np.float32)

    if manual_bg_hex:
        bg = np.array(ImageColor.getrgb(manual_bg_hex), dtype=np.float32)
    else:
        bg = estimate_background_color(rgb)

    dist = color_distance(rgb, bg)

    # Conservative definite-background mask.
    definite_bg = dist <= tolerance
    definite_bg = connected_to_border(definite_bg)

    # Protect thin structures by avoiding aggressive closing.
    if edge_preserve > 0:
        definite_bg = ndimage.binary_opening(
            definite_bg,
            structure=np.ones((2, 2), dtype=bool),
            iterations=1,
        )

    # Soft matte based on distance from background.
    lower = float(tolerance)
    upper = float(tolerance + max(1, softness))
    matte = np.clip((dist - lower) / max(1.0, upper - lower), 0.0, 1.0)

    # Only fully remove pixels connected to the outside.
    matte[definite_bg] = 0.0

    # Regions not connected to the outside remain protected.
    potential_bg = dist <= upper
    outside_soft = connected_to_border(potential_bg)
    matte[~outside_soft] = 1.0

    # Preserve original transparency if source already has alpha.
    matte *= original_alpha / 255.0

    if edge_trim_px > 0:
        foreground = matte > 0.03
        eroded = ndimage.binary_erosion(
            foreground,
            structure=np.ones((3, 3), dtype=bool),
            iterations=edge_trim_px,
            border_value=0,
        )
        # Soft transition around contracted edge.
        feather = ndimage.distance_transform_edt(eroded)
        contraction_alpha = np.clip(feather, 0.0, 1.0)
        matte *= contraction_alpha

    alpha = np.clip(matte * 255.0, 0, 255).astype(np.uint8)
    clean_rgb = decontaminate_colors(rgb, alpha, bg, decontamination)

    out = np.dstack([clean_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def process_existing_png(
    image: Image.Image,
    mode: str,
    strength: str,
    edge_trim_px: int,
) -> Image.Image:
    rgba = np.array(pil_to_rgba(image), dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32)

    strength_map = {
        "Off": 0.0,
        "Gentle": 0.25,
        "Medium": 0.50,
        "Strong": 0.75,
    }
    s = strength_map[strength]

    if s == 0.0:
        return Image.fromarray(rgba, mode="RGBA")

    edge = (alpha > 0) & (alpha < 250)
    edge_band = ndimage.binary_dilation(edge, iterations=1)

    if mode == "Light halo":
        target = np.full((1, 1, 3), 255.0, dtype=np.float32)
    else:
        # Conservative dark neutral target.
        target = np.full((1, 1, 3), 32.0, dtype=np.float32)

    a = np.clip(alpha / 255.0, 0.0, 1.0)
    safe_a = np.maximum(a, 0.08)[..., None]
    recovered = (rgb - (1.0 - a[..., None]) * target) / safe_a
    recovered = np.clip(recovered, 0, 255)

    weight = (edge_band.astype(np.float32) * (1.0 - a) * s)[..., None]
    rgb = rgb * (1.0 - weight) + recovered * weight

    # Gentle cleanup of extremely weak exterior pixels.
    weak = (alpha > 0) & (alpha < (10 + 25 * s))
    weak_outside = connected_to_border(weak | (alpha == 0)) & weak
    alpha[weak_outside] *= max(0.0, 1.0 - 0.65 * s)

    if edge_trim_px > 0:
        fg = alpha > 2
        eroded = ndimage.binary_erosion(
            fg,
            structure=np.ones((3, 3), dtype=bool),
            iterations=edge_trim_px,
            border_value=0,
        )
        alpha[~eroded] *= 0.35

    out = np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ]
    )
    return Image.fromarray(out, mode="RGBA")


def composite_on_background(image: Image.Image, color: str) -> Image.Image:
    rgba = pil_to_rgba(image)
    bg = Image.new("RGBA", rgba.size, ImageColor.getrgb(color) + (255,))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Processing mode",
    ["Remove full background", "Clean existing transparent PNG"],
)

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

preview_bg = st.sidebar.selectbox(
    "Preview background",
    ["White", "Light gray", "Dark", "Custom"],
)

preview_colors = {
    "White": "#FFFFFF",
    "Light gray": "#D8D8D8",
    "Dark": "#222222",
}

if preview_bg == "Custom":
    preview_color = st.sidebar.color_picker("Custom preview color", "#B9C7D8")
else:
    preview_color = preview_colors[preview_bg]

if mode == "Remove full background":
    st.sidebar.subheader("Background removal")

    background_choice = st.sidebar.radio(
        "Background color",
        ["Auto-detect from image border", "Choose manually"],
    )

    manual_bg = None
    if background_choice == "Choose manually":
        manual_bg = st.sidebar.color_picker("Background color", "#FFFFFF")

    tolerance = st.sidebar.slider(
        "Background tolerance",
        min_value=5,
        max_value=80,
        value=24,
        help="Higher values remove colors further away from the detected background.",
    )

    softness = st.sidebar.slider(
        "Edge softness",
        min_value=2,
        max_value=80,
        value=30,
        help="Controls the transition between transparent background and motif.",
    )

    preserve_level = st.sidebar.select_slider(
        "Fine-detail protection",
        options=["Maximum", "High", "Normal"],
        value="High",
    )
    preserve_map = {"Maximum": 2, "High": 1, "Normal": 0}

    decontamination = st.sidebar.slider(
        "Color decontamination",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Removes the former background color from soft edge pixels.",
    )

    edge_trim_px = st.sidebar.slider(
        "Optional edge trim",
        min_value=0,
        max_value=2,
        value=0,
        help="Keep at 0 for delicate watercolor and fine details.",
    )

else:
    st.sidebar.subheader("Halo cleanup")

    halo_mode = st.sidebar.radio(
        "Halo type",
        ["Light halo", "Dark halo"],
    )

    halo_strength = st.sidebar.select_slider(
        "Cleanup strength",
        options=["Off", "Gentle", "Medium", "Strong"],
        value="Gentle",
    )

    edge_trim_px = st.sidebar.slider(
        "Optional edge trim",
        min_value=0,
        max_value=2,
        value=0,
    )


# -----------------------------
# Main processing
# -----------------------------
if not uploaded_files:
    st.info(
        "Upload a JPEG/PNG with its original background, or an already transparent PNG with a halo."
    )
    st.stop()

processed_items: list[Tuple[str, Image.Image]] = []

for uploaded in uploaded_files:
    try:
        source = Image.open(uploaded)
        source = ImageOps.exif_transpose(source)

        if mode == "Remove full background":
            result = process_background_removal(
                image=source,
                tolerance=tolerance,
                softness=softness,
                edge_preserve=preserve_map[preserve_level],
                manual_bg_hex=manual_bg,
                decontamination=decontamination,
                edge_trim_px=edge_trim_px,
            )
        else:
            result = process_existing_png(
                image=source,
                mode=halo_mode,
                strength=halo_strength,
                edge_trim_px=edge_trim_px,
            )

        output_name = f"{Path(uploaded.name).stem}_clean.png"
        processed_items.append((output_name, result))

        st.subheader(uploaded.name)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Before**")
            st.image(
                composite_on_background(source, preview_color),
                use_container_width=True,
            )

        with col2:
            st.markdown("**After**")
            st.image(
                composite_on_background(result, preview_color),
                use_container_width=True,
            )

        st.download_button(
            label=f"Download {output_name}",
            data=image_to_png_bytes(result),
            file_name=output_name,
            mime="image/png",
            key=f"download_{uploaded.name}",
        )

        with st.expander("Check on four backgrounds"):
            bg_cols = st.columns(4)
            checks = [
                ("White", "#FFFFFF"),
                ("Gray", "#B8B8B8"),
                ("Dark", "#1F1F1F"),
                ("Color", "#9FB7C9"),
            ]
            for column, (label, color) in zip(bg_cols, checks):
                with column:
                    st.caption(label)
                    st.image(
                        composite_on_background(result, color),
                        use_container_width=True,
                    )

        st.divider()

    except Exception as exc:
        st.error(f"Could not process {uploaded.name}: {exc}")

if len(processed_items) > 1:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, result in processed_items:
            archive.writestr(filename, image_to_png_bytes(result))

    st.download_button(
        label="Download all cleaned PNGs as ZIP",
        data=zip_buffer.getvalue(),
        file_name="cleaned_pngs.zip",
        mime="application/zip",
    )

st.warning(
    "For very pale motifs on a nearly identical background, automatic removal can never be "
    "perfectly guaranteed. Start with conservative settings and check the result on a dark background."
)
