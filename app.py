
import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageColor, ImageOps
from scipy import ndimage


st.set_page_config(
    page_title="Halo Studio",
    page_icon="✨",
    layout="wide",
)

st.title("Halo Studio")
st.caption(
    "Color decontamination for transparent watercolor and clipart PNGs. "
    "The alpha channel is preserved unless weak-pixel cleanup is enabled."
)


def open_rgba(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGBA")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def composite_on_color(image: Image.Image, color: str) -> Image.Image:
    rgba = open_rgba(image)
    background = Image.new(
        "RGBA",
        rgba.size,
        ImageColor.getrgb(color) + (255,),
    )
    return Image.alpha_composite(background, rgba).convert("RGB")


def make_edge_band(alpha: np.ndarray, width: int) -> np.ndarray:
    visible = alpha > 0
    if not np.any(visible):
        return visible
    distance_inside = ndimage.distance_transform_edt(visible)
    return visible & (distance_inside <= max(1, int(width)))


def nearest_interior_rgb(
    rgb: np.ndarray,
    alpha: np.ndarray,
    edge_width: int,
) -> np.ndarray:
    visible = alpha > 0
    distance_inside = ndimage.distance_transform_edt(visible)

    seed = (
        visible
        & (distance_inside >= max(3, edge_width + 2))
        & (alpha >= 180)
    )

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


def exact_decontamination(
    rgb: np.ndarray,
    alpha: np.ndarray,
    former_bg: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Recover foreground color F from:
        observed = alpha * F + (1-alpha) * background

    Applied only to semi-transparent pixels in the selected edge band.
    """
    a = np.clip(alpha / 255.0, 0.0, 1.0)
    safe_a = np.maximum(a, 0.035)[..., None]
    bg = former_bg.reshape(1, 1, 3).astype(np.float32)

    recovered = (rgb - (1.0 - a[..., None]) * bg) / safe_a
    recovered = np.clip(recovered, 0.0, 255.0)

    partial = mask & (alpha > 0) & (alpha < 255)
    weight = (
        partial.astype(np.float32)
        * np.clip((255.0 - alpha) / 210.0, 0.0, 1.0)
        * strength
    )[..., None]

    return rgb * (1.0 - weight) + recovered * weight


def repair_opaque_fringe(
    rgb: np.ndarray,
    alpha: np.ndarray,
    former_bg: np.ndarray,
    edge_band: np.ndarray,
    inner_rgb: np.ndarray,
    strength: float,
    tolerance: float,
) -> np.ndarray:
    """
    Repair opaque fringe pixels only when they are:
    - in the narrow outer edge band,
    - close to the former background color,
    - visibly different from nearby interior motif color.
    """
    bg = former_bg.reshape(1, 1, 3).astype(np.float32)

    distance_to_bg = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    distance_to_inner = np.sqrt(np.sum((rgb - inner_rgb) ** 2, axis=2))

    near_background = np.clip(
        1.0 - distance_to_bg / max(1.0, tolerance),
        0.0,
        1.0,
    )

    differs_from_inner = np.clip(
        (distance_to_inner - 3.0) / 52.0,
        0.0,
        1.0,
    )

    opaque_or_nearly = np.clip((alpha - 170.0) / 85.0, 0.0, 1.0)

    weight = (
        edge_band.astype(np.float32)
        * near_background
        * differs_from_inner
        * opaque_or_nearly
        * strength
    )
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    return rgb * (1.0 - weight) + inner_rgb * weight


def remove_weak_outer_pixels(
    alpha: np.ndarray,
    edge_band: np.ndarray,
    strength: float,
) -> np.ndarray:
    cleaned = alpha.copy()
    threshold = int(round(3 + 18 * strength))
    weak = edge_band & (cleaned > 0) & (cleaned <= threshold)
    cleaned[weak] = 0
    return cleaned


def process_image(
    image: Image.Image,
    former_background_hex: str,
    cleanup_strength: str,
    edge_width: int,
    repair_opaque: bool,
    remove_weak: bool,
) -> Image.Image:
    rgba = np.array(open_rgba(image), dtype=np.uint8)

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32)

    strength_map = {
        "Off": 0.0,
        "Gentle": 0.45,
        "Medium": 0.75,
        "Strong": 1.0,
    }
    strength = strength_map[cleanup_strength]

    if strength == 0.0 or not np.any(alpha > 0):
        return Image.fromarray(rgba, mode="RGBA")

    former_bg = np.array(
        ImageColor.getrgb(former_background_hex),
        dtype=np.float32,
    )

    edge_band = make_edge_band(alpha, edge_width)

    # 1. Exact decontamination of semi-transparent edge pixels.
    rgb = exact_decontamination(
        rgb=rgb,
        alpha=alpha,
        former_bg=former_bg,
        mask=edge_band,
        strength=strength,
    )

    # 2. Optional repair of solid fringe pixels.
    if repair_opaque:
        interior_rgb = nearest_interior_rgb(rgb, alpha, edge_width)
        rgb = repair_opaque_fringe(
            rgb=rgb,
            alpha=alpha,
            former_bg=former_bg,
            edge_band=edge_band,
            inner_rgb=interior_rgb,
            strength=strength,
            tolerance=150.0,
        )

    # 3. Optional deletion of only nearly invisible outer remnants.
    if remove_weak:
        alpha = remove_weak_outer_pixels(
            alpha=alpha,
            edge_band=edge_band,
            strength=strength,
        )

    # Hidden RGB values in fully transparent pixels can create export fringes.
    rgb[alpha <= 0] = 0

    output = np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ]
    )
    return Image.fromarray(output, mode="RGBA")


def alpha_preview(image: Image.Image) -> Image.Image:
    alpha = np.array(open_rgba(image), dtype=np.uint8)[..., 3]
    return Image.fromarray(alpha, mode="L")


def edge_preview(image: Image.Image, width: int) -> Image.Image:
    alpha = np.array(open_rgba(image), dtype=np.uint8)[..., 3]
    edge = make_edge_band(alpha.astype(np.float32), width)
    preview = np.zeros((*edge.shape, 3), dtype=np.uint8)
    preview[edge] = 255
    return Image.fromarray(preview, mode="RGB")


st.sidebar.header("Cleanup settings")

cleanup_strength = st.sidebar.select_slider(
    "Cleanup strength",
    options=["Off", "Gentle", "Medium", "Strong"],
    value="Medium",
)

edge_width = st.sidebar.slider(
    "Halo width",
    min_value=1,
    max_value=6,
    value=3,
    help="Only this narrow band along the transparent edge is processed.",
)

former_background = st.sidebar.color_picker(
    "Former background color",
    "#FFFFFF",
    help="Choose the background color that was removed before the halo appeared.",
)

repair_opaque = st.sidebar.checkbox(
    "Repair solid white/gray fringe",
    value=True,
)

remove_weak = st.sidebar.checkbox(
    "Remove extremely weak outer pixels",
    value=True,
)

preview_choice = st.sidebar.selectbox(
    "Preview background",
    ["White", "Light gray", "Dark", "Custom"],
)

preview_colors = {
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
    preview_color = preview_colors[preview_choice]

debug_mode = st.sidebar.checkbox(
    "Developer / Debug mode",
    value=False,
)

uploaded_files = st.file_uploader(
    "Upload one or more transparent PNG files",
    type=["png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload a transparent PNG to begin.")
    st.stop()

processed_files = []

for index, uploaded in enumerate(uploaded_files):
    try:
        source = Image.open(uploaded)
        source = ImageOps.exif_transpose(source)

        has_alpha = source.mode in ("RGBA", "LA") or (
            source.mode == "P" and "transparency" in source.info
        )

        if not has_alpha:
            st.error(
                f"{uploaded.name}: This PNG contains no transparency."
            )
            continue

        cleaned = process_image(
            image=source,
            former_background_hex=former_background,
            cleanup_strength=cleanup_strength,
            edge_width=edge_width,
            repair_opaque=repair_opaque,
            remove_weak=remove_weak,
        )

        output_name = f"{Path(uploaded.name).stem}_halo_clean.png"
        processed_files.append((output_name, cleaned))

        st.subheader(uploaded.name)

        before_col, after_col = st.columns(2)

        with before_col:
            st.markdown("**Before**")
            st.image(
                composite_on_color(source, preview_color),
                use_container_width=True,
            )

        with after_col:
            st.markdown("**After**")
            st.image(
                composite_on_color(cleaned, preview_color),
                use_container_width=True,
            )

        st.download_button(
            "Download cleaned PNG",
            data=image_to_png_bytes(cleaned),
            file_name=output_name,
            mime="image/png",
            key=f"download_{index}",
        )

        with st.expander("Check on four backgrounds"):
            columns = st.columns(4)
            checks = [
                ("White", "#FFFFFF"),
                ("Gray", "#BEBEBE"),
                ("Dark", "#202020"),
                ("Color", "#91A9BD"),
            ]

            for column, (label, color) in zip(columns, checks):
                with column:
                    st.caption(label)
                    st.image(
                        composite_on_color(cleaned, color),
                        use_container_width=True,
                    )

        if debug_mode:
            rgba = np.array(open_rgba(source), dtype=np.uint8)
            alpha = rgba[..., 3]

            st.markdown("### Debug information")
            st.write(
                {
                    "Image mode": source.mode,
                    "Dimensions": f"{source.width} × {source.height}",
                    "Transparent pixels": int(np.sum(alpha == 0)),
                    "Partially transparent pixels": int(
                        np.sum((alpha > 0) & (alpha < 255))
                    ),
                    "Opaque pixels": int(np.sum(alpha == 255)),
                }
            )

            debug_cols = st.columns(2)

            with debug_cols[0]:
                st.markdown("**Alpha channel**")
                st.image(
                    alpha_preview(source),
                    use_container_width=True,
                )

            with debug_cols[1]:
                st.markdown("**Processed edge band**")
                st.image(
                    edge_preview(source, edge_width),
                    use_container_width=True,
                )

        st.divider()

    except Exception as exc:
        st.error(f"Could not process {uploaded.name}: {exc}")

if len(processed_files) > 1:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename, image in processed_files:
            archive.writestr(filename, image_to_png_bytes(image))

    st.download_button(
        "Download all cleaned PNGs as ZIP",
        data=zip_buffer.getvalue(),
        file_name="halo_cleaned_pngs.zip",
        mime="application/zip",
    )

st.caption(
    "Recommended starting point for your farm PNG: "
    "Medium · Halo width 3 · Former background white · "
    "Repair solid fringe enabled."
)
