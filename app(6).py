
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageColor, ImageOps
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Halo Studio",
    page_icon="🪄",
    layout="wide",
)

st.title("Halo Studio")
st.caption(
    "Halbautomatische Randkorrektur für transparente PNG-Motive. "
    "Der Pinsel wirkt nur in einer schmalen Zone entlang der Außenkante."
)


def open_rgba(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGBA")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def composite(image: Image.Image, color: str) -> Image.Image:
    rgba = open_rgba(image)
    bg = Image.new("RGBA", rgba.size, ImageColor.getrgb(color) + (255,))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def build_edge_zone(alpha: np.ndarray, width: int) -> np.ndarray:
    visible = alpha > 0
    if not np.any(visible):
        return visible
    distance_inside = ndimage.distance_transform_edt(visible)
    return visible & (distance_inside <= max(1, int(width)))


def nearest_interior_colors(
    rgb: np.ndarray,
    alpha: np.ndarray,
    edge_width: int,
) -> np.ndarray:
    visible = alpha > 0
    distance_inside = ndimage.distance_transform_edt(visible)

    seed = (
        visible
        & (distance_inside >= max(3, edge_width + 2))
        & (alpha >= 170)
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


def resize_for_canvas(
    image: Image.Image,
    max_width: int = 950,
    max_height: int = 700,
) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(max_width / width, max_height / height, 1.0)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def canvas_mask_to_original(
    mask_rgba: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    if mask_rgba is None:
        return np.zeros((original_size[1], original_size[0]), dtype=bool)

    alpha = mask_rgba[..., 3]
    mask_image = Image.fromarray(alpha.astype(np.uint8), mode="L")
    mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)
    return np.array(mask_image) > 10


def apply_edge_heal(
    original: Image.Image,
    paint_mask: np.ndarray,
    edge_width: int,
    strength: float,
    trim_outermost: bool,
) -> Image.Image:
    rgba = np.array(open_rgba(original), dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32)

    edge_zone = build_edge_zone(alpha, edge_width)
    active = edge_zone & paint_mask

    if not np.any(active):
        return Image.fromarray(rgba, mode="RGBA")

    inner_rgb = nearest_interior_colors(rgb, alpha, edge_width)

    # Replace the contaminated edge color with the closest clean interior color.
    weight = np.clip(strength, 0.0, 1.0)
    rgb[active] = (
        rgb[active] * (1.0 - weight)
        + inner_rgb[active] * weight
    )

    # Optional removal of only the outermost pixel ring under the brush.
    if trim_outermost:
        visible = alpha > 0
        distance_inside = ndimage.distance_transform_edt(visible)
        outermost = visible & (distance_inside <= 1.05) & paint_mask
        alpha[outermost] = 0.0

    rgb[alpha <= 0] = 0

    out = np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ]
    )
    return Image.fromarray(out, mode="RGBA")


def restore_from_original(
    edited: Image.Image,
    original: Image.Image,
    restore_mask: np.ndarray,
) -> Image.Image:
    edited_rgba = np.array(open_rgba(edited), dtype=np.uint8)
    original_rgba = np.array(open_rgba(original), dtype=np.uint8)

    edited_rgba[restore_mask] = original_rgba[restore_mask]
    return Image.fromarray(edited_rgba, mode="RGBA")


st.sidebar.header("Werkzeug")

tool = st.sidebar.radio(
    "Modus",
    ["Halo Brush", "Restore Brush"],
)

edge_width = st.sidebar.slider(
    "Geschützte Randzone",
    min_value=1,
    max_value=8,
    value=3,
    help="Der Pinsel darf nur innerhalb dieser Außenkante wirken.",
)

brush_size = st.sidebar.slider(
    "Pinselgröße",
    min_value=5,
    max_value=100,
    value=30,
)

strength_label = st.sidebar.select_slider(
    "Korrekturstärke",
    options=["Gentle", "Medium", "Strong"],
    value="Medium",
)
strength_map = {
    "Gentle": 0.45,
    "Medium": 0.72,
    "Strong": 1.0,
}

trim_outermost = st.sidebar.checkbox(
    "Äußersten Pixelring unter dem Pinsel entfernen",
    value=False,
    help="Nur aktivieren, wenn ein fester weißer Rand stehen bleibt.",
)

preview_bg = st.sidebar.selectbox(
    "Arbeits-Hintergrund",
    ["White", "Light gray", "Dark", "Custom"],
)

preview_colors = {
    "White": "#FFFFFF",
    "Light gray": "#C8C8C8",
    "Dark": "#1E1E1E",
}

if preview_bg == "Custom":
    preview_color = st.sidebar.color_picker(
        "Eigene Hintergrundfarbe",
        "#91A9BD",
    )
else:
    preview_color = preview_colors[preview_bg]

uploaded = st.file_uploader(
    "Transparente PNG hochladen",
    type=["png"],
)

if uploaded is None:
    st.info("Lade eine transparente PNG hoch.")
    st.stop()

source = Image.open(uploaded)
source = open_rgba(source)

if "source_name" not in st.session_state or st.session_state.source_name != uploaded.name:
    st.session_state.source_name = uploaded.name
    st.session_state.original_image = source.copy()
    st.session_state.edited_image = source.copy()
    st.session_state.canvas_counter = 0

original = st.session_state.original_image
edited = st.session_state.edited_image

canvas_background = composite(edited, preview_color)
canvas_background, canvas_scale = resize_for_canvas(canvas_background)

st.markdown(
    "**Male grob über den störenden Rand.** "
    "Die Korrektur wirkt nur in der eingestellten Außenkante."
)

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.35)",
    stroke_width=brush_size,
    stroke_color="rgba(255, 0, 0, 0.95)",
    background_image=canvas_background,
    update_streamlit=True,
    height=canvas_background.height,
    width=canvas_background.width,
    drawing_mode="freedraw",
    key=f"halo_canvas_{st.session_state.canvas_counter}",
)

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    apply_clicked = st.button(
        "Pinsel anwenden",
        type="primary",
        use_container_width=True,
    )

with button_col2:
    clear_clicked = st.button(
        "Pinselstriche löschen",
        use_container_width=True,
    )

with button_col3:
    reset_clicked = st.button(
        "Original wiederherstellen",
        use_container_width=True,
    )

if reset_clicked:
    st.session_state.edited_image = original.copy()
    st.session_state.canvas_counter += 1
    st.rerun()

if clear_clicked:
    st.session_state.canvas_counter += 1
    st.rerun()

if apply_clicked:
    if canvas_result.image_data is None:
        st.warning("Male zuerst über eine problematische Stelle.")
    else:
        paint_mask = canvas_mask_to_original(
            canvas_result.image_data,
            original.size,
        )

        if tool == "Halo Brush":
            st.session_state.edited_image = apply_edge_heal(
                original=edited,
                paint_mask=paint_mask,
                edge_width=edge_width,
                strength=strength_map[strength_label],
                trim_outermost=trim_outermost,
            )
        else:
            st.session_state.edited_image = restore_from_original(
                edited=edited,
                original=original,
                restore_mask=paint_mask,
            )

        st.session_state.canvas_counter += 1
        st.rerun()

edited = st.session_state.edited_image

st.markdown("### Kontrolle auf verschiedenen Hintergründen")

preview_cols = st.columns(4)
checks = [
    ("White", "#FFFFFF"),
    ("Gray", "#BEBEBE"),
    ("Dark", "#202020"),
    ("Color", "#91A9BD"),
]

for column, (label, color) in zip(preview_cols, checks):
    with column:
        st.caption(label)
        st.image(
            composite(edited, color),
            use_container_width=True,
        )

output_name = f"{Path(uploaded.name).stem}_halo_studio.png"

st.download_button(
    "Bereinigte PNG herunterladen",
    data=png_bytes(edited),
    file_name=output_name,
    mime="image/png",
    use_container_width=True,
)

st.caption(
    "Empfehlung: Randzone 3 px, Medium, äußersten Pixelring zunächst ausgeschaltet. "
    "Nur bei einem festen weißen Saum einschalten."
)
