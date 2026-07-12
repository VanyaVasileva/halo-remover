
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
    "Male grob über den störenden Rand. Der Brush verändert nur eine schmale Zone "
    "direkt an der transparenten Außenkante."
)


def open_rgba(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGBA")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def composite(image: Image.Image, color: str) -> Image.Image:
    rgba = open_rgba(image)
    background = Image.new(
        "RGBA",
        rgba.size,
        ImageColor.getrgb(color) + (255,),
    )
    return Image.alpha_composite(background, rgba).convert("RGB")


def resize_for_editor(
    image: Image.Image,
    max_width: int = 1000,
    max_height: int = 720,
) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(max_width / width, max_height / height, 1.0)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def build_edge_zone(alpha: np.ndarray, width: int) -> np.ndarray:
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
        & (alpha >= 170)
    )

    if not np.any(seed):
        seed = visible & (
            distance_inside >= max(2, edge_width + 1)
        )

    if not np.any(seed):
        seed = visible & (alpha >= 220)

    if not np.any(seed):
        seed = visible

    _, nearest = ndimage.distance_transform_edt(
        ~seed,
        return_indices=True,
    )

    return rgb[nearest[0], nearest[1]]


def extract_brush_mask(
    canvas_rgba: np.ndarray | None,
    original_size: tuple[int, int],
) -> np.ndarray:
    """
    The brush is bright magenta. Detect only those painted pixels,
    then resize the mask back to the original PNG dimensions.
    """
    width, height = original_size

    if canvas_rgba is None:
        return np.zeros((height, width), dtype=bool)

    data = np.asarray(canvas_rgba)

    red = data[..., 0]
    green = data[..., 1]
    blue = data[..., 2]
    alpha = data[..., 3]

    magenta = (
        (red >= 210)
        & (blue >= 210)
        & (green <= 120)
        & (alpha >= 80)
    )

    mask_image = Image.fromarray(
        (magenta.astype(np.uint8) * 255),
        mode="L",
    )
    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST,
    )

    return np.asarray(mask_image) > 0


def apply_halo_brush(
    image: Image.Image,
    brush_mask: np.ndarray,
    edge_width: int,
    strength: float,
    remove_outer_ring: bool,
) -> Image.Image:
    rgba = np.array(open_rgba(image), dtype=np.uint8)

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32)

    edge_zone = build_edge_zone(alpha, edge_width)
    active = edge_zone & brush_mask

    if not np.any(active):
        return Image.fromarray(rgba, mode="RGBA")

    interior_rgb = nearest_interior_rgb(
        rgb=rgb,
        alpha=alpha,
        edge_width=edge_width,
    )

    rgb[active] = (
        rgb[active] * (1.0 - strength)
        + interior_rgb[active] * strength
    )

    if remove_outer_ring:
        visible = alpha > 0
        distance_inside = ndimage.distance_transform_edt(visible)

        outer_ring = (
            visible
            & (distance_inside <= 1.05)
            & brush_mask
        )

        alpha[outer_ring] = 0.0

    rgb[alpha <= 0] = 0

    output = np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ]
    )

    return Image.fromarray(output, mode="RGBA")


def restore_brush(
    edited: Image.Image,
    original: Image.Image,
    brush_mask: np.ndarray,
) -> Image.Image:
    edited_array = np.array(open_rgba(edited), dtype=np.uint8)
    original_array = np.array(open_rgba(original), dtype=np.uint8)

    edited_array[brush_mask] = original_array[brush_mask]

    return Image.fromarray(edited_array, mode="RGBA")


st.sidebar.header("Brush-Einstellungen")

tool = st.sidebar.radio(
    "Werkzeug",
    ["Halo Brush", "Restore Brush"],
)

edge_width = st.sidebar.slider(
    "Geschützte Randzone",
    min_value=1,
    max_value=8,
    value=3,
    help=(
        "Der Halo Brush darf nur innerhalb dieser schmalen Außenkante wirken. "
        "Das Motivinnere bleibt geschützt."
    ),
)

brush_size = st.sidebar.slider(
    "Pinselgröße",
    min_value=5,
    max_value=120,
    value=35,
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

remove_outer_ring = st.sidebar.checkbox(
    "Äußersten Pixelring unter dem Brush entfernen",
    value=False,
    help=(
        "Nur einschalten, wenn ein fester weißer Rand stehen bleibt. "
        "Der Brush entfernt dann ausschließlich den äußersten Pixelring."
    ),
)

preview_choice = st.sidebar.selectbox(
    "Arbeits-Hintergrund",
    ["White", "Light gray", "Dark", "Custom"],
)

preview_colors = {
    "White": "#FFFFFF",
    "Light gray": "#C8C8C8",
    "Dark": "#1E1E1E",
}

if preview_choice == "Custom":
    preview_color = st.sidebar.color_picker(
        "Eigene Hintergrundfarbe",
        "#91A9BD",
    )
else:
    preview_color = preview_colors[preview_choice]

uploaded = st.file_uploader(
    "Transparente PNG hochladen",
    type=["png"],
)

if uploaded is None:
    st.info("Lade eine transparente PNG hoch.")
    st.stop()

source = open_rgba(Image.open(uploaded))

file_signature = (
    uploaded.name,
    source.size,
)

if (
    "file_signature" not in st.session_state
    or st.session_state.file_signature != file_signature
):
    st.session_state.file_signature = file_signature
    st.session_state.original_image = source.copy()
    st.session_state.edited_image = source.copy()
    st.session_state.canvas_version = 0

original = st.session_state.original_image
edited = st.session_state.edited_image

editor_background = composite(edited, preview_color)
editor_background, _ = resize_for_editor(editor_background)

st.markdown(
    "**Male grob über die störenden weißen oder grauen Ränder.** "
    "Der Brush korrigiert ausschließlich die geschützte Außenkante."
)

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 255, 0.45)",
    stroke_width=brush_size,
    stroke_color="rgba(255, 0, 255, 0.95)",
    background_image=editor_background,
    update_streamlit=True,
    height=editor_background.height,
    width=editor_background.width,
    drawing_mode="freedraw",
    key=f"halo_canvas_{st.session_state.canvas_version}",
)

button_1, button_2, button_3 = st.columns(3)

with button_1:
    apply_clicked = st.button(
        "Brush anwenden",
        type="primary",
        use_container_width=True,
    )

with button_2:
    clear_clicked = st.button(
        "Pinselstriche löschen",
        use_container_width=True,
    )

with button_3:
    reset_clicked = st.button(
        "Original wiederherstellen",
        use_container_width=True,
    )

if reset_clicked:
    st.session_state.edited_image = original.copy()
    st.session_state.canvas_version += 1
    st.rerun()

if clear_clicked:
    st.session_state.canvas_version += 1
    st.rerun()

if apply_clicked:
    brush_mask = extract_brush_mask(
        canvas_rgba=canvas_result.image_data,
        original_size=original.size,
    )

    if not np.any(brush_mask):
        st.warning("Male zuerst über einen problematischen Rand.")
    else:
        if tool == "Halo Brush":
            st.session_state.edited_image = apply_halo_brush(
                image=edited,
                brush_mask=brush_mask,
                edge_width=edge_width,
                strength=strength_map[strength_label],
                remove_outer_ring=remove_outer_ring,
            )
        else:
            st.session_state.edited_image = restore_brush(
                edited=edited,
                original=original,
                brush_mask=brush_mask,
            )

        st.session_state.canvas_version += 1
        st.rerun()

edited = st.session_state.edited_image

st.markdown("### Kontrolle auf vier Hintergründen")

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
    "Empfohlener Start: Randzone 3 px, Medium, Pixelring ausgeschaltet. "
    "Nur über die sichtbar problematischen Stellen malen."
)
