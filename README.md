# Background & Halo Remover

A conservative Streamlit prototype for:

- removing a mostly uniform background from JPEG or PNG images
- preserving fine watercolor and clipart details
- cleaning light or dark halos from already transparent PNGs
- previewing results on multiple backgrounds
- batch export as transparent PNG files

## Files

- `app.py`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud.
4. Create a new app from the repository.
5. Set the main file path to `app.py`.
