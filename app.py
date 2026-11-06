import csv
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
LOCAL = (RAIZ / "Imagenes_Finales-20261102T234933Z-1-001" / "Imagenes_Finales", RAIZ / "splits" / "splits_final.csv")
REPO = (AQUI / "fotos_test", AQUI / "fotos_test" / "etiquetas.csv")
FOTOS, SPLITS = LOCAL if LOCAL[1].exists() else REPO
LADO = 224


@st.cache_resource
def cargar():
    meta = json.loads((AQUI / "modelos" / "clases.json").read_text(encoding="utf-8"))
    modelos = {t: torch.jit.load(str(AQUI / "modelos" / f"{t}.pt"), map_location="cpu").eval() for t in meta}
    return meta, modelos


@st.cache_data
def dataset():
    if not (SPLITS.exists() and FOTOS.exists()):
        return []
    with open(SPLITS, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (FOTOS / r["archivo"]).exists()]


def recortar_centro(imagen):
    ancho, alto = imagen.size
    lado = min(ancho, alto)
    x0, y0 = (ancho - lado) // 2, (alto - lado) // 2
    return np.asarray(imagen.crop((x0, y0, x0 + lado, y0 + lado)).resize((LADO, LADO), Image.BOX))


def predecir(modelo, recorte, size):
    x = torch.from_numpy(recorte.copy()).permute(2, 0, 1).float().unsqueeze(0)
    if size != LADO:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False).round()
    with torch.no_grad():
        return torch.softmax(modelo(x), dim=1)[0].numpy()


meta, modelos = cargar()
filas = dataset()

st.title("¿Qué famoso es?")

real = None
if filas and st.radio("Origen", ["Dataset", "Subir foto"], horizontal=True) == "Dataset":
    splits = sorted({r["split"] for r in filas}, key=["test", "val", "train"].index)
    split = st.radio("Split", splits + ["todos"], horizontal=True) if len(splits) > 1 else "test"
    opciones = [r for r in filas if split in (r["split"], "todos")]
    if st.button("Foto aleatoria") or st.session_state.get("split") != split:
        st.session_state.update(split=split, fila=random.choice(opciones))
    real = st.session_state["fila"]
    st.caption(f'{real["archivo"]} · {real["split"]}')
    imagen = Image.open(FOTOS / real["archivo"]).convert("RGB")
else:
    archivo = st.file_uploader("Foto", type=["jpg", "jpeg", "png"])
    imagen = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB") if archivo else None

if imagen is not None:
    recorte = recortar_centro(imagen)
    st.image(recorte, width=LADO)
    for tarea, col in (("identidad", "actor"), ("emocion", "emocion")):
        p = predecir(modelos[tarea], recorte, meta[tarea]["img_size"])
        clases = meta[tarea]["clases"]
        orden = np.argsort(p)[::-1][:3]
        texto = " · ".join(f"{clases[i]} {p[i]:.0%}" for i in orden)
        marca = "" if real is None else (" ✓" if clases[orden[0]] == real[col] else f' ✗ (real: {real[col]})')
        st.write(f"**{tarea}**: {texto}{marca}")
