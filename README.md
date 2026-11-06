# App "¿Qué famoso es?"

Subís una foto y dice cuál de los 20 actores del dataset es y qué emoción muestra.

- Identidad: CNN Inception entrenada desde cero (0,971 de F1 macro en test).
- Emoción: MobileNetV2 con transfer learning (0,484 de F1 macro en test).
- Dos orígenes: elegir una foto del dataset (una al azar, filtrable por split; muestra si acierta) o subir una foto propia.
- La foto tiene que venir recortada a la cara; si no es cuadrada se toma el cuadrado central y se lleva a 224×224.
- Localmente usa el dataset completo; publicada usa las 411 fotos de test incluidas en `fotos_test/`.
- El modelo solo conoce a esas 20 personas y siempre elige una: si la confianza es baja, lo avisa.

## Correr localmente

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Los modelos (`modelos/*.pt`, TorchScript) se regeneran desde la raíz del proyecto con `python tools/exportar_app.py`.

## Publicar en Streamlit Community Cloud

1. Crear un repositorio en GitHub (recomendado: **privado**) y subir el contenido de esta carpeta
   (`app.py`, `requirements.txt`, `modelos/`). No subir `.venv/` (ya está en `.gitignore`) ni imágenes del dataset.
2. Entrar a https://share.streamlit.io con la cuenta de GitHub, "Create app", elegir el repo, rama y `app.py`.
3. En la configuración de la app, en "Sharing", dejarla visible solo para emails elegidos
   (el enunciado pide que el material se use solo dentro del curso).
