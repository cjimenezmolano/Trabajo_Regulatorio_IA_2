# ¿Qué famoso es?

TP1 de IA y Aprendizaje Automático II (UCA Rosario).

Clasifica de qué famoso es una foto (20 personas) y qué emoción tiene.

- Identidad: CNN tipo Inception entrenada desde cero.
- Emoción: MobileNetV2 con transfer learning.

Se puede elegir una foto de test al azar o subir una propia (recortada a la cara).

```
pip install -r requirements.txt
streamlit run app.py
```
