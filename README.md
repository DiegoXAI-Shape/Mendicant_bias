# Mendicant Bias: Clasificador de Gatos vs Perros (From Kaggle) 🐾

> El sistema de visión aritifical diseñado y siendo todavía trabajado para superar los sesgos de textura, contexto y brillo, para evitar que la red neuronal convolucional aprenda "atajos" y sea flojo al predecir diciendo: "Hay jaula, entonces es gato", lo cual es una premisa falsa para la clasificación de perros y gatos.

# Retos
El principal reto que afronté con este proyecto fue la ineficiencia de mis primeras arquitecturas caseras de forma secuencial la cual, no obtuvo nada de buenas métricas,

Otro gran reto que afronté fueron los sesgos y que mi modelo sea flojo al predecir en base a las características de las imágenes, sobre todo el sesgo de textura y el sesgo de brillo.

**Restricción técnica:** no usar modelos pre-entrenados, para ir conociendo más a fondo las arquitecturas modernas como ResNet en base a su paper: https://arxiv.org/pdf/1512.03385

## Metodología e Ingeniería

### 1. Diagnóstico de Sesgos (XAI)
Utilicé **Grad-CAM (Gradient-weighted Class Activation Mapping)** implementado manualmente con *hooks* de PyTorch para visualizar qué estaba "mirando" el modelo.
* **Hallazgo:** El modelo inicial tenía un fuerte **Sesgo de Contexto**. Clasificaba "Gato" al detectar barrotes verticales (jaulas) y "Perro" al detectar texturas de suelo, ignorando al animal.

### 2. Limpieza y Preprocesamiento
* Implementación de scripts de **Bash/Python** para filtrar imágenes corruptas.
* **Data Augmentation** estratégico (Rotación, Random Invert, Color Jitter) para obligar al modelo a aprender formas y no solo colores/texturas.

### 3. Arquitectura: "Mendicant Bias v3"
Diseñé e implementé una variante de **ResNet-18 desde cero** en PyTorch:
* **Stem Agresivo:** Convolución 7x7 inicial para reducción espacial rápida.
* **Bloques Residuales Custom:** Implementación manual de *Skip Connections* para evitar el desvanecimiento del gradiente.
* **Regularización:** Uso intensivo de `BatchNormalization`, `Dropout2d` (espacial) y `Weight Decay` para penalizar la memorización.

## 📈 Resultados

| Métrica | Modelo v1 (CNN Simple) | Mendicant Bias v3 (ResNet Custom) |
| :--- | :--- | :--- |
| **Precisión (Val)** | 78.3% | **96.45%** |
| **Loss** | 0.46 | **0.11** |
| **Generalización** | Pobre (Sesgo de textura) | **Robusta** (Ignora fondos) |

### 🖼️ Evidencia Visual (Grad-CAM)
*(Aquí puedes poner tus imágenes del antes y después)*
* **Antes:** El heatmap se encendía en la jaula.
* **Después:** El heatmap se enfoca exclusivamente en la cara y orejas del animal, ignorando el entorno.

## 💻 Tecnologías
* **PyTorch** (Entrenamiento, Arquitectura, Hooks)
* **OpenCV** (Preprocesamiento y visualización)
* **Pandas/NumPy** (Análisis de datos)
* **CUDA** (Entrenamiento en GPU)

---
*Proyecto desarrollado por Diego Asael Hernández Cardona.*
