# 🛰️ Roland-Infinity / Daowa-Maad: Attention ResU-Net for Semantic Segmentation

> *"A biologically selective AI commanded by the Infinity."*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

## 📖 Descripción
**Roland-Infinity** es un modelo de segmentación semántica de alto rendimiento diseñado para distinguir mascotas (perros y gatos) de fondos complejos con una precisión al **90%**.

A diferencia de las U-Nets tradicionales, este modelo implementa **Attention Gates** personalizadas que le permiten aprender características semánticas robustas. Esto le otorga la capacidad única de ignorar oclusiones (rejas), objetos extraños (ropa) y representaciones no biológicas (dibujos/emojis), enfocándose puramente en las características biológicas del animal.

## 🧠 Arquitectura: Daowa-Maad
El modelo utiliza una arquitectura híbrida construida desde cero:

1.  **Encoder (Bajada):** Bloques residuales estilo ResNet para una extracción profunda de características.
2.  **Attention Gates:**
    * En lugar de pasar toda la información del Encoder al Decoder a través de las *skip connections* (como en una U-Net estándar), implementamos mecanismos de atención.
    * Estos actúan como filtros que suprimen las regiones irrelevantes de la imagen (fondo, ropa, ruido) y resaltan las características salientes (ojos, orejas, textura de pelo) antes de la fusión.
3.  **Decoder (Subida):** Recuperación de la resolución espacial mediante upsampling bilineal y convoluciones refinadas.

### 📉 Métricas de Entrenamiento
* **Loss Function:** Estrategia "Burn-in" (Cross Entropy inicial -> Generalized Dice Loss).
* **Optimizador:** Adam.
* **Precisión en Test (Dev):** **90.34%** (Datos / imágenes que nunca ha visto).
* **Comportamiento:** Alta generalización y resistencia al overfitting.

## 🚀 Resultados

El modelo demuestra una robustez inusual en escenarios difíciles:

| Escenario | Resultado | Análisis |
| :--- | :--- | :--- |
| **Oclusión (Rejas)** | ✅ **Éxito** | El modelo ignora los barrotes y segmenta al perro detrás de ellos. |
| **Out-of-Distribution (Ropa)** | ✅ **Éxito** | Distingue la textura del gato vs. la textura de la tela (trajes/corbatas), recortando solo al animal. |
| **Falsos Positivos (Emojis)** | ✅ **Éxito** | Discrimina entre un gato real y ediciones digitales (manos de emoji/stickers). |

## 🛠️ Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/DiegoXAI-Shape/Mendicant_bias.git](https://github.com/DiegoXAI-Shape/Mendicant_bias.git)
   cd Mendicant_bias

2. **Instalar las dependencias:**
   ```bash
   pip install -r requeriments.txt
   
3. **Predecir:**
  ```
  import torch
  from model import Daowa_maad # Asegúrate de importar tu clase

  # Cargar el modelo
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Daowa_maad(num_clases=3).to(device)
  model.load_state_dict(torch.load("Roland_Epoch20.pth", map_location=device))
  model.eval()
 ```

-----------------------------------------------------------------------------------------------------------------------------------------------------------
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
