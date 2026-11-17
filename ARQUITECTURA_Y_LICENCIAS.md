# Arquitectura y licenciamiento del modelo OCR (NellOCR)

Fecha: 2026-11-06

## 1. Resumen ejecutivo

**¿Es 100% open source y libre para uso comercial?**

**Sí, a partir de este reentrenamiento.** El checkpoint original (primera y
segunda versión) sí incluía datos de **XFUND-ES**, con licencia
**CC BY-NC-SA 4.0 (No Comercial)**, lo cual lo dejaba fuera de uso comercial.
Decidimos quitar XFUND-ES del dataset y reentrenar el modelo **solo con datos
sintéticos propios** (sin ninguna restricción de licencia). Ese checkpoint
"comercial" queda guardado en `output/best_accuracy/` una vez termina el
entrenamiento en curso.

El checkpoint anterior (con XFUND, **no comercial**) se conservó como
respaldo en `output_xfund_no_comercial/` solo para fines de comparación/
investigación — **no debe distribuirse ni usarse comercialmente**, y por eso
está explícitamente excluido del `.gitignore` del repositorio público.

Con el checkpoint nuevo: **código, modelo base, dataset de entrenamiento y
librerías de la API son 100% open source y permiten uso comercial sin
restricciones.**

---

## 2. Arquitectura del modelo

### 2.1 Pipeline de OCR (dos modelos en cadena)

```
Imagen → [Detección de texto] → recortes por línea → [Reconocimiento de texto] → texto
```

| Etapa | Modelo | Arquitectura | Origen de los pesos |
|---|---|---|---|
| Detección | `PP-OCRv5_mobile_det` | Red de detección de texto PP-OCRv5 (DB-based) | Modelo oficial de PaddleOCR, **sin fine-tuning** (se usa tal cual) |
| Reconocimiento | `latin_PP-OCRv5_mobile_rec` | Ver 2.2 | Modelo oficial de PaddleOCR + **fine-tuning propio** |

Solo el modelo de **reconocimiento** fue reentrenado; el de **detección** es
el oficial de PaddleOCR sin modificar.

### 2.2 Arquitectura del modelo de reconocimiento

- **Backbone**: PPLCNetV3 (scale 0.95) — red convolucional ligera
- **Cabezas (MultiHead)**:
  - **CTCHead** (principal): Neck tipo SVTR (dims 120, depth 2, hidden_dims 120, kernel_size [1,3]) + cabeza CTC
  - **NRTRHead** (auxiliar, guía de entrenamiento): dim 384, longitud máxima de secuencia 25
- **Función de pérdida**: `MultiLoss` = `CTCLoss + NRTRLoss`
- **Algoritmo**: SVTR_LCNet (el mismo usado en PP-OCRv5 oficial)
- **Diccionario de caracteres**: 836 símbolos (latino extendido: tildes, ñ, ¿¡, símbolos técnicos, moneda, etc.), heredado íntegro del modelo oficial

### 2.3 Hiperparámetros del fine-tuning

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Learning rate | 0.00015 (Cosine, warmup 5 épocas) |
| Weight decay (L2) | 3e-5 |
| Batch size | 8 |
| Épocas | 20 |
| Resolución de entrada | 48×320 px (crops de una sola línea) |
| Hardware de entrenamiento | GPU local (NVIDIA RTX 4060, CUDA/cuDNN) |

### 2.4 Datos de entrenamiento — versión comercial actual (2 lotes, ambos propios)

| Lote | Contenido | Origen | Licencia |
|---|---|---|---|
| Sintético (hojas de vida) | ~322 líneas generadas con PIL, vocabulario de CV en español | Generado por nosotros | Propio, sin restricción |
| Sintético (refuerzo técnico) | ~480 líneas con separador `\|`, siglas (SQL/API/PHP), dígitos, fechas | Generado por nosotros | Propio, sin restricción |

> **Historial:** una versión intermedia del modelo incluyó además ~3.077
> líneas reales de **XFUND-ES** (Microsoft Research Asia, licencia
> **CC BY-NC-SA 4.0 — No Comercial**) para ganar robustez frente a ruido de
> escaneo real. Esa versión se conservó por separado en
> `output_xfund_no_comercial/` y **no se usa ni se distribuye** — fue
> reemplazada por el checkpoint actual, entrenado solo con los dos lotes
> sintéticos de arriba, sin ninguna restricción de licencia.

> Nota sobre fuentes tipográficas: las imágenes sintéticas se renderizaron con
> fuentes de Windows (Arial, Calibri, Georgia, Consolas). Esto **no** genera
> ninguna restricción de licencia sobre el modelo o las imágenes resultantes
> — es equivalente a tomar una captura de pantalla de texto; no se redistribuye
> el archivo de la fuente.

---

## 3. Herramientas y dependencias de software

### 3.1 Entrenamiento

| Herramienta | Función | Licencia | ¿Uso comercial? |
|---|---|---|---|
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) (paddlepaddle-gpu) | Framework de deep learning | Apache 2.0 | ✅ Sí |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Modelos OCR + scripts de entrenamiento (`tools/train.py`) | Apache 2.0 | ✅ Sí |
| [PaddleX](https://github.com/PaddlePaddle/PaddleX) | Orquestación del pipeline de entrenamiento/export | Apache 2.0 | ✅ Sí |
| [PPOCRLabel](https://pypi.org/project/PPOCRLabel/) | Herramienta de etiquetado manual (parte del ecosistema PaddleOCR) | Apache 2.0 | ✅ Sí |
| CUDA / cuDNN (NVIDIA) | Runtime de GPU | Propietario (EULA de NVIDIA, gratuito para este uso) | ✅ Sí, bajo el EULA estándar de NVIDIA — no se redistribuye, solo se usa localmente |

### 3.2 Servicio / API

| Herramienta | Función | Licencia | ¿Uso comercial? |
|---|---|---|---|
| [FastAPI](https://github.com/fastapi/fastapi) | Framework web de la API | MIT | ✅ Sí |
| [Uvicorn](https://github.com/encode/uvicorn) | Servidor ASGI | BSD-3-Clause | ✅ Sí |
| [Pillow](https://github.com/python-pillow/Pillow) | Procesamiento de imágenes | MIT-CMU (HPND) | ✅ Sí |
| [NumPy](https://numpy.org/) | Cómputo numérico | BSD-3-Clause | ✅ Sí |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) | Convierte las páginas de un PDF en imágenes (motor PDFium) | Apache 2.0 / BSD-3-Clause | ✅ Sí |
| [OpenCV](https://opencv.org/) | Corrige la inclinación de hojas escaneadas | Apache 2.0 | ✅ Sí |
| [python-multipart](https://pypi.org/project/python-multipart/) | Subida de archivos en FastAPI | Apache 2.0 | ✅ Sí |

**Todo el software del stack (framework, modelos base, librerías de la API) es
open source con licencias permisivas (Apache 2.0 / MIT / BSD) y permite uso
comercial sin restricciones ni regalías.** El único punto de atención es el
*dataset* usado para entrenar, cubierto en la sección siguiente.

---

## 4. Qué se hizo con XFUND-ES (historial de la decisión)

### 4.1 Por qué era un problema

XFUND (incluido el subconjunto en español que usamos) está licenciado bajo
**CC BY-NC-SA 4.0**, que prohíbe explícitamente el uso comercial de los datos
y de cualquier obra derivada "sustancialmente" de ellos. Un modelo entrenado
con esos datos se considera una obra derivada, así que ese checkpoint
intermedio no podía ofrecerse como parte de un producto o servicio
comercial.

Esto nunca afectó al código, al framework, ni al modelo base preentrenado de
PaddleOCR — esos siempre fueron libres. El límite era específicamente ese
checkpoint fine-tuneado con datos de XFUND.

### 4.2 Decisión tomada

Se eligió la opción más limpia: **quitar por completo las ~3.077 líneas de
XFUND-ES** de `dataset/train.txt` / `dataset/val.txt` (identificables por el
prefijo `xfund_es_`) y **reentrenar desde cero** usando únicamente los lotes
sintéticos propios. El checkpoint resultante en `output/best_accuracy/` no
tiene ninguna dependencia de datos con licencia restrictiva.

El checkpoint anterior (con XFUND) se conservó únicamente como referencia
histórica en `output_xfund_no_comercial/`, fuera del repositorio público y
sin uso en producción.

### 4.3 Si en el futuro se necesita más robustez a ruido de escaneo real

Si más adelante el modelo necesita leer documentos físicamente escaneados o
fotografiados (no solo exportaciones digitales limpias), la robustez que
aportaba XFUND tendría que recuperarse con:
- más variedad sintética (ruido, desenfoque, rotación, compresión JPEG), o
- datos reales propios, recolectados y anotados con las autorizaciones
  correspondientes (no un dataset de terceros con licencia NC).

---

## 5. Resumen de la cadena de custodia de cada componente

```
Código de entrenamiento/inferencia .......... Apache 2.0 / MIT / BSD   → libre y comercial
Modelo base preentrenado (PP-OCRv5) ......... Apache 2.0               → libre y comercial
Datos sintéticos propios (2 lotes) ........... Sin restricción          → libre y comercial
─────────────────────────────────────────────────────────────────────────────
Checkpoint actual (output/best_accuracy)  ..... solo datos propios      → libre y comercial ✅
Checkpoint histórico (output_xfund_no_comercial/) → incluye XFUND-ES   → NO comercial, no se distribuye
```

**Sources:**
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [XFUND GitHub](https://github.com/doc-analysis/XFUND)
- [FastAPI GitHub](https://github.com/fastapi/fastapi)
- [Pillow LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
