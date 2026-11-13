# Procedimiento técnico de entrenamiento — NellOCR

Documento de referencia (runbook) con el procedimiento completo, de punta a
punta, para reproducir el fine-tuning del modelo de reconocimiento OCR,
incluyendo los problemas reales encontrados y cómo se resolvieron. Pensado
para que cualquiera pueda repetir el proceso
sin tener que redescubrir los mismos errores.

---

## 1. Entorno

### 1.1 Hardware usado
- GPU: NVIDIA RTX 4060 (local, Windows)
- CUDA Compute Capability: 8.9
- Driver API: 13.5 / Runtime API: 12.6

### 1.2 Instalación del entorno Python

```powershell
python -m venv .venv
.venv\Scripts\pip install paddlepaddle-gpu paddleocr pillow
```

Esto instala lo mínimo para **inferencia** (correr OCR con modelos ya
entrenados). Para **entrenar** hace falta un paso adicional (ver 1.4).

### 1.3 Variables de entorno necesarias

Estas tres variables deben estar presentes en **toda** invocación de
entrenamiento (en PowerShell, antes del comando):

```powershell
$env:DISABLE_MODEL_SOURCE_CHECK = "True"
$env:PADDLE_PDX_PADDLEOCR_PATH = "<ruta>\.venv\Lib\site-packages\paddlex\repo_manager\repos\PaddleOCR"
$env:PYTHONUTF8 = "1"
```

Por qué cada una (detalle en la sección 3, "Problemas encontrados"):
- `DISABLE_MODEL_SOURCE_CHECK`: evita una verificación de conectividad lenta al arrancar.
- `PADDLE_PDX_PADDLEOCR_PATH`: le dice a PaddleX dónde está el código fuente de entrenamiento de PaddleOCR (clonado aparte, ver 1.4).
- `PYTHONUTF8`: evita un crash de encoding en Windows al mostrar la barra de progreso de descargas.

### 1.4 Instalar el plugin de entrenamiento de PaddleOCR

El paquete `paddleocr` de pip **solo trae inferencia**. Entrenar requiere
clonar el repo fuente completo vía PaddleX:

```powershell
python -m paddlex --install PaddleOCR -y
```

Esto clona `PaddleOCR` (y su dependencia `PaddleNLP`) dentro de
`.venv\Lib\site-packages\paddlex\repo_manager\repos\`. Tarda varios minutos
(~70 dependencias adicionales).

---

## 2. Estructura del dataset esperado

PaddleX espera, para el módulo `text_recognition`, esta estructura exacta
dentro de `dataset/`:

```
dataset/
├── dict.txt          # diccionario de caracteres (uno por línea)
├── train.txt          # "images/train/archivo.png\ttexto exacto"
├── val.txt             # mismo formato, para validación
└── images/
    ├── train/*.png     # recortes de UNA sola línea de texto cada uno
    └── val/*.png
```

Puntos clave:
- **Cada imagen es una sola línea de texto**, no una página completa (el
  modelo de reconocimiento no hace segmentación, solo lee lo que se le pasa).
- **`dict.txt` debe coincidir exactamente** con el diccionario del checkpoint
  preentrenado del que partes (mismo orden, mismos caracteres) — si no, el
  tamaño de la última capa (clasificador) no calza con los pesos
  preentrenados. Se extrae así, desde el modelo oficial descargado:

```python
import yaml
with open(r"<cache_paddlex>\official_models\latin_PP-OCRv5_mobile_rec\inference.yml", encoding="utf-8") as f:
    d = yaml.safe_load(f)
chars = d["PostProcess"]["character_dict"]
with open("dataset/dict.txt", "w", encoding="utf-8", newline="\n") as out:
    for c in chars:
        out.write(c + "\n")
```

---

## 3. Generación del dataset (scripts propios)

### 3.1 `generar_dataset.py` — lote base
Renderiza con PIL líneas de texto típicas de una hoja de vida en español
(nombres, títulos, experiencia, correos, teléfonos, fechas) usando varias
fuentes de Windows (Arial, Calibri, Georgia) y tamaños aleatorios. Cada línea
se guarda como imagen + entrada en `train.txt`/`val.txt` (split 85/15).

```powershell
python generar_dataset.py
```

### 3.2 `generar_dataset_tecnico.py` — lote de refuerzo
Se añadió **después** de detectar errores sistemáticos probando el modelo
contra CVs reales (ver sección 5). Refuerza específicamente:
- El separador `|` en encabezados de contacto (el modelo lo confundía con `1`/`J`)
- Siglas técnicas en mayúsculas (SQL, API, PHP, .NET, etc.)
- Dígitos en fechas y cifras (confusión `0`/`o`)
- Espaciado correcto alrededor de puntuación

```powershell
python generar_dataset_tecnico.py
```

Este script **append-ea** a `dataset/train.txt`/`val.txt` existentes (no los
sobreescribe), por lo que el orden de ejecución importa si se parte de cero.

### 3.2b `generar_dataset_multiarea.py` — lote multi-área
Generaliza el modelo a hojas de vida de **cualquier área profesional** (no solo
tecnología) y refuerza los patrones que seguían fallando: separador `|`,
confusión `I`/`l`/`1`, dígitos, porcentajes y viñetas (`•`, `·`). También
**append-ea**.

```powershell
python generar_dataset_multiarea.py
```

**Dataset final (sin datos personales reales):** los tres scripts, en ese
orden y con semilla fija, producen **1.302 líneas de entrenamiento y 228 de
validación** (322+408+572 y 56+72+100). Todos los datos de contacto que
aparecen en las líneas sintéticas son ficticios (`nombre.apellido@correo.com`,
`310 123 4567`, etc.). Para regenerarlo desde cero hay que borrar
`dataset/images`, `dataset/train.txt` y `dataset/val.txt`, **crear**
`dataset/images/train` y `dataset/images/val` (el script base no crea las
carpetas) y ejecutar los tres scripts en orden.

### 3.3 (Descartado) `importar_xfund_es.py` — datos reales de XFUND-ES
Se probó añadir ~3.077 líneas reales del dataset público XFUND-ES (formularios
españoles escaneados) para ganar robustez a ruido real. **Se descartó** porque
XFUND tiene licencia CC BY-NC-SA 4.0 (no comercial) — ver
`ARQUITECTURA_Y_LICENCIAS.md`. El script se conserva por transparencia pero no
se usa en el pipeline por defecto.

---

## 4. Comando de entrenamiento

### 4.1 Validar el dataset antes de entrenar (siempre primero)

```powershell
python main.py -c configs\latin_rec_finetune.yaml -o Global.mode=check_dataset
```

Debe terminar con `Check dataset passed!`. Requiere `matplotlib` instalado
(ver problema #1 en la sección 5).

### 4.2 Entrenar

```powershell
$env:DISABLE_MODEL_SOURCE_CHECK="True"
$env:PADDLE_PDX_PADDLEOCR_PATH="C:\Projects\OCR\NellOCR\.venv\Lib\site-packages\paddlex\repo_manager\repos\PaddleOCR"
$env:PYTHONUTF8="1"
python main.py -c configs\latin_rec_finetune.yaml -o Global.mode=train
```

`main.py` es un wrapper mínimo de 5 líneas sobre `paddlex.engine.Engine` — el
patrón estándar de PaddleX (lee la config, decide el modo, delega en el
trainer correspondiente).

### 4.3 Configuración (`configs/latin_rec_finetune.yaml`)

```yaml
Global:
  model: latin_PP-OCRv5_mobile_rec
  dataset_dir: "C:/Projects/OCR/NellOCR/dataset"
  device: gpu:0

Train:
  epochs_iters: 20
  batch_size: 8
  learning_rate: 0.00015   # ver sección 6 — por qué este valor
  pretrain_weight_path: https://paddle-model-ecology.bj.bcebos.com/.../latin_PP-OCRv5_mobile_rec_pretrained.pdparams
  eval_interval: 1
  save_interval: 1
```

El entrenamiento genera, en cada época, un checkpoint en
`output/iter_epoch_N/` y actualiza `output/best_accuracy/` solo si la métrica
de esa época supera a la anterior — así que al final siempre tienes la mejor
versión vista durante todo el entrenamiento, no necesariamente la última.

---

## 5. Problemas reales encontrados y cómo se resolvieron

Documentados en orden cronológico, porque cada uno bloqueaba al siguiente:

| # | Síntoma | Causa | Solución |
|---|---|---|---|
| 1 | `DependencyError: deep_analyse is not ready` en `check_dataset` | Falta `matplotlib` | `pip install matplotlib` |
| 2 | `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` | `matplotlib` ≥3.10 eliminó ese método, usado internamente por PaddleX | `pip install "matplotlib<3.10"` |
| 3 | `UnsupportedParamError: 'latin_PP-OCRv5_mobile_rec' is not a registered model name` en modo `train` | Falta el plugin de entrenamiento de PaddleOCR | `python -m paddlex --install PaddleOCR -y` |
| 4 | `FileNotFoundError` al instalar el plugin (`repo_manager\repos` no existe) | Bug del instalador: no crea el directorio padre si no existe | Crear manualmente `mkdir .venv\...\repo_manager\repos` antes de instalar |
| 5 | `FileNotFoundError: ...PaddleOCR_api\configs\latin_PP-OCRv5_mobile_rec.yaml` | Bug de empaquetado: PaddleX 3.3.13 no incluye esa carpeta de configs nativas en el wheel de pip | Copiar manualmente el `.yml` nativo desde el repo clonado: `repo_manager/repos/PaddleOCR/configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml` → `paddlex/repo_apis/PaddleOCR_api/configs/latin_PP-OCRv5_mobile_rec.yaml` |
| 6 | `UnicodeDecodeError: 'charmap' codec can't decode byte...` a mitad de la descarga del checkpoint preentrenado | PaddleX decodifica la salida del subproceso de entrenamiento con la codificación ANSI de Windows (cp1252), pero la barra de progreso (`tqdm`) emite UTF-8 | `$env:PYTHONUTF8="1"` antes de entrenar |
| 8 | `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe0` al lanzar el entrenamiento desde Git Bash | Un subproceso de Windows imprime un mensaje en español (`INFORMACIÓN: no se pudo encontrar ningún archivo...`) con la página de códigos de la consola, y PaddleX lo decodifica como UTF-8 | Lanzar desde PowerShell con la consola en UTF-8: `chcp 65001; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8` además de `PYTHONUTF8=1` |
| 7 | `ValueError: all input arrays must have the same shape` al evaluar | **Bug real de PaddleX**: fuerza el mismo `batch_size` en entrenamiento y evaluación, pero el preprocesamiento de evaluación (`resize_norm_img_chinese`) ajusta el ancho de cada imagen según su propio aspect ratio — con líneas de texto de longitud muy variable, el `collate_fn` no puede apilar tensores de ancho distinto | Parche de una línea en `paddlex/repo_apis/PaddleOCR_api/text_rec/config.py`, método `update_batch_size`: quitar la línea que fuerza `Eval.loader.batch_size_per_card` al mismo valor que train (el yaml nativo ya pedía `batch_size_per_card: 1` para eval, con razón) |

> Los problemas #1, #2, #6 y #7 son bugs/incompatibilidades reales del
> ecosistema PaddleX en Windows al momento de este proyecto (noviembre 2026),
> no errores de configuración propios. Si una versión futura de PaddleX los
> corrige, estos pasos dejarán de ser necesarios.

---

## 6. Historial de intentos de entrenamiento (por qué el learning rate importa)

| Intento | Dataset | LR | Resultado |
|---|---|---|---|
| 1 | Sintético puro (322 líneas) | 0.001 (default) | acc 0.89 en época 2, luego **diverge completamente** (acc→0, loss→50+) en las épocas siguientes |
| 2 | Sintético + XFUND-ES (2.938 líneas) | 0.00015 | acc 0.862 en época 16, **estable, sin divergencia** durante las 20 épocas |
| 3 | + lote técnico (refuerzo `\|`/SQL/dígitos) (3.346 líneas) | 0.00015 | acc 0.861 en época 2, corrige el bug del `\|` y la mayoría de siglas técnicas |
| 4 (comercial) | Solo sintético propio, sin XFUND (730 líneas) | 0.00015 | acc 0.929 en época 4 — mejor resultado, dataset más chico y enfocado converge mejor |
| 5 | Base + técnico + multi-área (1.302 train / 228 val) | 0.00015 | acc 0.907 en época 4, norm_edit_dis 0.9964; sin mejora en las épocas 5–20 |
| 6 (**actual**) | Igual que el 5, regenerado con datos de contacto 100% ficticios | 0.00015 | acc **0.877** en época 4, norm_edit_dis 0.9957; sin mejora en las épocas 5–20 |

> Los intentos 5 y 6 no son comparables en sentido estricto: el conjunto de
> validación cambió (la línea de contacto ficticia reemplazó a una real). La
> diferencia de ~3 puntos equivale a unas 7 de las 228 líneas de validación y
> está dentro de la variación normal entre entrenamientos; la distancia de
> edición casi no cambió. En ambos, la mejor época es la 4 y las 16 siguientes
> no mejoran, lo que sugiere que bastarían menos épocas. El intento 6 es el que
> se publica porque su dataset no contiene datos personales.

**Lección clave:** el `learning_rate` por defecto del yaml oficial (0.001) es
demasiado alto para un dataset de fine-tuning pequeño (cientos/miles de
líneas, no millones) — el modelo sobreajusta y diverge en pocas épocas.
Bajarlo a **0.00015** (con el mismo warmup de 5 épocas y schedule Cosine)
fue lo que permitió que el entrenamiento convergiera de forma estable en
todos los intentos posteriores.

---

## 7. Exportar y probar el modelo resultante

El entrenamiento ya exporta automáticamente un modelo de inferencia en
`output/best_accuracy/inference/` (formato `inference.json` +
`inference.pdiparams` + `inference.yml`, ~7.9 MB). Para probarlo:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",       # modelo oficial, sin fine-tuning
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec", # nombre base (para que cargue el dict/arquitectura correctos)
    text_recognition_model_dir="output/best_accuracy/inference",  # pesos fine-tuneados
    device="gpu:0",
)
resultados = ocr.predict("imagen.png")
```

**Importante:** hay que pasar `text_recognition_model_name` aunque también se
pase `text_recognition_model_dir` — si se omite, PaddleOCR asume el modelo
"server" por defecto y falla con `AssertionError: Model name mismatch`.

---

## 8. Checklist para repetir el entrenamiento desde cero

1. `python -m venv .venv` + instalar dependencias (sección 1.2)
2. `python -m paddlex --install PaddleOCR -y` (plugin de entrenamiento, sección 1.4)
3. Si aparece el problema #5: copiar el `.yml` nativo manualmente (sección 5)
4. Crear `dataset/images/train` y `dataset/images/val`, y ejecutar `generar_dataset.py`, `generar_dataset_tecnico.py` y `generar_dataset_multiarea.py` (en ese orden)
5. Extraer `dict.txt` desde el modelo oficial (sección 2)
6. `python main.py -c configs\latin_rec_finetune.yaml -o Global.mode=check_dataset`
7. Con las 3 variables de entorno activas (desde PowerShell con la consola en UTF-8, problema #8): `python main.py -c configs\latin_rec_finetune.yaml -o Global.mode=train`
8. Verificar `output/best_accuracy/inference/` y probar con un script como `prueba_modelo_entrenado.py`
