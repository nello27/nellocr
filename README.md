# NellOCR — Fine-tuning de un modelo de OCR en español (100% local, open source)

Modelo de reconocimiento óptico de caracteres (OCR) especializado en
documentos en español —enfocado en hojas de vida—, entrenado con
**fine-tuning** sobre PaddleOCR (PP-OCRv5) usando GPU local (NVIDIA RTX
4060). Todo el pipeline, desde la generación de datos hasta el servicio vía
API, corre sin dependencias de servicios en la nube.

## Qué hace

Dado una imagen (captura, escaneo o foto de un documento), detecta cada línea
de texto y la transcribe, devolviendo texto + confianza + coordenadas por
línea — listo para integrarse en cualquier pipeline de extracción de datos.

```
Imagen → detección de líneas (PP-OCRv5_mobile_det)
       → reconocimiento de texto (modelo fine-tuneado propio)
       → JSON con texto, confianza y caja por línea
```

## Por qué existe este proyecto

Los modelos de OCR genéricos leen bien texto impreso limpio, pero fallan en
patrones específicos de un dominio (siglas técnicas en mayúsculas, separadores
como `|`, vocabulario propio de hojas de vida en español con tildes/ñ). Este
proyecto documenta el proceso completo de llevar un modelo preentrenado desde
"funciona en general" hasta "funciona bien para este caso de uso concreto",
incluyendo los errores reales encontrados en el camino y cómo se resolvieron.

## Arquitectura y stack

Ver [`ARQUITECTURA_Y_LICENCIAS.md`](./ARQUITECTURA_Y_LICENCIAS.md) para el
detalle completo (hiperparámetros, arquitectura del modelo, licencias de cada
dependencia y del dataset). Resumen rápido:

- **Framework**: PaddlePaddle + PaddleOCR + PaddleX (Apache 2.0)
- **Modelo base**: `latin_PP-OCRv5_mobile_rec` (SVTR_LCNet + CTC/NRTR heads)
- **Fine-tuning**: 20 épocas, Adam, LR 0.00015 con warmup, sobre datos
  sintéticos generados localmente (sin datasets de licencia restrictiva)
- **Servicio**: API REST con FastAPI + Uvicorn

## Estructura del repositorio

```
generar_dataset.py            # genera líneas sintéticas de hojas de vida en español
generar_dataset_tecnico.py    # refuerza separadores "|", siglas técnicas y dígitos
main.py                       # punto de entrada de entrenamiento (wrapper de PaddleX Engine)
configs/                      # configuración de fine-tuning (hiperparámetros)
api_ocr.py                    # API REST (FastAPI) que sirve el modelo entrenado
prueba_ocr.py                 # script de prueba con el modelo base (sin fine-tuning)
prueba_modelo_entrenado.py    # script de prueba con el modelo fine-tuneado
importar_xfund_es.py          # (opcional) importa datos reales de XFUND-ES — ver nota de licencia abajo
ARQUITECTURA_Y_LICENCIAS.md   # documentación técnica y de licenciamiento completa
```

> Los pesos del modelo entrenado, el dataset generado y las imágenes de
> prueba **no** están incluidos en este repositorio (ver `.gitignore`) —
> se reproducen ejecutando los scripts de generación y entrenamiento.

## Cómo reproducirlo

```bash
# 1. Entorno
python -m venv .venv
.venv\Scripts\pip install paddlepaddle-gpu paddleocr paddlex[ocr] fastapi uvicorn python-multipart

# 2. Generar dataset sintético
python generar_dataset.py
python generar_dataset_tecnico.py

# 3. Validar y entrenar
python main.py -c configs/latin_rec_finetune.yaml -o Global.mode=check_dataset
python main.py -c configs/latin_rec_finetune.yaml -o Global.mode=train

# 4. Probar el modelo resultante
python prueba_modelo_entrenado.py

# 5. Servir como API
uvicorn api_ocr:app --host 0.0.0.0 --port 8000
```

## Nota sobre licenciamiento de datos

Este repositorio, por defecto, **no depende de ningún dataset de licencia
restrictiva**: el dataset de entrenamiento se genera 100% de forma sintética
con `generar_dataset.py` + `generar_dataset_tecnico.py`. El script
`importar_xfund_es.py` es opcional y, si se usa, incorpora datos reales del
dataset público XFUND-ES (licencia **CC BY-NC-SA 4.0 — no comercial**); un
modelo entrenado incluyendo esos datos **no debe usarse comercialmente**. Ver
el detalle completo en `ARQUITECTURA_Y_LICENCIAS.md`.

## Licencia

Código bajo licencia [Apache 2.0](./LICENSE) — libre para uso, modificación y
distribución, incluido uso comercial.

---

Proyecto desarrollado por Nell Yesid Olaya Calderón como ejercicio práctico de
fine-tuning de modelos de OCR y despliegue local.
