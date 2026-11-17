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
anonimizador.py               # anonimización de datos personales (correo, teléfono, dirección, nombres)
documentos.py                 # convierte PDF / Word / imagen escaneada en imágenes limpias para el OCR
tests/                        # pruebas unitarias (datos ficticios)
prueba_ocr.py                 # script de prueba con el modelo base (sin fine-tuning)
prueba_modelo_entrenado.py    # script de prueba con el modelo fine-tuneado
importar_xfund_es.py          # (opcional) importa datos reales de XFUND-ES — ver nota de licencia abajo
ARQUITECTURA_Y_LICENCIAS.md   # documentación técnica y de licenciamiento completa
PROCESAMIENTO_DE_DOCUMENTOS.md  # cómo se convierten PDF / Word / escaneos en imágenes limpias
```

> Los pesos del modelo entrenado, el dataset generado y las imágenes de
> prueba **no** están incluidos en este repositorio (ver `.gitignore`) —
> se reproducen ejecutando los scripts de generación y entrenamiento.

## Cómo reproducirlo

```bash
# 1. Entorno
python -m venv .venv
.venv\Scripts\pip install paddlepaddle-gpu paddleocr paddlex[ocr] fastapi uvicorn python-multipart pypdfium2

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

## Endpoints de la API

Documentación interactiva en `http://localhost:8000/docs`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/ocr` | Devuelve cada línea reconocida con su caja, confianza y página |
| POST | `/ocr/texto` | Devuelve todo el texto en una sola variable, con anonimización opcional |

### `POST /ocr/texto`

Parámetros: `file` (imagen, PDF o Word `.docx`; multipart) y `anonimizar` (query, `true` por defecto).

```bash
curl -X POST "http://localhost:8000/ocr/texto?anonimizar=true" -F "file=@cv.png"
```

```json
{
  "archivo": "cv.png",
  "anonimizado": true,
  "texto": "[NOMBRE]\nIngeniera de Software\nTel: [TELEFONO]\n[EMAIL]\n...",
  "datos_anonimizados": {"email": 1, "telefono": 1, "direccion": 1, "nombre": 1, "nombre_referencia": 2}
}
```

Con `anonimizar=false` se devuelve el texto original y no aparece `datos_anonimizados`.

### Formatos de entrada

Ambos endpoints aceptan lo que llega de un escáner, y el tipo se detecta por el **contenido** del archivo, no por la extensión:

| Formato | Qué se hace |
|---|---|
| Imagen (PNG, JPG, TIFF, BMP, WEBP) | Se usa directamente |
| PDF | Cada página se convierte en imagen a 200 DPI (hasta 10 páginas) |
| Word `.docx` | Se extraen las imágenes del documento en orden (se ignoran logos menores de 300 px; hasta 10) |

Antes del OCR, cada imagen se limpia (`documentos.py`): se aplana la transparencia sobre fondo blanco, se corrige la orientación EXIF, se **endereza la hoja si el escáner la dejó inclinada** (hasta ±6°) y se recupera el contraste si está desteñida. En un archivo de varias páginas el nombre del candidato se busca en la primera y se enmascara en todas; la respuesta incluye `paginas` y el texto de cada página va separado por una línea en blanco.

Se rechazan con un mensaje claro (HTTP 400): `.doc` antiguo (guárdalo como `.docx`), PDF con contraseña o dañado, Word sin imágenes (un documento de texto digital, no escaneado), archivos de más de 25 MB y formatos desconocidos. No se corrige una página rotada 90° o 180°.

Detalle del proceso en [`PROCESAMIENTO_DE_DOCUMENTOS.md`](./PROCESAMIENTO_DE_DOCUMENTOS.md).

```bash
curl -X POST http://localhost:8000/ocr/texto -F "file=@cv_escaneado.pdf"
```

**Qué se enmascara** (`anonimizador.py`, basado en reglas, sin servicios externos):

| Token | Dato |
|---|---|
| `[EMAIL]` | Correos, incluso con errores de OCR (espacios, coma por punto, "@" perdida bajo una etiqueta "Mail:") |
| `[TELEFONO]` | Teléfonos en distintos formatos; no confunde fechas, años ni cifras |
| `[DIRECCION]` | Direcciones y líneas con etiqueta "Dirección:" / "Residencia:" |
| `[NOMBRE]` | Nombre del candidato (por tamaño de letra en la parte superior) y sus menciones posteriores, y nombres de personas citadas como referencias laborales |

Los nombres de empresas y los cargos **no** se enmascaran, por ser información pública.
La respuesta solo incluye conteos por tipo, nunca los valores originales.

**Límites:** no cubre otros nombres de terceros fuera de la sección de referencias,
números de documento, usuarios de redes sociales ni URLs. Es una ayuda, no una
garantía: revisa el resultado antes de compartirlo.

Pruebas (datos ficticios): `python tests/test_anonimizador.py` y `python tests/test_documentos.py`

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
