"""API REST mínima para servir el modelo de OCR fine-tuneado.

Carga el pipeline de PaddleOCR UNA sola vez al arrancar (no por request).

Endpoints:
  GET  /health      -> estado del servicio
  POST /ocr         -> texto por línea, con caja y confianza
  POST /ocr/texto   -> TODO el texto en un solo string, con anonimización
                       opcional de correos, teléfonos, direcciones y del
                       nombre del candidato (no de las empresas)

Uso:
    uvicorn api_ocr:app --host 0.0.0.0 --port 8000

Luego:
    curl -X POST http://localhost:8000/ocr -F "file=@mi_imagen.png"
    curl -X POST http://localhost:8000/ocr/texto -F "file=@mi_imagen.png"
    curl -X POST "http://localhost:8000/ocr/texto?anonimizar=false" -F "file=@mi_imagen.png"
"""

import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

from anonimizador import anonimizar_lineas

app = FastAPI(title="NellOCR API", version="1.1")

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="output/best_accuracy/inference",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


async def _ocr_por_lineas(file: UploadFile):
    """Valida la imagen subida y corre el OCR.

    Devuelve (lineas, alto_imagen): la lista de líneas con texto, confianza y
    caja, y el alto en píxeles de la imagen (para ubicar la parte superior).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    contenido = await file.read()
    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    resultados = ocr.predict(np.array(imagen))

    lineas = []
    for resultado in resultados:
        res = resultado.json["res"] if hasattr(resultado, "json") else resultado
        textos = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        cajas = res.get("rec_boxes", [])
        for texto, score, caja in zip(textos, scores, cajas):
            lineas.append({
                "texto": texto,
                "confianza": round(float(score), 4),
                "caja": [int(v) for v in caja],
            })
    return lineas, imagen.size[1]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def extraer_texto(file: UploadFile = File(...)):
    lineas, _ = await _ocr_por_lineas(file)
    return {"archivo": file.filename, "lineas": lineas}


@app.post("/ocr/texto")
async def extraer_texto_unico(
    file: UploadFile = File(...),
    anonimizar: bool = Query(
        True,
        description="Si es true (por defecto), reemplaza correos, teléfonos, "
                    "direcciones y el nombre del candidato por [EMAIL], "
                    "[TELEFONO], [DIRECCION] y [NOMBRE]. Los nombres de "
                    "empresas no se enmascaran.",
    ),
):
    lineas, alto_imagen = await _ocr_por_lineas(file)

    if not anonimizar:
        # Todo el texto de la imagen en una sola variable, una línea por renglón.
        texto_completo = "\n".join(linea["texto"] for linea in lineas)
        return {"archivo": file.filename, "anonimizado": False, "texto": texto_completo}

    # Todo el texto en una sola variable, ya anonimizado.
    texto_completo, conteo = anonimizar_lineas(lineas, alto_imagen)
    # Solo se devuelven cantidades por tipo, nunca los valores originales.
    return {
        "archivo": file.filename,
        "anonimizado": True,
        "texto": texto_completo,
        "datos_anonimizados": conteo,
    }
