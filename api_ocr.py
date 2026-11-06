"""API REST mínima para servir el modelo de OCR fine-tuneado.

Carga el pipeline de PaddleOCR UNA sola vez al arrancar (no por request) y
expone un endpoint POST /ocr que recibe una imagen y devuelve el texto
reconocido por línea, con su caja y su confianza.

Uso:
    uvicorn api_ocr:app --host 0.0.0.0 --port 8000

Luego:
    curl -X POST http://localhost:8000/ocr -F "file=@mi_imagen.png"
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

app = FastAPI(title="NellOCR API", version="1.0")

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="output/best_accuracy/inference",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def extraer_texto(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    contenido = await file.read()
    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    imagen_np = np.array(imagen)
    resultados = ocr.predict(imagen_np)

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

    return {"archivo": file.filename, "lineas": lineas}
