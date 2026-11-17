"""API REST mínima para servir el modelo de OCR fine-tuneado.

Carga el pipeline de PaddleOCR UNA sola vez al arrancar (no por request).

Endpoints:
  GET  /health      -> estado del servicio
  POST /ocr         -> texto por línea, con caja y confianza
                       (acepta imagen, PDF o Word .docx escaneado)
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

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from paddleocr import PaddleOCR
import numpy as np

from anonimizador import anonimizar_lineas, nombres_del_candidato
from documentos import ArchivoNoSoportado, cargar_paginas

app = FastAPI(title="NellOCR API", version="1.2")

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="output/best_accuracy/inference",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


async def _ocr_por_paginas(file: UploadFile):
    """Convierte el archivo subido (imagen, PDF o Word) a imágenes limpias y corre el OCR.

    Devuelve una lista con una entrada por página: (lineas, alto_imagen). Cada línea
    trae texto, confianza y caja; el alto sirve para ubicar la parte superior de la
    página (donde suele estar el nombre).
    """
    contenido = await file.read()
    try:
        paginas = cargar_paginas(contenido)
    except ArchivoNoSoportado as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer el archivo")

    salida = []
    for imagen in paginas:
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
        salida.append((lineas, imagen.size[1]))
    return salida


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def extraer_texto(file: UploadFile = File(...)):
    paginas = await _ocr_por_paginas(file)
    lineas = [dict(linea, pagina=n) for n, (ls, _) in enumerate(paginas, start=1) for linea in ls]
    return {"archivo": file.filename, "paginas": len(paginas), "lineas": lineas}


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
    paginas = await _ocr_por_paginas(file)

    if not anonimizar:
        # Todo el texto del archivo en una sola variable, una línea por renglón.
        texto_completo = "\n\n".join("\n".join(l["texto"] for l in ls) for ls, _ in paginas)
        return {"archivo": file.filename, "paginas": len(paginas),
                "anonimizado": False, "texto": texto_completo}

    # El nombre del candidato se busca en la primera página y se enmascara en todas.
    primera_lineas, primer_alto = paginas[0]
    nombres = nombres_del_candidato(primera_lineas, primer_alto)
    textos, conteo = [], {}
    for n, (lineas, alto) in enumerate(paginas):
        texto, c = anonimizar_lineas(lineas, alto, nombres=nombres, buscar_nombre=(n == 0))
        textos.append(texto)
        for clave, valor in c.items():
            conteo[clave] = conteo.get(clave, 0) + valor
    # Solo se devuelven cantidades por tipo, nunca los valores originales.
    return {
        "archivo": file.filename,
        "paginas": len(paginas),
        "anonimizado": True,
        "texto": "\n\n".join(textos),
        "datos_anonimizados": conteo,
    }
