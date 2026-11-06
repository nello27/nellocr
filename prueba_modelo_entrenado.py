from paddleocr import PaddleOCR

# Igual que prueba_ocr.py, pero usando nuestro modelo de reconocimiento
# fine-tuneado (dataset sintético + XFUND-ES real) en vez del oficial.
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="output/best_accuracy/inference",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

resultados = ocr.predict("prueba_ocr.png")

for resultado in resultados:
    resultado.print()
    resultado.save_to_json("resultados_modelo_entrenado")

print("\nPrueba con modelo fine-tuneado finalizada.")
