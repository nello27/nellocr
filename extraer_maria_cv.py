from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="output/best_accuracy/inference",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

resultados = ocr.predict("imagenes_crudas/Maria_cv.png")

for resultado in resultados:
    resultado.print()
    resultado.save_to_json("resultados_maria_cv")

print("\nExtracción de Maria_cv.png finalizada.")
