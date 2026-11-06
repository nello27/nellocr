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

resultados = ocr.predict("imagenes_crudas/Cv_Nell_2.png")

for resultado in resultados:
    resultado.print()
    resultado.save_to_json("resultados_cv_nell_2")

print("\nExtracción de Cv_Nell_2.png finalizada.")
