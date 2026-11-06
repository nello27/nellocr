from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# Crear imagen de prueba
image = Image.new("RGB", (1400, 450), "white")
draw = ImageDraw.Draw(image)

font = ImageFont.truetype(
    r"C:\Windows\Fonts\arial.ttf", 48
)

draw.text((50, 60), "HOJA DE VIDA", fill="black", font=font)
draw.text((50, 150), "Ingeniero de Sistemas", fill="black", font=font)
draw.text((50, 240), "Experiencia en Python, SQL y desarrollo web", fill="black", font=font)
draw.text((50, 330), "Correo: prueba@ejemplo.com", fill="black", font=font)

image.save("prueba_ocr.png")

# Inicializar OCR
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Ejecutar reconocimiento
resultados = ocr.predict("prueba_ocr.png")

for resultado in resultados:
    resultado.print()
    resultado.save_to_json("resultados")

print("\nPrueba OCR finalizada.")
