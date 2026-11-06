"""Genera un dataset sintético de líneas de texto en español para fine-tuning
del modelo de reconocimiento latin_PP-OCRv5_mobile_rec.

Cada imagen contiene UNA sola línea de texto (recorte ajustado), como espera
el modelo de reconocimiento. Las transcripciones van en train.txt / val.txt
en formato "ruta_relativa\ttexto".
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
]
FONT_PATHS = [p for p in FONT_PATHS if os.path.exists(p)]

NOMBRES = ["Juan Pérez", "María López", "Carlos Gómez", "Ana Torres", "Luis Ramírez",
           "Sofía Herrera", "Diego Morales", "Valentina Castro", "Andrés Jiménez", "Camila Rojas"]
TITULOS = ["Ingeniero de Sistemas", "Ingeniera Industrial", "Licenciado en Administración",
           "Contador Público", "Diseñadora Gráfica", "Médico Cirujano", "Abogada",
           "Arquitecto", "Psicóloga Clínica", "Técnico en Electrónica"]
SECCIONES = ["HOJA DE VIDA", "PERFIL PROFESIONAL", "EXPERIENCIA LABORAL",
             "FORMACIÓN ACADÉMICA", "HABILIDADES", "REFERENCIAS", "DATOS PERSONALES",
             "CERTIFICACIONES", "IDIOMAS", "OBJETIVO PROFESIONAL"]
EXPERIENCIAS = [
    "Experiencia en Python, SQL y desarrollo web",
    "Manejo avanzado de Excel y bases de datos",
    "Más de 5 años liderando equipos de trabajo",
    "Conocimientos en redes y ciberseguridad",
    "Gestión de proyectos bajo metodologías ágiles",
    "Atención al cliente y ventas corporativas",
    "Desarrollo de aplicaciones móviles en Android",
    "Análisis de datos con Power BI y Tableau",
    "Mantenimiento de equipos industriales",
    "Elaboración de informes financieros mensuales",
]
CORREOS = [
    "Correo: prueba@ejemplo.com",
    "Email: contacto@empresa.com.co",
    "Correo electrónico: info@correo.net",
    "E-mail: soporte@dominio.org",
]
TELEFONOS = [
    "Teléfono: +57 300 123 4567",
    "Celular: (601) 555-7890",
    "Tel: 311-222-3344",
]
DIRECCIONES = [
    "Dirección: Calle 45 # 12-34, Bogotá",
    "Residencia: Av. Siempre Viva 742, Medellín",
    "Ubicación: Carrera 7 No. 10-20, Cali",
]
FECHAS = [
    "Fecha de nacimiento: 14/02/1995",
    "Nacido el 03 de agosto de 1990",
    "Disponibilidad inmediata desde enero de 2026",
]
EXTRA = [
    "Idiomas: Español (nativo), Inglés (avanzado)",
    "Estado civil: Soltero/a",
    "Nacionalidad: Colombiana",
    "LinkedIn: linkedin.com/in/usuario",
    "Referencia laboral: Empresa S.A.S. - Gerente",
    "Universidad Nacional, 2012 - 2017",
    "Certificado en Scrum Master (PSM I)",
    "Disponible para viajar y reubicarse",
    "¿Buscas un perfil con experiencia comprobada?",
    "Salario esperado: a convenir",
]

ALL_LINES = NOMBRES + TITULOS + SECCIONES + EXPERIENCIAS + CORREOS + TELEFONOS + DIRECCIONES + FECHAS + EXTRA


def render_line(text, font_path, font_size):
    font = ImageFont.truetype(font_path, font_size)
    tmp_img = Image.new("RGB", (10, 10), "white")
    tmp_draw = ImageDraw.Draw(tmp_img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    pad = 10
    width = (bbox[2] - bbox[0]) + pad * 2
    height = (bbox[3] - bbox[1]) + pad * 2

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, fill="black", font=font)
    return img


def build_samples(lines, n_variants):
    samples = []
    for text in lines:
        for _ in range(n_variants):
            font_path = random.choice(FONT_PATHS)
            font_size = random.randint(32, 44)
            img = render_line(text, font_path, font_size)
            samples.append((text, img))
    return samples


def save_split(samples, split_name):
    split_img_dir = os.path.join(IMAGES_DIR, split_name)
    label_lines = []
    for idx, (text, img) in enumerate(samples):
        file_name = f"{idx:04d}.png"
        img.save(os.path.join(split_img_dir, file_name))
        rel_path = f"images/{split_name}/{file_name}"
        label_lines.append(f"{rel_path}\t{text}")
    label_path = os.path.join(DATASET_DIR, f"{split_name}.txt")
    with open(label_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(label_lines) + "\n")
    return len(label_lines)


def main():
    all_samples = build_samples(ALL_LINES, n_variants=6)
    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * 0.15))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    n_train = save_split(train_samples, "train")
    n_val_saved = save_split(val_samples, "val")

    print(f"Dataset generado en: {DATASET_DIR}")
    print(f"  train: {n_train} imágenes")
    print(f"  val:   {n_val_saved} imágenes")


if __name__ == "__main__":
    main()
