"""Genera un lote adicional de líneas sintéticas enfocado en los puntos débiles
detectados al probar el modelo con CVs reales:
  - separador "|" en encabezados de contacto
  - siglas técnicas en mayúsculas (SQL, API, PHP, NET, etc.)
  - dígitos (sobre todo "0") en fechas y cifras
  - espaciado correcto entre palabras/puntuación
  - palabras con tilde mezcladas con términos técnicos en inglés

Se añade al dataset existente (dataset/train.txt y dataset/val.txt), igual que
hicimos con XFUND-ES.
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

random.seed(7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\consola.ttf",
]
FONT_PATHS = [p for p in FONT_PATHS if os.path.exists(p)]

# Encabezados de contacto con separador "|" (el punto más débil detectado)
CONTACTO = [
    "Bogotá D.C. | 310 123 4567 | nombre.apellido@correo.com | linkedin.com/in/nombre-apellido | github.com/usuario123",
    "Medellín, Colombia | 300 555 1234 | contacto@correo.com | linkedin.com/in/usuario | github.com/usuario",
    "Ingeniero de Sistemas | Desarrollo Backend | Automatización e Integración",
    "Analista de Datos | Business Intelligence | Machine Learning",
    "Cali, Colombia | +57 312 456 7890 | nombre.apellido@empresa.com | github.com/dev",
    "Desarrollador Full Stack | React | Node.js | PostgreSQL",
    "Soporte Técnico | Help Desk | Gestión de Incidentes",
    "Arquitecto de Software | Microservicios | Cloud Computing",
]

# Siglas y términos técnicos en mayúsculas (otro punto débil)
TECNICO = [
    "Desarrollo y lenguajes: PHP, Python, JavaScript, C#",
    "Backend y frameworks: PHPRunner, Laravel, Django, .NET, APIs REST",
    "Bases de datos: SQL Server, T-SQL, MySQL, PostgreSQL, Oracle",
    "Cloud y herramientas: Azure DevOps, Azure Functions, AWS, GCP, Docker",
    "IA y datos: Machine Learning, scikit-learn, Gemini API, NLP, Pandas, NumPy",
    "Automatización: n8n, Power Automate, Zapier, integración de APIs",
    "Control de versiones: Git, GitHub, GitLab, Bitbucket",
    "Seguridad: migración de MD5 y SHA1 a Bcrypt",
    "Frontend: Angular, React, Vue.js, TypeScript, HTML5, CSS3",
    "Metodologías: SCRUM, Kanban, DevOps, CI/CD",
    "Procesamiento de más de 400.000 registros históricos",
    "Optimización de procedimientos almacenados SQL Server",
    "Certificación AWS Cloud Practitioner, 2024",
    "API REST desarrollada con Node.js y Express",
    "Base de datos PostgreSQL con más de 2.000.000 de filas",
]

# Frases completas (bullets de experiencia), mezclando tildes + inglés técnico
EXPERIENCIA = [
    "Desarrollo, mantenimiento y refactorización de módulos con PHP y SQL Server.",
    "Optimización de consultas y procedimientos almacenados y vistas para auditoría.",
    "Implementación de mejoras en mecanismos de protección de credenciales.",
    "Desarrollo de un modelo de clasificación con Python y scikit-learn.",
    "Participación en revisión y comprensión de código en proyectos Angular y .NET.",
    "Diseño e implementación de flujos de automatización con n8n y Power Automate.",
    "Integración de servicios de Inteligencia Artificial como Gemini API.",
    "Elaboración de diagramas, documentación funcional y técnica.",
    "Atención y resolución de incidentes sobre aplicaciones misionales críticas.",
    "Construcción de soluciones tecnológicas orientadas al análisis de problemas.",
    "Extracción de metadatos judiciales y generación de reportes.",
    "Gestión de trabajo y requerimientos mediante Azure DevOps.",
    "Exposición práctica a servicios de Azure Functions y Azure Table Storage.",
    "Experiencia profesional en desarrollo de software backend desde 2020.",
    "Trabajó desde 2023 hasta 2026 en proyectos de misión crítica.",
]

# Fechas y rangos (refuerzo de dígitos, sobre todo el "0")
FECHAS = [
    "Agosto - noviembre 2026",
    "Junio 2020 - mayo 2026",
    "Enero 2019 - diciembre 2020",
    "2010 - 2015",
    "Desde el año 2000 hasta 2010",
    "Periodo: 01/01/2020 al 31/12/2026",
    "10.000 horas de experiencia acumulada",
    "100% de cumplimiento en entregas",
    "Versión 2.0.0 del sistema",
    "Código postal 110010, Bogotá",
]

ALL_LINES = CONTACTO + TECNICO + EXPERIENCIA + FECHAS


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
            font_size = random.randint(28, 42)
            img = render_line(text, font_path, font_size)
            samples.append((text, img))
    return samples


def append_split(samples, split_name, start_idx):
    split_dir = os.path.join(IMAGES_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    label_path = os.path.join(DATASET_DIR, f"{split_name}.txt")
    lines_out = []
    for i, (text, img) in enumerate(samples):
        idx = start_idx + i
        file_name = f"tecnico_{split_name}_{idx:04d}.png"
        img.save(os.path.join(split_dir, file_name))
        lines_out.append(f"images/{split_name}/{file_name}\t{text}")
    with open(label_path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines_out) + "\n")
    return len(lines_out)


def main():
    all_samples = build_samples(ALL_LINES, n_variants=10)
    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * 0.15))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    n_train = append_split(train_samples, "train", 0)
    n_val_saved = append_split(val_samples, "val", 0)

    print(f"Añadidas {n_train} líneas técnicas a train.txt")
    print(f"Añadidas {n_val_saved} líneas técnicas a val.txt")


if __name__ == "__main__":
    main()
