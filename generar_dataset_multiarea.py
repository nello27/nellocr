"""Lote de refuerzo para generalizar el modelo a hojas de vida de CUALQUIER
área profesional (no solo IT/tecnología), y para reforzar patrones de
caracteres que seguían fallando tras el reentrenamiento sin XFUND:
  - separador "|" (con más repeticiones y contextos)
  - confusión I / l / 1 (mayúscula i, ele minúscula, uno)
  - dígitos y porcentajes
  - viñetas tipo bullet ("•", "·")

Se añade (append) al dataset existente en dataset/train.txt y dataset/val.txt.
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

random.seed(99)

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
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]
FONT_PATHS = [p for p in FONT_PATHS if os.path.exists(p)]

MEDICINA = [
    "Médico Cirujano | Especialista en Cardiología",
    "Residente de Medicina Interna, Hospital Nacional",
    "Colegiado No. 123456, Colegio Médico de Colombia",
    "Experiencia en atención de pacientes críticos y urgencias",
    "Certificación en Soporte Vital Avanzado (ACLS)",
    "Diagnóstico y tratamiento de enfermedades crónicas",
    "Enfermera Jefe | Unidad de Cuidados Intensivos",
]

DERECHO = [
    "Abogada | Especialista en Derecho Laboral",
    "Tarjeta Profesional No. 987654 del Consejo Superior de la Judicatura",
    "Litigio civil, comercial y de familia",
    "Asesoría jurídica a empresas del sector financiero",
    "Maestría en Derecho Penal, Universidad Externado",
]

DOCENCIA = [
    "Docente de Matemáticas | Educación Media",
    "Licenciada en Pedagogía Infantil",
    "Diseño curricular y evaluación por competencias",
    "10 años de experiencia en educación básica y media",
    "Coordinadora Académica, Colegio San José",
]

VENTAS_COMERCIAL = [
    "Ejecutivo de Ventas | Gestión de Cuentas Clave",
    "Cumplimiento de metas comerciales superior al 120%",
    "Negociación con proveedores y clientes corporativos",
    "Representante Comercial Zona Norte",
    "Estrategias de fidelización y retención de clientes",
]

CONTADURIA_FINANZAS = [
    "Contador Público | Tarjeta Profesional 45678-T",
    "Elaboración de estados financieros y declaraciones tributarias",
    "Auditoría interna y control de costos",
    "Analista Financiero Sénior, Sector Bancario",
    "Conciliaciones bancarias y presupuesto anual",
]

CONSTRUCCION_INGENIERIA = [
    "Ingeniero Civil | Especialista en Estructuras",
    "Supervisión de obra y control de cronogramas",
    "Diseño de proyectos de infraestructura vial",
    "Matrícula Profesional COPNIA No. 112233",
    "Gerencia de proyectos de construcción residencial",
]

DISENO_MARKETING = [
    "Diseñadora Gráfica | Branding y Comunicación Visual",
    "Gestión de redes sociales y campañas digitales",
    "Manejo de Adobe Photoshop, Illustrator e InDesign",
    "Estrategia de marketing digital y posicionamiento SEO",
    "Diseño de identidad corporativa para startups",
]

ADMINISTRACION_RRHH = [
    "Administradora de Empresas | Gestión del Talento Humano",
    "Procesos de selección, contratación e inducción de personal",
    "Administración de nómina y seguridad social",
    "Clima organizacional y bienestar laboral",
    "Especialista en Gestión Humana, Universidad Javeriana",
]

# Refuerzo explícito de patrones de caracteres problemáticos
REFUERZO_PATRONES = [
    "Teléfono | Correo | LinkedIn | Instagram | Portafolio",
    "Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4",
    "III. Experiencia Laboral",
    "Inglés Intermedio | Italiano Básico | Francés Inicial",
    "ID: 1234567890 | NIT: 900.123.456-1",
    "Instituto | Industria | Innovación | Inversión",
    "100% | 95% | 80% | 50% | 15%",
    "01/01/2020 al 31/12/2021",
    "Logró incrementar las ventas en un 15%",
    "Implementó mejoras en los procesos internos",
    "Lideró un equipo interdisciplinario de 10 personas",
    "LinkedIn: linkedin.com/in/usuario-profesional",
    "Certificación ISO 9001 | Lean Six Sigma | PMI",
    "I. Perfil | II. Experiencia | III. Educación | IV. Idiomas",
]

ALL_LINES = (
    MEDICINA
    + DERECHO
    + DOCENCIA
    + VENTAS_COMERCIAL
    + CONTADURIA_FINANZAS
    + CONSTRUCCION_INGENIERIA
    + DISENO_MARKETING
    + ADMINISTRACION_RRHH
    + REFUERZO_PATRONES
)


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
            font_size = random.randint(26, 44)
            img = render_line(text, font_path, font_size)
            samples.append((text, img))
    return samples


def append_split(samples, split_name):
    split_dir = os.path.join(IMAGES_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    label_path = os.path.join(DATASET_DIR, f"{split_name}.txt")
    existing = len([f for f in os.listdir(split_dir) if f.startswith("multiarea_")])
    lines_out = []
    for i, (text, img) in enumerate(samples):
        idx = existing + i
        file_name = f"multiarea_{split_name}_{idx:04d}.png"
        img.save(os.path.join(split_dir, file_name))
        lines_out.append(f"images/{split_name}/{file_name}\t{text}")
    with open(label_path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines_out) + "\n")
    return len(lines_out)


def main():
    all_samples = build_samples(ALL_LINES, n_variants=12)
    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * 0.15))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    n_train = append_split(train_samples, "train")
    n_val_saved = append_split(val_samples, "val")

    print(f"Añadidas {n_train} líneas multi-área a train.txt")
    print(f"Añadidas {n_val_saved} líneas multi-área a val.txt")


if __name__ == "__main__":
    main()
