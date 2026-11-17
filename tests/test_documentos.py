"""Pruebas de la conversión de PDF / Word / imagen a imágenes limpias (datos ficticios).

    python tests/test_documentos.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import documentos  # noqa: E402
from documentos import ArchivoNoSoportado, cargar_paginas, detectar_tipo  # noqa: E402


def _fuente(tam):
    for nombre in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def _hoja(ancho=900, alto=1200):
    """Una página de texto ficticio: renglones negros sobre blanco."""
    img = Image.new("RGB", (ancho, alto), "white")
    d = ImageDraw.Draw(img)
    fuente = _fuente(26)
    for i in range(18):
        d.text((60, 60 + i * 58), f"Experiencia laboral en la empresa Acme numero {i} de la lista", font=fuente, fill="black")
    return img


def _bytes(img, formato, **kw):
    b = io.BytesIO()
    img.save(b, formato, **kw)
    return b.getvalue()


def _docx(imagenes):
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    for img in imagenes:
        doc.add_picture(io.BytesIO(_bytes(img, "PNG")), width=Inches(5))
    b = io.BytesIO()
    doc.save(b)
    return b.getvalue()


def _falla(contenido, fragmento):
    try:
        cargar_paginas(contenido)
    except ArchivoNoSoportado as e:
        assert fragmento in str(e), str(e)
        return
    raise AssertionError("debía fallar")


# --- Detección de tipo -----------------------------------------------------
def test_detecta_el_tipo_por_el_contenido():
    hoja = _hoja()
    assert detectar_tipo(_bytes(hoja, "PNG")) == "imagen"
    assert detectar_tipo(_bytes(hoja, "JPEG")) == "imagen"
    assert detectar_tipo(_bytes(hoja, "PDF")) == "pdf"
    assert detectar_tipo(_docx([hoja])) == "docx"
    assert detectar_tipo(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"0" * 50) == "doc"
    assert detectar_tipo(b"hola esto no es un archivo") == "desconocido"


# --- Imágenes --------------------------------------------------------------
def test_imagen_limpia_se_devuelve_sin_cambios_de_tamano():
    hoja = _hoja()
    (salida,) = cargar_paginas(_bytes(hoja, "PNG"))
    assert salida.mode == "RGB" and salida.size == hoja.size


def test_png_con_transparencia_queda_sobre_fondo_blanco():
    rgba = Image.new("RGBA", (400, 300), (0, 0, 0, 0))     # totalmente transparente
    ImageDraw.Draw(rgba).text((20, 20), "Hola", font=_fuente(30), fill=(0, 0, 0, 255))
    (salida,) = cargar_paginas(_bytes(rgba, "PNG"))
    assert salida.getpixel((380, 280)) == (255, 255, 255)


def test_imagen_desteñida_recupera_contraste():
    gris = Image.new("RGB", (900, 1200), (225, 225, 225))
    d = ImageDraw.Draw(gris)
    for i in range(18):
        d.text((60, 60 + i * 58), f"Texto muy palido numero {i} de la lista de ejemplo", font=_fuente(26), fill=(170, 170, 170))
    (salida,) = cargar_paginas(_bytes(gris, "PNG"))
    claros, oscuros = salida.convert("L").getextrema()
    assert oscuros - claros > 200


# --- Inclinación -----------------------------------------------------------
def test_endereza_una_hoja_escaneada_torcida():
    torcida = _hoja().rotate(3.0, resample=Image.BICUBIC, expand=True, fillcolor="white")
    assert abs(documentos._angulo_de_inclinacion(torcida)) >= 2.0
    (salida,) = cargar_paginas(_bytes(torcida, "PNG"))
    assert abs(documentos._angulo_de_inclinacion(salida)) <= 0.5


def test_no_gira_una_hoja_que_ya_esta_derecha():
    assert documentos._angulo_de_inclinacion(_hoja()) == 0.0


# --- PDF -------------------------------------------------------------------
def test_pdf_de_dos_paginas_da_dos_imagenes():
    pdf = _bytes(_hoja(), "PDF", save_all=True, append_images=[_hoja()], resolution=150)
    paginas = cargar_paginas(pdf)
    assert len(paginas) == 2
    assert all(p.mode == "RGB" and min(p.size) > 800 for p in paginas)


def test_pdf_danado_o_vacio_da_mensaje_claro():
    _falla(b"%PDF-1.4\nesto no es un pdf valido", "PDF")
    _falla(b"", "vacío")


def test_pdf_con_demasiadas_paginas_se_rechaza():
    pdf = _bytes(_hoja(300, 400), "PDF", save_all=True,
                 append_images=[_hoja(300, 400) for _ in range(documentos.MAX_PAGINAS)])
    _falla(pdf, "máximo")


# --- Word ------------------------------------------------------------------
def test_docx_con_una_hoja_escaneada():
    paginas = cargar_paginas(_docx([_hoja()]))
    assert len(paginas) == 1 and paginas[0].size[0] >= 800


def test_docx_ignora_logos_pequenos_y_conserva_el_orden():
    logo = Image.new("RGB", (120, 60), "blue")
    p1, p2 = _hoja(900, 1200), _hoja(800, 1000)
    paginas = cargar_paginas(_docx([logo, p1, p2]))
    assert [p.size for p in paginas] == [p1.size, p2.size]


def test_docx_sin_imagenes_se_rechaza():
    from docx import Document

    doc = Document()
    doc.add_paragraph("Solo texto")
    b = io.BytesIO()
    doc.save(b)
    _falla(b.getvalue(), "no contiene imágenes")


# --- Formatos no soportados ------------------------------------------------
def test_formatos_no_soportados():
    _falla(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"0" * 50, ".docx")
    _falla(b"hola esto no es un archivo", "Formato no soportado")
    _falla(b"PK\x03\x04 zip roto", "Formato no soportado")


def test_archivo_demasiado_grande(monkeypatch=None):
    original = documentos.MAX_BYTES
    documentos.MAX_BYTES = 100
    try:
        _falla(b"x" * 101, "máximo")
    finally:
        documentos.MAX_BYTES = original


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, funcion in pruebas:
        try:
            funcion()
            print(f"OK    {nombre}")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"FALLA {nombre}: {type(e).__name__}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas")
    sys.exit(1 if fallos else 0)
