"""Conversión de archivos subidos (imagen, PDF o Word) a imágenes limpias para el OCR.

El modelo solo lee imágenes. Las hojas de vida escaneadas llegan como:
  - imagen suelta (PNG, JPG, TIFF, BMP, WEBP),
  - PDF (cada página es una imagen escaneada, o una página con texto digital),
  - Word .docx con la página escaneada pegada como imagen.

`cargar_paginas` detecta el tipo por el CONTENIDO del archivo (no por la
extensión ni por el Content-Type que declare el cliente) y devuelve una lista
de imágenes PIL RGB, ya limpias: fondo transparente aplanado sobre blanco,
orientación EXIF corregida, inclinación del escáner corregida y contraste
recuperado si la hoja está desteñida.

Dependencias (todas de licencia permisiva, aptas para uso comercial):
  pypdfium2 (Apache-2.0 / BSD-3, motor PDFium) · Pillow (HPND) · OpenCV (Apache-2.0)
  · NumPy (BSD). El .docx se lee con la biblioteca estándar (zipfile).
No se usa PyMuPDF a propósito: su licencia es AGPL.

Límites: no se aceptan .doc antiguos (guardar como .docx), PDFs con contraseña,
ni más de MAX_PAGINAS páginas. No se corrige una página rotada 90° o 180°.
"""

import io
import re
import zipfile
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_BYTES = 25 * 1024 * 1024      # tamaño máximo del archivo subido
MAX_PAGINAS = 10                  # una hoja de vida rara vez pasa de 3-4 páginas
DPI_PDF = 200                     # resolución al rasterizar un PDF
LADO_MAX_PX = 4000                # tope del lado mayor (memoria y tiempo del OCR)
LADO_MIN_IMAGEN_DOCX = 300        # imágenes más pequeñas en un .docx se toman por logos

# Inclinación: se prueban ángulos en ±RANGO_GRADOS y solo se corrige si mejora claro.
RANGO_GRADOS = 6.0
PASO_GRADOS = 0.25
ANGULO_MINIMO = 0.4
MEJORA_MINIMA = 1.05

Image.MAX_IMAGE_PIXELS = 120_000_000   # evita "bombas de descompresión"


class ArchivoNoSoportado(ValueError):
    """El archivo no es una imagen, un PDF ni un .docx utilizable."""


# ---------------------------------------------------------------------------
# Detección de tipo y extracción
# ---------------------------------------------------------------------------
def detectar_tipo(contenido: bytes) -> str:
    """'pdf', 'docx', 'doc' (Word antiguo), 'imagen' o 'desconocido'."""
    if contenido[:5] == b"%PDF-":
        return "pdf"
    if contenido[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(contenido)) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        return "desconocido"
    if contenido[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "doc"
    try:
        Image.open(io.BytesIO(contenido)).verify()
        return "imagen"
    except Exception:
        return "desconocido"


def _paginas_de_pdf(contenido: bytes):
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(contenido)
    except pdfium.PdfiumError as e:
        raise ArchivoNoSoportado(
            "No se pudo abrir el PDF (¿está dañado o protegido con contraseña?)"
        ) from e
    try:
        if len(pdf) == 0:
            raise ArchivoNoSoportado("El PDF no tiene páginas")
        if len(pdf) > MAX_PAGINAS:
            raise ArchivoNoSoportado(
                f"El PDF tiene {len(pdf)} páginas; el máximo es {MAX_PAGINAS}")
        paginas = []
        for pagina in pdf:
            ancho, alto = pagina.get_size()
            escala = min(DPI_PDF / 72, LADO_MAX_PX / max(ancho, alto))
            paginas.append(pagina.render(scale=escala).to_pil())
        return paginas
    finally:
        pdf.close()


_NS_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_RE_EMBED = re.compile(rb'r:(?:embed|id)="(rId\d+)"')


def _paginas_de_docx(contenido: bytes):
    """Imágenes de un .docx en el orden en que aparecen en el documento."""
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        destinos = {r.get("Id"): r.get("Target") for r in rels.iter(f"{_NS_REL}Relationship")
                    if r.get("Type", "").endswith("/image")}
        orden = []
        for rid in _RE_EMBED.findall(z.read("word/document.xml")):
            rid = rid.decode()
            if rid in destinos and rid not in orden:
                orden.append(rid)

        imagenes = []
        for rid in orden:
            ruta = destinos[rid].lstrip("/")
            ruta = ruta if ruta.startswith("word/") else f"word/{ruta}"
            try:
                imagenes.append(Image.open(io.BytesIO(z.read(ruta))))
                imagenes[-1].load()
            except Exception:
                continue  # formato que Pillow no lee (p. ej. EMF/WMF): se omite

    if not imagenes:
        raise ArchivoNoSoportado(
            "El Word no contiene imágenes legibles (¿es un documento de texto? "
            "Este endpoint espera una hoja de vida escaneada)")
    grandes = [im for im in imagenes if min(im.size) >= LADO_MIN_IMAGEN_DOCX]
    imagenes = grandes or imagenes          # si todas son pequeñas, se conservan
    if len(imagenes) > MAX_PAGINAS:
        raise ArchivoNoSoportado(
            f"El Word tiene {len(imagenes)} imágenes; el máximo es {MAX_PAGINAS}")
    return imagenes


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def _aplanar_a_rgb(imagen: Image.Image) -> Image.Image:
    """RGB sobre fondo blanco (un PNG con transparencia saldría con fondo negro)."""
    imagen = ImageOps.exif_transpose(imagen)
    if imagen.mode in ("RGBA", "LA") or (imagen.mode == "P" and "transparency" in imagen.info):
        rgba = imagen.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(fondo, rgba).convert("RGB")
    return imagen.convert("RGB")


def _limitar_tamano(imagen: Image.Image) -> Image.Image:
    mayor = max(imagen.size)
    if mayor <= LADO_MAX_PX:
        return imagen
    k = LADO_MAX_PX / mayor
    return imagen.resize((round(imagen.width * k), round(imagen.height * k)), Image.LANCZOS)


def _angulo_de_inclinacion(imagen: Image.Image) -> float:
    """Ángulo (grados, antihorario) que hay que girar la imagen para enderezar el texto.

    Perfil de proyección: con el texto horizontal, las sumas de píxeles oscuros por
    fila alternan entre picos (renglones) y valles (interlineado), y su varianza es
    máxima. Se prueba una rejilla de ángulos sobre una versión reducida.
    """
    gris = np.array(imagen.convert("L"))
    alto, ancho = gris.shape
    k = 800 / max(alto, ancho)
    if k < 1:
        gris = cv2.resize(gris, (round(ancho * k), round(alto * k)), interpolation=cv2.INTER_AREA)
        alto, ancho = gris.shape
    _, bn = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if bn.mean() > 127:       # fondo oscuro (tinta clara): se invierte para contar el texto
        bn = 255 - bn
    centro = (ancho / 2, alto / 2)

    def puntaje(grados):
        m = cv2.getRotationMatrix2D(centro, grados, 1.0)
        rot = cv2.warpAffine(bn, m, (ancho, alto), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1, dtype=np.float64)))

    base = puntaje(0.0)
    if base == 0:
        return 0.0
    mejor, mejor_puntaje = 0.0, base
    for grados in np.arange(-RANGO_GRADOS, RANGO_GRADOS + PASO_GRADOS, PASO_GRADOS):
        p = puntaje(float(grados))
        if p > mejor_puntaje:
            mejor, mejor_puntaje = float(grados), p
    if abs(mejor) < ANGULO_MINIMO or mejor_puntaje < base * MEJORA_MINIMA:
        return 0.0
    return mejor


def limpiar(imagen: Image.Image) -> Image.Image:
    """Deja la imagen lista para el OCR (ver la descripción del módulo)."""
    imagen = _limitar_tamano(_aplanar_a_rgb(imagen))

    gris = np.array(imagen.convert("L"))
    p1, p99 = np.percentile(gris, [1, 99])
    if p99 - p1 < 160:                      # hoja desteñida o con poco contraste
        imagen = ImageOps.autocontrast(imagen, cutoff=1)

    grados = _angulo_de_inclinacion(imagen)
    if grados:
        imagen = imagen.rotate(grados, resample=Image.BICUBIC, expand=True,
                               fillcolor=(255, 255, 255))
    return imagen


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def cargar_paginas(contenido: bytes):
    """Convierte el archivo subido en una lista de imágenes PIL RGB limpias.

    Lanza ArchivoNoSoportado (con un mensaje apto para el usuario) si no se puede.
    """
    if not contenido:
        raise ArchivoNoSoportado("El archivo está vacío")
    if len(contenido) > MAX_BYTES:
        raise ArchivoNoSoportado(
            f"El archivo supera el máximo de {MAX_BYTES // (1024 * 1024)} MB")

    tipo = detectar_tipo(contenido)
    if tipo == "pdf":
        crudas = _paginas_de_pdf(contenido)
    elif tipo == "docx":
        crudas = _paginas_de_docx(contenido)
    elif tipo == "doc":
        raise ArchivoNoSoportado(
            "El formato .doc (Word antiguo) no es compatible: guárdalo como .docx o PDF")
    elif tipo == "imagen":
        crudas = [Image.open(io.BytesIO(contenido))]
    else:
        raise ArchivoNoSoportado(
            "Formato no soportado: se aceptan imágenes (PNG, JPG, TIFF, BMP, WEBP), PDF y .docx")
    return [limpiar(c) for c in crudas]
