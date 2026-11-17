# Procesamiento de documentos: de PDF / Word / imagen a imágenes limpias

El modelo solo lee imágenes. Las hojas de vida escaneadas llegan como imagen
suelta, PDF o Word (`.docx`) con la página pegada como imagen. El módulo
`documentos.py` convierte cualquiera de ellos en una lista de imágenes RGB
limpias, una por página, que `api_ocr.py` pasa al OCR.

```
archivo subido ──► detectar tipo ──► extraer imágenes ──► limpiar ──► OCR (una vez por página)
                   (por contenido)   PDF / DOCX / imagen   EXIF, fondo, contraste, inclinación
```

## 1. Detección del tipo (por contenido)

No se confía en la extensión ni en el `Content-Type` que declare el cliente.
Se leen los primeros bytes:

| Señal | Tipo |
|---|---|
| Empieza con `%PDF-` | PDF |
| Empieza con `PK` y el zip contiene `word/document.xml` | Word `.docx` |
| Empieza con `D0 CF 11 E0 A1 B1 1A E1` | Word `.doc` antiguo (se rechaza) |
| Pillow puede abrirlo y verificarlo | Imagen (PNG, JPG, TIFF, BMP, WEBP) |
| Cualquier otra cosa | No soportado |

## 2. Extracción de imágenes

- **PDF**: `pypdfium2` (motor PDFium) dibuja cada página a **200 DPI**, con tope
  de 4000 px en el lado mayor. Un escaneo es una foto pegada en la página, así
  que sale igual; una página con texto digital se dibuja como píxeles.
- **Word**: un `.docx` es un zip con XML. Las imágenes están en `word/media/`
  pero sin orden; el orden real se obtiene leyendo `word/document.xml`
  (referencias `r:embed="rIdN"`) y resolviendo cada `rIdN` en
  `word/_rels/document.xml.rels`. Se descartan las imágenes de menos de 300 px
  (logos); si todas son pequeñas, se conservan. Formatos que Pillow no lee
  (EMF/WMF) se omiten.
- **Imagen**: se usa tal cual.

## 3. Limpieza de cada imagen

| Paso | Por qué |
|---|---|
| Aplanar transparencia sobre fondo blanco | Un PNG transparente saldría con fondo negro |
| Corregir orientación EXIF | Las fotos de celular traen la rotación en metadatos |
| Recuperar contraste (si el rango entre los percentiles 1 y 99 es menor de 160) | Hojas desteñidas |
| Enderezar la hoja | El escáner suele dejarla inclinada unos grados |

**Enderezado (perfil de proyección).** Se reduce la imagen (lado mayor 800 px),
se binariza con Otsu y se gira virtualmente entre −6° y +6° en pasos de 0,25°.
En cada ángulo se suman los píxeles de texto por fila y se mide la **varianza**
de esas sumas: con el texto horizontal, los renglones y el interlineado se
distinguen al máximo. Se aplica el ángulo de mayor varianza solo si es de al
menos 0,4° y mejora la varianza en 5 % o más; la imagen real se gira con
interpolación bicúbica y fondo blanco. En una prueba con una hoja girada 3° se
estimó −3,25° y el resultado del OCR fue idéntico al de la hoja derecha.

## 4. Varias páginas

`/ocr` y `/ocr/texto` corren el OCR una vez por página.

- `/ocr` devuelve todas las líneas con el campo `pagina` (las cajas son relativas
  a su página) y el total en `paginas`.
- `/ocr/texto` separa las páginas con una línea en blanco. El **nombre del
  candidato se busca en la primera página** y sus palabras se enmascaran en
  todas; en las páginas siguientes no se busca nombre, para que un título grande
  no se confunda con él.

## 5. Límites y errores

| Condición | Respuesta (HTTP 400) |
|---|---|
| Archivo vacío o de más de 25 MB | Mensaje con el máximo |
| Más de 10 páginas (PDF) o 10 imágenes (Word) | Mensaje con el máximo |
| PDF dañado o con contraseña | «No se pudo abrir el PDF…» |
| Word sin imágenes legibles (documento de texto digital) | «El Word no contiene imágenes legibles…» |
| `.doc` antiguo | «Guárdalo como .docx o PDF» |
| Formato desconocido | Lista de formatos aceptados |

No se corrige una página rotada 90° o 180°. Un escaneo con compresión JPEG
fuerte pierde detalle antes de llegar a la API, y el OCR puede fallar en
algún dato (con calidad 70, `Felipe_Cv` pasó de 3 a 2 correos detectados); la
conversión no puede recuperar eso. Se probó con plantillas ficticias, no con
escaneos reales de terceros.

## 6. Licencias y pruebas

| Biblioteca | Uso | Licencia |
|---|---|---|
| pypdfium2 (PDFium) | Rasterizar PDF | Apache 2.0 / BSD-3-Clause |
| Pillow | Abrir y transformar imágenes | MIT-CMU (HPND) |
| OpenCV | Estimar la inclinación | Apache 2.0 |
| NumPy | Cálculo | BSD-3-Clause |

El `.docx` se lee con la biblioteca estándar (`zipfile`, `xml`). Se evitó
PyMuPDF a propósito: su licencia es AGPL y el proyecto busca ser de uso
comercial libre.

Pruebas (datos ficticios, sin el modelo): `python tests/test_documentos.py`
(14 pruebas: detección de tipo, transparencia, contraste, enderezado, PDF de
una y varias páginas, Word con logos, y los casos de error).
