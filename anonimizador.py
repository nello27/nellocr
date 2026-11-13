"""Anonimización de datos sensibles en texto extraído por OCR.

Detecta y reemplaza, mediante reglas (sin modelos ni dependencias externas):
  - correos electrónicos  -> [EMAIL]
  - números de teléfono    -> [TELEFONO]
  - direcciones postales   -> [DIRECCION]
  - nombre del candidato   -> [NOMBRE]   (las empresas NO se enmascaran)

Diseñado para el texto que sale del OCR, que NO es texto limpio: el modelo a
veces inserta espacios alrededor de la "@", cambia un punto por una coma
("ejemplo,com"), lee "|" como "1" o pierde las tildes. Los patrones toleran
esos errores comunes.

Criterio general: ante la duda se prefiere sobre-enmascarar (un falso
positivo solo tapa de más) antes que dejar pasar un dato personal.

Cómo se detecta el nombre del candidato (anonimizar_lineas):
  1. Por layout: en el 35% superior de la página, la línea con aspecto de
     nombre (2-6 palabras en total, capitalizadas, sin palabras típicas de
     CV como "Ingeniero" o "Perfil") y de letra más grande. Es lo habitual
     en un CV, y es lo que lo distingue del nombre de una empresa.
  2. Por etiqueta: "Nombre: ...", "Apellidos: ...", "Candidato: ...".
  3. Las palabras del nombre encontradas se enmascaran también en el resto del
     texto (ignorando mayúsculas y tildes).
  4. Referencias: bajo un encabezado "Referencias", en su misma columna (por las
     cajas), se enmascara el nombre de cada persona (la primera línea de cada
     bloque). El cargo y la empresa que siguen no se tocan.

Limitaciones conocidas (ver README):
  - De otras personas solo enmascara las citadas como referencia (bajo un
    encabezado "Referencias"); no detecta jefes u otros nombres sueltos en el
    texto, ni números de documento, usuarios de redes ni URLs.
  - Si el nombre no es el texto más grande de la parte superior y no tiene
    etiqueta, no se detecta. La respuesta informa cuántos nombres se
    enmascararon (0 = revisar).
  - Las direcciones se detectan por patrones típicos en español/Latinoamérica
    (Calle/Carrera/Avenida + número, o etiquetas "Dirección:", "Domicilio:").
"""

import re
import unicodedata

EMAIL = "[EMAIL]"
TELEFONO = "[TELEFONO]"
DIRECCION = "[DIRECCION]"
NOMBRE = "[NOMBRE]"

# Perezoso + lookahead: el "resto del campo" no consume los espacios que
# preceden a un separador "|" ni cruza de línea. Se usa \n|\Z (y no $) para que
# funcione igual sobre una línea suelta que sobre un texto de varias líneas.
_FIN_DE_CAMPO = r"(?=[ \t]*(?:\||\n|\Z))"
_LETRAS = "A-Za-zÁÉÍÓÚÜáéíóúüÑñ"

# --- Correos ---------------------------------------------------------------
# Tolera espacios alrededor de la "@" y la coma en lugar del punto del
# dominio (errores típicos del OCR), p. ej. "nombre@ ejemplo,com".
_RE_EMAIL = re.compile(r"[\w.+-]+[ \t]*@[ \t]*[\w-]+(?:[.,][\w-]+)+")

# --- Direcciones -----------------------------------------------------------
# Vías "numeradas": el número va justo después ("Carrera 7 No. 10-20"). Se exige
# eso porque "Carrera profesional de 10 años" es una frase común en un CV.
_VIAS_NUMERADAS = (
    r"(?:(?i:Carrera)|Cra|CRA|Cr|CR|Kr|KR|Cll?|CLL?|Diag|DIAG|Mz|MZ|(?i:Manzana))"
)
# Vías que suelen llevar nombre antes del número ("Av. Siempre Viva 742").
_VIAS_NOMBRADAS = (
    r"(?:(?i:Calle|Avenida|Diagonal|Transversal|Autopista|Carretera)|Av|AV|Avda)"
)
_RE_DIR_VIA = re.compile(
    rf"\b(?:{_VIAS_NUMERADAS}\b\.?[ \t]*\d"
    rf"|{_VIAS_NOMBRADAS}\b\.?[ \t]*(?:[{_LETRAS}]{{1,15}}[ \t]+){{0,3}}\d)"
    rf"[^|\n]*?{_FIN_DE_CAMPO}"
)
# "Dirección: ...", "Residencia: ...", "Domicilio: ..." -> se conserva la etiqueta.
_RE_DIR_ETIQUETA = re.compile(
    r"(?i)(\b(?:Direcci[oó]n(?:[ \t]+de[ \t]+residencia)?|Residencia|Domicilio)"
    rf"[ \t]*:[ \t]*)([^|\n]+?){_FIN_DE_CAMPO}"
)

# --- Teléfonos -------------------------------------------------------------
# Candidato: secuencia de dígitos con separadores típicos (espacio, guion,
# punto, paréntesis, "+" inicial). Luego se valida para descartar fechas,
# rangos de años y cifras con separador de miles.
_RE_TEL_CANDIDATO = re.compile(r"(?<![\w/.,])\+?\(?\d[\d ().\-]{5,20}\d(?![\w/])")
# El "1" opcional del inicio cubre un error típico del OCR: leer un separador
# "|" pegado al rango como un "1" ("Software 1 2026-2030").
_RE_RANGO_ANIOS = re.compile(r"(?:1[ \t]+)?(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}")
_RE_MILES = re.compile(r"\d{1,3}(?:\.\d{3})+")
_RE_FECHA = re.compile(r"\d{1,2}[-.]\d{1,2}[-.](?:19|20)?\d{2}")

# --- Nombres ---------------------------------------------------------------
_RE_NOMBRE_ETIQUETA = re.compile(
    r"(?i)(\b(?:Nombres?(?:[ \t]+(?:y[ \t]+apellidos?|completo|del[ \t]+candidat[oa]))?"
    r"|Apellidos?|Candidat[oa])[ \t]*:[ \t]*)([^|\n]+?)" + _FIN_DE_CAMPO
)
_RE_PALABRA = re.compile(rf"[{_LETRAS}][{_LETRAS}'’\-]*")
_RE_PALABRA_DE_NOMBRE = re.compile(rf"[{_LETRAS}][{_LETRAS}'’\-]*\.?")

# Partículas que pueden ir dentro de un nombre ("de la", "del", "y").
_PARTICULAS = {"de", "del", "la", "las", "los", "y", "e", "da", "do", "dos", "das", "di"}
# Títulos: pueden aparecer en la línea del nombre, pero no son parte de él.
_TITULOS = {"dr", "dra", "ing", "lic", "mg", "msc", "phd", "sr", "sra", "srta",
            "mba", "esp", "abg", "arq", "cpa"}
# Palabras típicas de un CV que NUNCA forman parte de un nombre: si una línea
# las contiene, no se considera línea de nombre (evita tomar "CURRÍCULUM VITAE"
# o "Ingeniero de Sistemas" por el nombre del candidato).
_PALABRAS_NO_NOMBRE = {
    "curriculum", "curriculo", "vitae", "cv", "hoja", "vida", "perfil",
    "profesional", "profesionales", "resumen", "experiencia", "laboral",
    "educacion", "formacion", "academica", "academico", "habilidades",
    "competencias", "contacto", "datos", "personales", "referencias", "idiomas",
    "objetivo", "ingeniero", "ingeniera", "ingenieria", "sistemas", "software",
    "desarrollador", "desarrolladora", "analista", "gerente", "director",
    "directora", "asistente", "tecnico", "tecnica", "abogado", "abogada",
    "medico", "medica", "doctor", "doctora", "licenciado", "licenciada",
    "contador", "contadora", "administrador", "administradora", "disenador",
    "disenadora", "docente", "profesor", "profesora", "arquitecto", "arquitecta",
    "consultor", "consultora", "coordinador", "coordinadora", "especialista",
    "experto", "experta", "senior", "junior", "estudiante", "egresado",
    "egresada", "universidad", "instituto", "colegio", "empresa", "trabajo",
    "proyecto", "proyectos", "cursos", "certificaciones", "logros", "intereses",
    "acerca", "presente", "actualidad", "telefono", "correo", "direccion",
    "ciudad", "pais", "portafolio", "linkedin", "github", "especializacion",
    "areas", "historial", "estudios", "conocimientos", "tecnologias",
}
_ZONA_SUPERIOR = 0.35   # fracción superior de la página donde se busca el nombre
_TOLERANCIA_ALTURA = 0.8  # una línea cuenta como "del tamaño del nombre" si
                          # mide al menos esto respecto a la más grande

_VARIANTES = {"a": "aáàäâ", "e": "eéèëê", "i": "iíìïî", "o": "oóòöô",
              "u": "uúùüû", "n": "nñ"}


def _sin_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )


def _norm(texto: str) -> str:
    return _sin_acentos(texto).lower()


# ---------------------------------------------------------------------------
# Teléfonos, direcciones
# ---------------------------------------------------------------------------
def _es_telefono(candidato: str) -> bool:
    digitos = re.sub(r"\D", "", candidato)
    if not 7 <= len(digitos) <= 15:
        return False
    texto = candidato.strip()
    for patron in (_RE_RANGO_ANIOS, _RE_MILES, _RE_FECHA):
        if patron.fullmatch(texto):
            return False
    return True


def _sub_telefonos(linea: str):
    cantidad = 0

    def reemplazo(m):
        nonlocal cantidad
        if _es_telefono(m.group(0)):
            cantidad += 1
            return TELEFONO
        return m.group(0)

    return _RE_TEL_CANDIDATO.sub(reemplazo, linea), cantidad


def _sub_direcciones(linea: str):
    linea, n_etiqueta = _RE_DIR_ETIQUETA.subn(lambda m: m.group(1) + DIRECCION, linea)
    linea, n_via = _RE_DIR_VIA.subn(DIRECCION, linea)
    return linea, n_etiqueta + n_via


def _parece_continuacion_de_direccion(linea: str) -> bool:
    """Segunda línea de una dirección partida, p. ej. "Ciudad, Estado, País"."""
    linea = linea.strip()
    return (
        0 < len(linea) <= 50
        and "," in linea
        and len(linea.split()) <= 6
        and not re.search(r"[\d@|]", linea)
    )


# ---------------------------------------------------------------------------
# Nombres
# ---------------------------------------------------------------------------
def _tokens_de_nombre(texto: str):
    """Palabras de un nombre que vale la pena enmascarar en el resto del texto."""
    vistos, tokens = set(), []
    for palabra in _RE_PALABRA.findall(texto):
        palabra = palabra.strip("-'’")
        clave = _norm(palabra)
        if (len(palabra) >= 3 and clave not in _PARTICULAS and clave not in _TITULOS
                and clave not in _PALABRAS_NO_NOMBRE and clave not in vistos):
            vistos.add(clave)
            tokens.append(palabra)
    return tokens


def _patron_token(token: str) -> str:
    """Regex de una palabra que acepta variantes con/sin tilde (el OCR las pierde)."""
    partes = []
    for c in token:
        base = _sin_acentos(c).lower()
        partes.append(f"[{_VARIANTES[base]}]" if base in _VARIANTES else re.escape(c))
    return "".join(partes)


def _compilar_nombres(tokens):
    """Una sola regex que agrupa palabras de nombre seguidas ("Carla de la Torre")
    en un único reemplazo."""
    if not tokens:
        return None
    t = "|".join(sorted({_patron_token(x) for x in tokens}, key=len, reverse=True))
    particulas = "|".join(sorted(_PARTICULAS))
    return re.compile(
        rf"(?<!\w)(?:{t})(?:[ \t]+(?:(?:{particulas})[ \t]+)*(?:{t}))*(?!\w)",
        re.IGNORECASE,
    )


def _parece_nombre(texto: str) -> bool:
    palabras = texto.split()
    if not 1 <= len(palabras) <= 5:
        return False
    for palabra in palabras:
        if not _RE_PALABRA_DE_NOMBRE.fullmatch(palabra):
            return False
        nucleo = palabra.strip(".")
        clave = _norm(nucleo)
        if clave in _PARTICULAS:
            continue
        if not nucleo[0].isupper() or clave in _PALABRAS_NO_NOMBRE:
            return False
    return True


def detectar_lineas_nombre(lineas, alto_imagen):
    """Índices de las líneas que contienen el nombre del candidato.

    `lineas` es una lista de dicts con "texto" y "caja" ([x1, y1, x2, y2]).
    Devuelve [] si no hay una línea claramente identificable como nombre.
    """
    if not lineas or not alto_imagen:
        return []
    candidatas = []
    for i, linea in enumerate(lineas):
        _, y1, _, y2 = linea["caja"]
        if y1 < alto_imagen * _ZONA_SUPERIOR and _parece_nombre(linea["texto"]):
            candidatas.append((i, y2 - y1))
    if not candidatas:
        return []
    mayor = max(alto for _, alto in candidatas)
    elegidas = [i for i, alto in candidatas if alto >= mayor * _TOLERANCIA_ALTURA]
    palabras = sum(len(lineas[i]["texto"].split()) for i in elegidas)
    return elegidas if 2 <= palabras <= 6 else []


# ---------------------------------------------------------------------------
# Referencias laborales / personales
# ---------------------------------------------------------------------------
# Cargos y marcas de empresa: una línea con estas palabras no se toma por un
# nombre (así se conservan "Encargada de Personal" o "Grupo Éxito").
_PALABRAS_NO_NOMBRE |= {
    "encargado", "encargada", "jefe", "jefa", "supervisor", "supervisora",
    "lider", "gerencia", "recursos", "humanos", "talento", "humano", "personal",
    "ventas", "mercadeo", "marketing", "operaciones", "produccion", "compras",
    "comercial", "financiero", "financiera", "contabilidad", "administrativo",
    "administrativa", "grupo", "fundacion", "banco", "hospital", "clinica",
    "servicios", "soluciones", "consultores", "asociados", "industrias",
    "laboratorios", "compania", "ltda", "sas", "inc", "corp",
}
_MAYUS = "A-ZÁÉÍÓÚÜÑ"
_RE_ENCABEZADO_REFERENCIAS = re.compile(
    r"^\W*(?:referencias?|references?)"
    r"(?:\s+(?:laborales?|personales?|profesionales?|comerciales?|familiares?)"
    r"(?:\s+y\s+(?:laborales?|personales?|profesionales?))?)?\W*$"
)
# Otros encabezados de sección: cierran el bloque de referencias.
_ENCABEZADOS_DE_SECCION = (
    "educacion", "experiencia", "cursos", "habilidades", "idiomas", "contacto",
    "perfil", "intereses", "logros", "certificaciones", "proyectos", "formacion",
    "estudios", "datos", "sobre mi", "acerca", "competencias", "areas",
    "historial", "conocimientos", "aptitudes", "objetivo", "hobbies",
    "publicaciones", "voluntariado",
)
_RE_ETIQUETA_CONTACTO = re.compile(
    r"^\W*(?:e-?mail|mail|correo|telefono|tel|celular|cel|movil|whatsapp|direccion|linkedin)\b"
)
# "Laura Gómez - Gerente de Ventas": solo el nombre del inicio, antes del separador.
_RE_NOMBRE_PREFIJO = re.compile(
    rf"^((?:[{_MAYUS}][{_LETRAS}'’\-]*\.?[ \t]+){{1,3}}[{_MAYUS}][{_LETRAS}'’\-]*)"
    r"[ \t]*(?=[-–—,|(:])"
)


def _es_encabezado_referencias(texto: str) -> bool:
    return bool(_RE_ENCABEZADO_REFERENCIAS.match(_norm(texto)))


def _es_encabezado_de_seccion(texto: str) -> bool:
    # Un encabezado es corto y no tiene ":" ni dígitos ni "@" (una etiqueta como
    # "Contacto: 300..." es un dato, no un encabezado).
    if ":" in texto or re.search(r"[\d@]", texto):
        return False
    clave = _norm(texto).strip(" .-–—•")
    palabras = clave.split()
    return 0 < len(palabras) <= 5 and any(clave.startswith(t) for t in _ENCABEZADOS_DE_SECCION)


def _es_linea_de_contacto(texto: str) -> bool:
    if _RE_ETIQUETA_CONTACTO.match(_norm(texto)) or _RE_EMAIL.search(texto):
        return True
    return any(_es_telefono(m.group(0)) for m in _RE_TEL_CANDIDATO.finditer(texto))


def _misma_columna(caja, caja_encabezado) -> bool:
    """True si la línea se superpone horizontalmente con el encabezado (con un
    margen proporcional a su ancho), es decir, está en la misma columna."""
    x1, _, x2, _ = caja
    hx1, _, hx2, _ = caja_encabezado
    margen = max(20, 0.4 * (hx2 - hx1))
    return min(x2, hx2 + margen) - max(x1, hx1 - margen) > 0


# --- Correos sin "@" (el OCR a veces lee la "@" como "e", "a" o "0") --------
# Sin la "@", la regla de correos no los ve. Se detectan por contexto: una línea
# con forma de dominio ("holaesitioincreible.com") justo debajo de una etiqueta
# "Mail:" / "Correo:", o en su misma línea.
_RE_ETIQUETA_CORREO = re.compile(r"^\W*(?:e-?mail|mail|correo(?:\s+electronico)?)\W*$")
_RE_DOMINIO_SUELTO = re.compile(r"[\w.+-]+(?:\.[A-Za-z]{2,6}){1,2}")
_RE_CORREO_SIN_ARROBA_EN_LINEA = re.compile(
    r"(?i)(\b(?:e-?mail|mail|correo(?:[ \t]+electr[oó]nico)?)[ \t]*:[ \t]*)"
    r"([\w.+-]+(?:\.[A-Za-z]{2,6}){1,2})(?![\w@])"
)


def detectar_correos_sin_arroba(lineas):
    """Índices de líneas que son un correo cuya "@" el OCR no leyó.

    Busca la línea más cercana por debajo de una etiqueta "Mail:" (en su misma
    columna, por las cajas: el OCR puede intercalar otras líneas entre las dos).
    """
    indices = set()
    for i, etiqueta in enumerate(lineas):
        if not _RE_ETIQUETA_CORREO.match(_norm(etiqueta["texto"])):
            continue
        _, y1, _, y2 = etiqueta["caja"]
        debajo = [
            j for j, linea in enumerate(lineas)
            if j != i and 0 < linea["caja"][1] - y1 <= 3 * max(y2 - y1, 1)
            and _misma_columna(linea["caja"], etiqueta["caja"])
        ]
        if not debajo:
            continue
        j = min(debajo, key=lambda k: lineas[k]["caja"][1])
        if _RE_DOMINIO_SUELTO.fullmatch(lineas[j]["texto"].strip()):
            indices.add(j)
    return indices


def _tramo_de_nombre(texto: str):
    """(inicio, fin) del nombre de una persona en la línea, o None."""
    if len(texto.split()) >= 2 and _parece_nombre(texto):
        return 0, len(texto)
    m = _RE_NOMBRE_PREFIJO.match(texto)
    if m and _parece_nombre(m.group(1)):
        return 0, m.end(1)
    return None


def detectar_referencias(lineas):
    """Nombres de las personas citadas como referencia.

    Busca un encabezado "Referencias" y, en SU MISMA COLUMNA (por las cajas; en
    CVs de dos columnas el orden de lectura del OCR mezcla las columnas), recorre
    las líneas siguientes hasta el próximo encabezado de sección. Cada
    referencia es un bloque: el nombre es la primera línea con aspecto de nombre
    y los datos de contacto (correo, teléfono) cierran el bloque. El cargo y la
    empresa que siguen al nombre no se tocan.

    Devuelve {índice_de_línea: (inicio, fin)} con el tramo de texto a enmascarar.
    """
    marcadas = {}
    for i, encabezado in enumerate(lineas):
        if not _es_encabezado_referencias(encabezado["texto"]):
            continue
        hy1 = encabezado["caja"][1]
        columna = sorted(
            (j for j, linea in enumerate(lineas)
             if j != i and linea["caja"][1] > hy1
             and _misma_columna(linea["caja"], encabezado["caja"])),
            key=lambda j: (lineas[j]["caja"][1], lineas[j]["caja"][0]),
        )
        esperando_nombre, vio_contacto = True, False
        for j in columna:
            texto = lineas[j]["texto"]
            if _es_encabezado_de_seccion(texto):
                break
            if _es_linea_de_contacto(texto):
                vio_contacto = True
                continue
            if vio_contacto:  # el contacto cerró el bloque anterior
                esperando_nombre, vio_contacto = True, False
            if not esperando_nombre:
                continue
            tramo = _tramo_de_nombre(texto)
            if tramo:
                marcadas[j] = tramo
                esperando_nombre = False
    return marcadas


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def anonimizar(texto: str, nombres=()):
    """Reemplaza correos, teléfonos, direcciones y nombres del texto.

    `nombres` son palabras de nombre ya conocidas (p. ej. detectadas por
    layout) que se enmascaran donde aparezcan. Además se detectan nombres por
    etiqueta ("Nombre: ...").

    Devuelve (texto_anonimizado, conteo), donde conteo es un diccionario con la
    cantidad de datos reemplazados por tipo. Nunca devuelve los valores
    originales.
    """
    conteo = {"email": 0, "telefono": 0, "direccion": 0, "nombre": 0}

    tokens = list(nombres)
    for m in _RE_NOMBRE_ETIQUETA.finditer(texto):
        tokens.extend(_tokens_de_nombre(m.group(2)))
    re_nombres = _compilar_nombres(tokens)

    salida = []
    linea_anterior_fue_direccion = False

    for linea in texto.split("\n"):
        nueva, n_email = _RE_EMAIL.subn(EMAIL, linea)
        nueva, n = _RE_CORREO_SIN_ARROBA_EN_LINEA.subn(lambda m: m.group(1) + EMAIL, nueva)
        n_email += n
        nueva, n_dir = _sub_direcciones(nueva)
        nueva, n_tel = _sub_telefonos(nueva)
        nueva, n_nom = _RE_NOMBRE_ETIQUETA.subn(lambda m: m.group(1) + NOMBRE, nueva)
        if re_nombres:
            nueva, n = re_nombres.subn(NOMBRE, nueva)
            n_nom += n

        es_continuacion = (
            n_dir == 0
            and n_email == 0
            and n_tel == 0
            and n_nom == 0
            and linea_anterior_fue_direccion
            and _parece_continuacion_de_direccion(linea)
        )
        if es_continuacion:
            nueva, n_dir = DIRECCION, 1
            linea_anterior_fue_direccion = False
        else:
            linea_anterior_fue_direccion = n_dir > 0

        conteo["email"] += n_email
        conteo["telefono"] += n_tel
        conteo["direccion"] += n_dir
        conteo["nombre"] += n_nom
        salida.append(nueva)

    return "\n".join(salida), conteo


def anonimizar_lineas(lineas, alto_imagen):
    """Anonimiza el resultado del OCR usando también la geometría de las líneas.

    `lineas`: lista de dicts con "texto" y "caja" ([x1, y1, x2, y2]).
    `alto_imagen`: alto en píxeles de la imagen original.
    Devuelve (texto_anonimizado, conteo), con todo el texto en un solo string.
    """
    indices = set(detectar_lineas_nombre(lineas, alto_imagen))
    referencias = {i: tramo for i, tramo in detectar_referencias(lineas).items()
                   if i not in indices}
    correos = detectar_correos_sin_arroba(lineas)
    textos, tokens = [], []
    for i, linea in enumerate(lineas):
        if i in correos:
            textos.append(EMAIL)
        elif i in indices:
            tokens.extend(_tokens_de_nombre(linea["texto"]))
            textos.append(NOMBRE)
        elif i in referencias:
            inicio, fin = referencias[i]
            textos.append(linea["texto"][:inicio] + NOMBRE + linea["texto"][fin:])
        else:
            textos.append(linea["texto"])

    texto, conteo = anonimizar("\n".join(textos), nombres=tokens)
    conteo["nombre"] += len(indices)
    conteo["email"] += len(correos)
    # Nombres de personas citadas como referencia, contados aparte para que
    # "nombre: 0" siga significando "no se encontró el nombre del candidato".
    conteo["nombre_referencia"] = len(referencias)
    return texto, conteo
