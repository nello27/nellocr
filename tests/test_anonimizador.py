"""Pruebas del módulo de anonimización (datos 100% ficticios).

Se pueden correr con pytest o directamente:
    python tests/test_anonimizador.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anonimizador import anonimizar, anonimizar_lineas  # noqa: E402


def _a(texto):
    return anonimizar(texto)[0]


# --- Correos ---------------------------------------------------------------
def test_email_normal():
    assert _a("Escríbeme a nombre.apellido@correo.com hoy") == "Escríbeme a [EMAIL] hoy"


def test_email_con_errores_de_ocr():
    # espacio después de la @ y coma en lugar de punto
    assert _a("Correo: nombre@ correo.com") == "Correo: [EMAIL]"
    assert _a("Correo: prueba@ejemplo,com") == "Correo: [EMAIL]"


def test_email_no_se_traga_el_texto_siguiente():
    assert _a("a@b.com, luego sigue el texto") == "[EMAIL], luego sigue el texto"


# --- Teléfonos -------------------------------------------------------------
def test_telefonos_en_distintos_formatos():
    for tel in ["+57 300 123 4567", "(601) 555-7890", "314 382 6104",
                "311-222-3344", "(55) 1234 5678", "3101234567"]:
        assert _a(f"Tel: {tel}") == "Tel: [TELEFONO]", tel


def test_no_confunde_fechas_anios_ni_cifras_con_telefonos():
    casos = [
        "Junio 2020 - mayo 2026",
        "2026-2030",
        "Maestría en Software 1 2026-2030",  # el OCR leyó el "|" como "1"
        "Mayo de 2030 - Marzo de 2033",
        "Más de 400.000 registros",
        "Base con 2.000.000 de filas",
        "Desde el 01/01/2020 al 31/12/2021",
        "Fecha 01-01-2020",
        "Certificación ISO 9001",
        "Colegiado No. 123456",
        "Cumplimiento del 100%",
        "Versión 2.0.0 del sistema",
        "10.000 horas de experiencia",
        "Gestiono un equipo de 20 ingenieros",
    ]
    for caso in casos:
        assert _a(caso) == caso, caso


# --- Direcciones -----------------------------------------------------------
def test_direcciones_tipicas():
    assert _a("Calle 45 # 12-34, Bogotá") == "[DIRECCION]"
    assert _a("Carrera 7 No. 10-20") == "[DIRECCION]"
    assert _a("Av. Siempre Viva 742, Medellín") == "[DIRECCION]"
    assert _a("Avenida Caracas 45-10 sur") == "[DIRECCION]"


def test_direccion_con_etiqueta_conserva_la_etiqueta():
    assert _a("Dirección: Barrio Centro, casa azul") == "Dirección: [DIRECCION]"
    assert _a("Residencia: Vereda La Palma") == "Residencia: [DIRECCION]"


def test_direccion_partida_en_dos_lineas():
    texto = "Calle 123, Colonia\nCiudad, Estado, País\nExperiencia laboral"
    assert _a(texto) == "[DIRECCION]\n[DIRECCION]\nExperiencia laboral"


def test_no_enmascara_palabras_comunes_de_un_cv():
    for caso in ["Carrera profesional en ventas", "Diseño de interiores",
                 "Carrera profesional de 10 años en ventas",
                 "Calle de la innovación"]:
        assert _a(caso) == caso, caso


def test_direccion_respeta_el_separador_pipe():
    texto = "Bogotá D.C. | Calle 45 # 12-34 | nombre@correo.com | 314 382 6104"
    assert _a(texto) == "Bogotá D.C. | [DIRECCION] | [EMAIL] | [TELEFONO]"


# --- General ---------------------------------------------------------------
def test_conteo_por_tipo():
    texto = "a@b.com y c@d.org\nCel 314 382 6104\nCalle 1 # 2-3"
    _, conteo = anonimizar(texto)
    assert conteo == {"email": 2, "telefono": 1, "direccion": 1, "nombre": 0}


def test_texto_sin_datos_sensibles_no_cambia():
    texto = "Ingeniero de Sistemas con experiencia en Python y SQL."
    salida, conteo = anonimizar(texto)
    assert salida == texto
    assert conteo == {"email": 0, "telefono": 0, "direccion": 0, "nombre": 0}


# --- Nombres del candidato -------------------------------------------------
def _linea(texto, y1, y2, x1=100, x2=600):
    return {"texto": texto, "caja": [x1, y1, x2, y2]}


def test_nombre_por_tamano_de_letra_y_menciones_posteriores():
    lineas = [
        _linea("CARLA MENDOZA RUIZ", 40, 90),                      # la más grande
        _linea("Ingeniera de Software", 100, 128),
        _linea("Carla trabajó en Acme S.A.S. con Mendoza Ruiz", 400, 425),
    ]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == (
        "[NOMBRE]\n"
        "Ingeniera de Software\n"
        "[NOMBRE] trabajó en Acme S.A.S. con [NOMBRE]"   # la empresa se conserva
    )
    assert conteo["nombre"] == 3


def test_nombre_partido_en_dos_lineas():
    lineas = [
        _linea("MARÍA LAURA", 40, 77),
        _linea("UGARTE LARIOS", 85, 119),
        _linea("Experta en software", 200, 225),
    ]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == "[NOMBRE]\n[NOMBRE]\nExperta en software"
    assert conteo["nombre"] == 2


def test_no_toma_el_titulo_del_documento_por_el_nombre():
    lineas = [
        _linea("CURRÍCULUM VITAE", 20, 80),                        # más grande, pero es un título
        _linea("JUAN PEREZ GOMEZ", 100, 145),
        _linea("Grupo Éxito S.A.", 300, 320),
    ]
    texto, _ = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == "CURRÍCULUM VITAE\n[NOMBRE]\nGrupo Éxito S.A."


def test_las_empresas_no_se_enmascaran_si_no_hay_nombre():
    lineas = [_linea("Trabajó en Grupo Bolívar y Seguros Alfa", 400, 425)]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == "Trabajó en Grupo Bolívar y Seguros Alfa"
    assert conteo["nombre"] == 0


def test_nombre_fuera_de_la_parte_superior_no_se_toma_por_el_candidato():
    lineas = [_linea("CARLA MENDOZA RUIZ", 700, 750)]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == "CARLA MENDOZA RUIZ"
    assert conteo["nombre"] == 0


def test_nombre_con_etiqueta_y_menciones_posteriores():
    texto, conteo = anonimizar("Nombre: Pedro Ramírez Soto\nGracias, Pedro, por todo.")
    assert texto == "Nombre: [NOMBRE]\nGracias, [NOMBRE], por todo."
    assert conteo["nombre"] == 2


def test_nombre_ignora_tildes_y_mayusculas_perdidas_por_el_ocr():
    texto, _ = anonimizar("Referencia de MARIA. Otra: maría.", nombres=["María"])
    assert texto == "Referencia de [NOMBRE]. Otra: [NOMBRE]."


def test_nombre_con_particulas_cuenta_como_un_solo_reemplazo():
    texto, conteo = anonimizar("Hola Carla de la Torre", nombres=["Carla", "Torre"])
    assert texto == "Hola [NOMBRE]"
    assert conteo["nombre"] == 1


# --- Referencias laborales -------------------------------------------------
def test_referencias_en_barra_lateral_con_columna_vecina_intercalada():
    # Réplica del diseño de un CV de dos columnas: las referencias van en la barra
    # lateral (x 26-160) y el OCR las intercala con la columna principal (x 274+).
    lineas = [
        _linea("REFERENCIAS", 520, 540, 58, 158),
        _linea("Atención al Público", 600, 615, 274, 400),          # otra columna
        _linea("SANDRA HARO", 556, 572, 26, 116),
        _linea("Encargada de Personal", 575, 590, 26, 130),         # cargo: se conserva
        _linea("Mail:", 596, 610, 32, 70),
        _linea("hola@sitioincreible.com", 612, 626, 32, 133),
        _linea("Curso de Manejo de Redes", 744, 758, 274, 420),     # otra columna
        _linea("Telefono:", 630, 644, 32, 80),
        _linea("(55) 1234-5678", 646, 660, 32, 110),
        _linea("RICARDO SALINAS", 670, 686, 26, 140),               # segundo bloque
        _linea("Encargado de Ventas", 690, 705, 26, 130),
        _linea("Mail:", 710, 724, 32, 70),
        _linea("hola@sitioincreible.com", 726, 740, 32, 133),
    ]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=825)
    assert texto.split("\n") == [
        "REFERENCIAS", "Atención al Público", "[NOMBRE]", "Encargada de Personal",
        "Mail:", "[EMAIL]", "Curso de Manejo de Redes", "Telefono:", "[TELEFONO]",
        "[NOMBRE]", "Encargado de Ventas", "Mail:", "[EMAIL]",
    ]
    assert conteo["nombre_referencia"] == 2
    assert conteo["nombre"] == 0   # "nombre: 0" sigue significando "sin nombre del candidato"


def test_referencias_se_detienen_en_el_siguiente_encabezado():
    lineas = [
        _linea("REFERENCIAS", 600, 620, 40, 200),
        _linea("JUAN PEREZ", 640, 656, 40, 160),
        _linea("Tel: 300 111 2222", 660, 674, 40, 180),
        _linea("IDIOMAS", 700, 720, 40, 150),
        _linea("Ingles Avanzado", 730, 745, 40, 160),   # parece un nombre, pero es otra sección
    ]
    texto, _ = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto.split("\n") == [
        "REFERENCIAS", "[NOMBRE]", "Tel: [TELEFONO]", "IDIOMAS", "Ingles Avanzado"]


def test_referencia_conserva_cargo_y_empresa():
    lineas = [
        _linea("Referencias laborales", 600, 620, 40, 220),
        _linea("Laura Gómez", 640, 656, 40, 160),
        _linea("Gerente de Ventas, Acme S.A.S.", 660, 676, 40, 260),
        _linea("laura@acme.com", 680, 694, 40, 180),
    ]
    texto, _ = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto.split("\n") == [
        "Referencias laborales", "[NOMBRE]", "Gerente de Ventas, Acme S.A.S.", "[EMAIL]"]


def test_referencia_con_nombre_y_cargo_en_la_misma_linea():
    lineas = [
        _linea("REFERENCIAS", 600, 620, 40, 200),
        _linea("Carlos Ruiz - Gerente de Operaciones", 640, 656, 40, 340),
    ]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto.split("\n")[1] == "[NOMBRE] - Gerente de Operaciones"
    assert conteo["nombre_referencia"] == 1


def test_correo_sin_arroba_bajo_etiqueta_mail():
    # El OCR leyó la "@" como "e": "holaesitioincreible.com"
    lineas = [
        _linea("Mail:", 596, 610, 32, 70),
        _linea("Otra cosa", 600, 614, 274, 400),
        _linea("holaesitioincreible.com", 612, 626, 32, 133),
    ]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=825)
    assert texto.split("\n") == ["Mail:", "Otra cosa", "[EMAIL]"]
    assert conteo["email"] == 1


def test_correo_sin_arroba_en_la_misma_linea():
    assert _a("Correo: holaesitioincreible.com") == "Correo: [EMAIL]"


def test_dominio_suelto_sin_etiqueta_no_se_toca():
    lineas = [_linea("Portafolio", 600, 614), _linea("miportafolio.com", 616, 630)]
    texto, _ = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == "Portafolio\nmiportafolio.com"


def test_sin_encabezado_de_referencias_no_se_enmascaran_nombres_de_terceros():
    lineas = [
        _linea("Trabajé con SANDRA HARO en Acme", 600, 620),
        _linea("JUAN PEREZ", 640, 656),
    ]
    texto, conteo = anonimizar_lineas(lineas, alto_imagen=1000)
    assert texto == "Trabajé con SANDRA HARO en Acme\nJUAN PEREZ"
    assert conteo["nombre_referencia"] == 0


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, funcion in pruebas:
        try:
            funcion()
            print(f"OK    {nombre}")
        except AssertionError as e:
            fallos += 1
            print(f"FALLA {nombre}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas")
    sys.exit(1 if fallos else 0)
