import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

PACIENTES_FILE = "pacientes.csv"
REPORTES_FILE = "reportes.csv"


# ==============================================================
# UTILIDADES BÁSICAS
# ==============================================================

def cargar_datos():
    try:
        pacientes = pd.read_csv(PACIENTES_FILE, dtype={"codigo": str, "nombre": str})
    except FileNotFoundError:
        pacientes = pd.DataFrame(columns=["codigo", "nombre"])

    try:
        reportes = pd.read_csv(
            REPORTES_FILE,
            dtype={"codigo": str},
            parse_dates=["fecha_periodo"],
            dayfirst=False
        )
    except FileNotFoundError:
        reportes = pd.DataFrame(columns=["codigo", "fecha_periodo", "codigo"])
    
    # Asegurar tipos y normalizar
    if "codigo" in pacientes.columns:
        pacientes["codigo"] = pacientes["codigo"].astype(str)
    if "codigo" in reportes.columns:
        reportes["codigo"] = reportes["codigo"].astype(str)
    if "fecha_periodo" in reportes.columns:
        reportes["fecha_periodo"] = pd.to_datetime(reportes["fecha_periodo"], errors="coerce")

    return pacientes, reportes


def guardar_datos(pacientes, reportes):
    pacientes.to_csv(PACIENTES_FILE, index=False)
    reportes.to_csv(REPORTES_FILE, index=False)


def registrar_paciente(pacientes, codigo, nombre):
    if codigo in pacientes["codigo"].values:
        print(f"⚠️ El código {codigo} ya existe.")
    else:
        pacientes.loc[len(pacientes)] = [codigo, nombre]
        print(f"✅ Paciente {nombre} registrada.")
    return pacientes


def registrar_periodo(reportes, pacientes, codigo, fecha):
    if str(codigo) not in pacientes["codigo"].astype(str).values:
        print("❌ Código no encontrado.")
        return reportes
    # Crear DataFrame nuevo con tipos definidos
    nueva_fila = pd.DataFrame.from_records(
        [{"codigo": codigo, "fecha_periodo": pd.to_datetime(fecha)}],
        columns=["codigo", "fecha_periodo"]
    )

    # Asegurar tipos antes de concatenar
    if reportes.empty:
        reportes = nueva_fila.copy()
    else:
        # Convertir columna fecha_periodo a datetime si aún no lo es
        if not np.issubdtype(reportes["fecha_periodo"].dtype, np.datetime64):
            reportes["fecha_periodo"] = pd.to_datetime(reportes["fecha_periodo"], errors="coerce")

        reportes = pd.concat([reportes, nueva_fila], ignore_index=True)
    print(f"🩸 Periodo registrado para #{codigo} en fecha: {fecha}.")
    return reportes


# ==============================================================
# PREDICCIÓN Y ESTADÍSTICA
# ==============================================================

def calcular_promedio_ciclo(reportes, codigo):
    """
    Calcula promedio y desviación del ciclo para 'codigo'.
    Usa fechas únicas y ordenadas para evitar difs cero por duplicados.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        return None, None

    # Normalizar fechas, ordenar y quitar duplicados
    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna().sort_values().drop_duplicates()
    if len(fechas) < 2:
        return None, None

    difs = fechas.diff().dt.days.dropna()
    if len(difs) == 0:
        return None, None

    promedio = difs.mean()
    desviacion = difs.std()

    # Evitar NaN → reemplazar por 0
    if pd.isna(promedio):
        promedio = 28.0  # Valor por defecto típico
    if pd.isna(desviacion):
        desviacion = 0.0

    return int(round(promedio)), int(round(desviacion))


def predecir_siguiente_periodo(reportes, codigo):
    _, reportes = cargar_datos()

    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        print("❌ No hay registros para esa paciente.")
        return None

    promedio, desviacion = calcular_promedio_ciclo(reportes, codigo)
    if promedio is None:
        print("🔹 Se necesitan al menos 2 fechas válidas para predecir el siguiente periodo.")
        return None

    # Usar la fecha más reciente (robusta ante desorden o duplicados)
    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna()
    if fechas.empty:
        print("❌ No hay fechas válidas.")
        return None
    
    ultima_fecha = fechas.max() #toma ultima fecha
    prediccion = ultima_fecha + timedelta(days=promedio)

    print(f"📅 Ciclo promedio: {promedio} ± {desviacion} días")
    print(f"🔮 Próximo periodo estimado: {prediccion.date()}")
    return prediccion


# ==============================================================
# CONSULTA Y GRAFICADO DE FASES
# ==============================================================

def graficar_fases(reportes, codigo, fechas_consulta):
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        print("❌ No hay datos para esa paciente.")
        return

    promedio, _ = calcular_promedio_ciclo(reportes, codigo)
    if promedio is None:
        promedio = 28  # valor por defecto

    fechas_consulta = sorted(fechas_consulta)
    # convertir a datetimes
    fechas_consulta = [pd.to_datetime(f, errors="coerce") for f in fechas_consulta]
    fechas_consulta = [f for f in fechas_consulta if not pd.isna(f)]
    if not fechas_consulta:
        print("❌ No se ingresaron fechas válidas.")
        return

    # Fecha más reciente registrada (robusta)
    fechas_registradas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna()
    ultima_fecha_reg = fechas_registradas.max() if not fechas_registradas.empty else None

    if len(fechas_consulta) == 1:
        fecha_ref = fechas_consulta[0]
        if ultima_fecha_reg is not None and fecha_ref >= ultima_fecha_reg:
            rango = pd.date_range(ultima_fecha_reg, ultima_fecha_reg + timedelta(days=promedio), freq="D")
            titulo = "Pronóstico futuro del ciclo"
        else:
            siguiente = fechas_registradas[fechas_registradas > fecha_ref].min() if not fechas_registradas.empty else None
            if pd.isna(siguiente) or siguiente is None:
                siguiente = fecha_ref + timedelta(days=promedio)
            rango = pd.date_range(fecha_ref, siguiente, freq="D")
            titulo = "Fases del ciclo (histórico)"
    else:
        if len(fechas_consulta) > 6:
            print("⚠️ Máximo 6 fechas permitidas. Tomando las más recientes.")
            fechas_consulta = fechas_consulta[-6:]
        inicio = fechas_consulta[0]
        fin = fechas_consulta[-1] + timedelta(days=promedio)
        rango = pd.date_range(inicio, fin, freq="D")
        titulo = "Fases del ciclo en rango múltiple"

    # Definimos fases
    dias = np.arange(len(rango))
    fases = []
    for d in dias:
        if d <= 4:
            fases.append("Menstrual")
        elif d <= 13:
            fases.append("Folicular")
        elif d == 14:
            fases.append("Ovulación")
        else:
            fases.append("Lútea")

    colores = {
        "Menstrual": "lightcoral",
        "Folicular": "gold",
        "Ovulación": "limegreen",
        "Lútea": "skyblue"
    }

    colores_fase = [colores[f] for f in fases]

    plt.figure(figsize=(10, 3))
    plt.bar(rango, np.ones(len(rango)), color=colores_fase, width=1.0)
    plt.title(f"{titulo} — Paciente {codigo}")
    plt.ylabel("Fases del ciclo")
    plt.xlabel("Fecha")
    plt.yticks([])
    plt.tight_layout()
    plt.show()


# ==============================================================
# INTERFAZ DE MENÚ
# ==============================================================

def menu():
    pacientes, reportes = cargar_datos()

    while True:
        print("\n=== SISTEMA DE SEGUIMIENTO MENSTRUAL ===")
        print("1️⃣ Registrar paciente")
        print("2️⃣ Registrar periodo")
        print("3️⃣ Ver predicción del próximo ciclo")
        print("4️⃣ Consultar y graficar por fecha(s)")
        print("0️⃣ Salir")
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            codigo = input("Código: ")
            nombre = input("Nombre: ")
            pacientes = registrar_paciente(pacientes, codigo, nombre)
            guardar_datos(pacientes, reportes)

        elif opcion == "2":
            codigo = input("Código de paciente: ")
            fecha = input("Fecha del periodo (YYYY-MM-DD): ")
            reportes = registrar_periodo(reportes, pacientes, codigo, fecha)
            guardar_datos(pacientes, reportes)

        elif opcion == "3":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            predecir_siguiente_periodo(reportes, codigo)

        elif opcion == "4":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            fechas_input = input("Ingrese una o varias fechas separadas por comas (YYYY-MM-DD): ")
            fechas = [pd.to_datetime(f.strip()) for f in fechas_input.split(",")]
            graficar_fases(reportes, codigo, fechas)

        elif opcion == "0":
            print("👋 Saliendo del sistema...")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    menu()
