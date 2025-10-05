import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from matplotlib.patches import Patch

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

    # Normalizar fechas de consulta
    fechas_consulta = sorted(pd.to_datetime(f, errors="coerce") for f in fechas_consulta)
    fechas_consulta = [f for f in fechas_consulta if not pd.isna(f)]
    if not fechas_consulta:
        print("❌ No se ingresaron fechas válidas.")
        return

    # Última fecha registrada
    fechas_registradas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna()
    if not fechas_registradas.empty:
        fecha_min_reg = fechas_registradas.min()
        fecha_max_reg = fechas_registradas.max()
    else:
        fecha_min_reg = fecha_max_reg = None

    # Definir rango de días a mostrar (~1 mes)
    dias_mostrar = 30
    fecha_ref = fechas_consulta[len(fechas_consulta)//2]  # fecha central si hay varias
    fecha_inicio = fecha_ref - timedelta(days=dias_mostrar // 2)
    fecha_fin = fecha_ref + timedelta(days=dias_mostrar // 2)

    rango = pd.date_range(fecha_inicio, fecha_fin, freq="D")

    # Colores por fase
    colores_fases = {
        "Menstrual": "lightcoral",
        "Folicular": "gold",
        "Ovulación": "limegreen",
        "Lútea": "skyblue"
    }

    # Determinar fase de cada día
    fases = []
    for i in range(len(rango)):
        if fechas_registradas.empty:
            d = i
        else:
            d = (rango[i] - fecha_min_reg).days % promedio
        if d <= 4:
            fases.append("Menstrual")
        elif d <= 13:
            fases.append("Folicular")
        elif 14 <= d <= 15:
            fases.append("Ovulación")
        else:
            fases.append("Lútea")

    # Gráfico
    #plt.figure(num="Linea de Tiempo de Período")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.canvas.manager.set_window_title("Línea de Tiempo de Período")   #nombre de la ventana que mostrará el gráfico

    # Dibujar bloques de color por fase
    inicio_fase = rango[0]
    fase_actual = fases[0]
    for i in range(1, len(rango)):
        if fases[i] != fase_actual or i == len(rango) - 1:
            fin_fase = rango[i] if fases[i] != fase_actual else rango[i] + timedelta(days=1)
            # Determinar si el bloque es estimación (fuera de registros)
            estimacion = False
            if fecha_min_reg and fecha_max_reg:
                if fin_fase < fecha_min_reg or inicio_fase > fecha_max_reg:
                    estimacion = True

            color = colores_fases[fase_actual]
            alpha = 0.4 if estimacion else 0.8
            hatch = '///' if estimacion else None

            ax.axvspan(inicio_fase, fin_fase, color=color, alpha=alpha, hatch=hatch)

            # Escribir nombre de fase en el bloque
            centro = inicio_fase + (fin_fase - inicio_fase) / 2
            plt.text(centro, 0.5, fase_actual, ha="center", va="center", fontsize=9, color="black")

            inicio_fase = rango[i]
            fase_actual = fases[i]

    # Líneas verticales para fechas consultadas
    for f in fechas_consulta:
        plt.axvline(f, color="black", linestyle="--", linewidth=1)
        '''plt.text(f, 1.05, f"{f.date()}", rotation=90, ha="center", va="bottom", fontsize=8)

        if f >= rango.min() and f <= rango.max():
            idx = (f - rango.min()).days
            fase = fases[idx]
            plt.text(f, 1.1, fase, rotation=90, ha="center", va="bottom", fontsize=8, color="black")'''

    # Ajustes visuales
    ax.set_xlim(rango.min(), rango.max())
    ax.set_xticks(rango)
    ax.set_xticklabels([d.strftime("%b-%d") for d in rango], rotation=45)
    ax.set_yticks([])
    plt.title(f"Fases del ciclo — Paciente {codigo}")

    # Leyenda explicativa
    leyenda = [Line2D([0], [0], color='black', linestyle='--', lw=1, label=f'Fecha consultada {f.date()}')]
    # Añadir bloques de fases a la leyenda
    '''for fase, color in colores_fases.items():
        leyenda.append(Line2D([0], [0], color=color, lw=6, label=fase))'''
    leyenda.append(Patch(facecolor='gray', edgecolor='black', alpha=0.4, hatch='///', label='Estimación'))
    ax.legend(handles=leyenda, loc='upper right', title="Simbología:")


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
            print(' >> Cierre ventana de grafico para continuar...')
            graficar_fases(reportes, codigo, fechas)

        elif opcion == "0":
            print("👋 Saliendo del sistema...")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    menu()
