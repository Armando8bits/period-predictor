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
            dtype={"codigo": str, "duracion":int},
            parse_dates=["fecha_periodo"],
            dayfirst=False
        )
    except FileNotFoundError:
        reportes = pd.DataFrame(columns=["codigo", "fecha_periodo", "duracion"])
    
    # Asegurar tipos y normalizar
    if "codigo" in pacientes.columns:
        pacientes["codigo"] = pacientes["codigo"].astype(str)
    if "codigo" in reportes.columns:
        reportes["codigo"] = reportes["codigo"].astype(str)
    if "duracion" in reportes.columns:
        reportes["duracion"] = reportes["duracion"].astype(int)
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


def registrar_periodo(reportes, pacientes, codigo, fecha, duracion=5):
    if duracion is None or duracion < 1:
        duracion = 5

    if str(codigo) not in pacientes["codigo"].astype(str).values:
        print("❌ Código no encontrado.")
        return reportes
    # Crear DataFrame nuevo con tipos definidos
    nueva_fila = pd.DataFrame.from_records(
        [{"codigo": codigo, "fecha_periodo": pd.to_datetime(fecha), "duracion": duracion}],
        columns=["codigo", "fecha_periodo", "duracion"]
    )

    # Asegurar tipos antes de concatenar
    if reportes.empty:
        reportes = nueva_fila.copy()
    else:
        # Convertir columna fecha_periodo a datetime si aún no lo es
        if not np.issubdtype(reportes["fecha_periodo"].dtype, np.datetime64):
            reportes["fecha_periodo"] = pd.to_datetime(reportes["fecha_periodo"], errors="coerce")

        reportes = pd.concat([reportes, nueva_fila], ignore_index=True)
    print(f"🩸 Periodo registrado para #{codigo} en fecha: {fecha} (duración: {duracion} días).")
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

def calcular_fases_siguientes(reportes, codigo):
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        print("❌ No hay datos para esa paciente.")
        return None

    # Calcular promedio de ciclo
    promedio, _ = calcular_promedio_ciclo(reportes, codigo)
    if promedio is None:
        promedio = 28  # valor por defecto

    # Última fecha registrada
    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna()
    if fechas.empty:
        print("❌ No hay fechas válidas para esta paciente.")
        return None

    ultima_fecha = fechas.max()
    # Definir duración típica de cada fase
    duracion_menstrual = 5
    duracion_folicular = 9
    duracion_ovulacion = 2
    duracion_lutea = promedio - (duracion_menstrual + duracion_folicular + duracion_ovulacion)

    # Calcular fechas estimadas
    fases = {
        "Menstrual": (ultima_fecha, ultima_fecha + timedelta(days=duracion_menstrual - 1)),
        "Folicular": (ultima_fecha + timedelta(days=duracion_menstrual),
                      ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular - 1)),
        "Ovulación": (ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular),
                      ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular + duracion_ovulacion - 1)),
        "Lútea": (ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular + duracion_ovulacion),
                  ultima_fecha + timedelta(days=promedio - 1))
    }

    # Convertir a DataFrame
    df_fases = pd.DataFrame([
        {"fase": fase, "inicio": fechas[0], "fin": fechas[1]} for fase, fechas in fases.items()
    ])

    return df_fases


# ==============================================================
# CONSULTA Y GRAFICADO DE FASES
# ==============================================================

def calcular_fases_periodo(reportes, codigo):
    """
    Calcula información del ciclo menstrual basado en registros históricos.
    Devuelve: promedio del ciclo, próxima fecha estimada y las fases relevantes.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        return None, None, None, None

    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna().sort_values()
    if len(fechas) < 2:
        return None, None, fechas.iloc[-1], None

    # Diferencias entre periodos (ciclo promedio)
    difs = fechas.diff().dropna().dt.days
    promedio = int(difs.mean())
    desviacion = int(difs.std() if not np.isnan(difs.std()) else 0)
    ultima_fecha = fechas.max()
    siguiente_estimado = ultima_fecha + timedelta(days=promedio)

    # Fases estimadas según ciclo típico de 28 días
    fases = [
        ("Menstrual", 0, 4),
        ("Folicular", 5, 13),
        ("Ovulación", 14, 15),
        ("Lútea", 16, promedio)
    ]

    return promedio, desviacion, siguiente_estimado, fases

def graficar_fases(reportes, codigo, fechas_consulta, promedio=None):
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        print("❌ No hay datos para esa paciente.")
        return

    if promedio is None:
        promedio, _, _, _ = calcular_fases_periodo(reportes, codigo)
        if promedio is None:
            promedio = 28

    # Normalizar fechas de consulta
    fechas_consulta = sorted(pd.to_datetime(f, errors="coerce") for f in fechas_consulta)
    fechas_consulta = [f for f in fechas_consulta if not pd.isna(f)]
    if not fechas_consulta:
        print("❌ No se ingresaron fechas válidas.")
        return

    fechas_registradas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna()
    if not fechas_registradas.empty:
        fecha_min_reg = fechas_registradas.min()
        fecha_max_reg = fechas_registradas.max()
    else:
        fecha_min_reg = fecha_max_reg = None

    # --- Ajuste de rango ---
    dias_mostrar = 30
    if len(fechas_consulta) == 1:
        fecha_ref = fechas_consulta[0]
        fecha_inicio = fecha_ref - timedelta(days=dias_mostrar // 2)
        fecha_fin = fecha_ref + timedelta(days=dias_mostrar // 2)
    else:
        margen = 2
        fecha_inicio = min(fechas_consulta) - timedelta(days=margen)
        fecha_fin = max(fechas_consulta) + timedelta(days=margen)

    rango = pd.date_range(fecha_inicio, fecha_fin, freq="D")

    # --- Fases ---
    colores_fases = {
        "Menstrual": "lightcoral",
        "Folicular": "gold",
        "Ovulación": "limegreen",
        "Lútea": "skyblue"
    }

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

    # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.canvas.manager.set_window_title("Línea de Tiempo de Período")

    inicio_fase = rango[0]
    fase_actual = fases[0]
    for i in range(1, len(rango)):
        if fases[i] != fase_actual or i == len(rango) - 1:
            fin_fase = rango[i] if fases[i] != fase_actual else rango[i] + timedelta(days=1)
            estimacion = False
            if fecha_min_reg and fecha_max_reg:
                if fin_fase < fecha_min_reg or inicio_fase > fecha_max_reg:
                    estimacion = True

            color = colores_fases[fase_actual]
            alpha = 0.4 if estimacion else 0.8
            hatch = '///' if estimacion else None

            ax.axvspan(inicio_fase, fin_fase, color=color, alpha=alpha, hatch=hatch)
            centro = inicio_fase + (fin_fase - inicio_fase) / 2
            ax.text(centro, 0.5, fase_actual, ha="center", va="center", fontsize=9, color="black")
            inicio_fase = rango[i]
            fase_actual = fases[i]

    # --- Líneas de fechas consultadas ---
    for f in fechas_consulta:
        ax.axvline(f, color="black", linestyle="--", linewidth=1)

    # --- Títulos y leyenda ---
    ax.set_xlim(rango.min(), rango.max())
    ax.set_xticks(rango)
    ax.set_xticklabels([d.strftime("%b-%d") for d in rango], rotation=45)
    ax.set_yticks([])
    ax.set_title(f"Fases del ciclo — Paciente {codigo}", loc="left")
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(1.0, 1.02, f"Impreso el {fecha_hora}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='gray')

    leyenda = [
        Line2D([0], [0], color='black', linestyle='--', lw=1, label='Fecha consultada'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.4, hatch='///', label='Estimación')
    ]
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
        print("1️⃣  Registrar paciente")
        print("2️⃣  Registrar periodo")
        print("3️⃣  Ver predicción del próximo ciclo")
        print("4️⃣  Consultar y graficar por fecha(s)")
        print("0️⃣  Salir")
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            codigo = input("Código: ")
            nombre = input("Nombre: ")
            if codigo and nombre:
                pacientes = registrar_paciente(pacientes, codigo, nombre)
                guardar_datos(pacientes, reportes)
            else:
                print("❌ Ingrese valores correctamente.")

        elif opcion == "2":
            codigo = input("Código de paciente: ")
            fecha = input("Fecha del periodo (YYYY-MM-DD): ")

            if not codigo and not fecha:
                print("❌ Ingrese valores correctamente.")
                continue 
            
            duracion_dias_input = input("Duración del periodo en días (opcional, por defecto 5): ")
            try:
                # Intenta convertir la entrada a un entero
                duracion_int = int(duracion_dias_input)
                # 1. Si la conversión tiene éxito, verifica si el número es válido
                if duracion_int >= 1:
                    duracion_dias = duracion_int  # Es un entero >= 1, guárdalo.
                else:
                    duracion_dias = None  # El número es < 1, guarda None.
            except ValueError:
                # 2. Si la conversión falla (no es un número), guarda None
                duracion_dias = None
            reportes = registrar_periodo(reportes, pacientes, codigo, fecha, duracion_dias)
            guardar_datos(pacientes, reportes)

        elif opcion == "3":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            df_fases = calcular_fases_siguientes(reportes, codigo)
            if codigo is not None and df_fases is not None:
                print("\n📅 Estimación de fases del siguiente ciclo:")
                for _, row in df_fases.iterrows():
                    print(f"\t🩸 {row['fase']}: Desde\t{row['inicio'].date()} \t→ {row['fin'].date()}")

                # También puedes mostrar la fecha estimada del siguiente periodo
                siguiente_inicio = df_fases.loc[df_fases['fase'] == 'Lútea', 'fin'].values[0]
                siguiente_inicio = pd.to_datetime(siguiente_inicio) + timedelta(days=1)
                print(f"\n\t🩸🔮 Próximo período estimado: {siguiente_inicio.date()}")

        elif opcion == "4":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            print("Ingrese una o varias fechas separadas por comas (YYYY-MM-DD), o deje vacío para ver el ciclo estimado.")
            fechas_input = input("Fecha (YYYY-MM-DD): ")
            # Si el usuario ingresa fechas → las usamos
            if fechas_input and codigo:
                try:
                    fechas = [pd.to_datetime(f.strip(), errors="coerce") for f in fechas_input.split(",")]
                    fechas = [f for f in fechas if not pd.isna(f)]
                    if not fechas:
                        print("❌ No se ingresaron fechas válidas.")
                    else:
                        print(' >> Cierre la ventana del gráfico para continuar...')
                        graficar_fases(reportes, codigo, fechas)
                except Exception as e:
                    print(f"❌ Error al procesar fechas: {e}")
            # Si no ingresó nada → calcular el ciclo estimado desde la última fecha registrada
            elif codigo and not fechas_input:
                print("📅 Mostrando el ciclo estimado a partir del último registro...")
                df_fases = calcular_fases_siguientes(reportes, codigo)
                if df_fases is not None:
                    # Extraemos el rango total de fechas estimadas
                    fechas = [df_fases["inicio"].min(), df_fases["fin"].max()]
                    print(' >> Cierre la ventana del gráfico para continuar...')
                    graficar_fases(reportes, codigo, fechas)
            else:
                print("❌ Ingrese valores correctamente.")

        elif opcion == "0":
            print("👋 Saliendo del sistema...")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    menu()
