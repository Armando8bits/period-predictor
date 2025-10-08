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
            dtype={"codigo": str, "duracion":"Int64"},
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
        reportes["duracion"] = reportes["duracion"].astype("Int64")
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
    
    nueva_fila = pd.DataFrame.from_records(
        [{"codigo": codigo, "fecha_periodo": pd.to_datetime(fecha), "duracion": duracion}],
        columns=["codigo", "fecha_periodo", "duracion"]
    )

    if reportes.empty:
        reportes = nueva_fila.copy()
    else:
        if not np.issubdtype(reportes["fecha_periodo"].dtype, np.datetime64):
            reportes["fecha_periodo"] = pd.to_datetime(reportes["fecha_periodo"], errors="coerce")

        reportes = pd.concat([reportes, nueva_fila], ignore_index=True)
    print(f"🩸 Periodo registrado para #{codigo} en fecha: {fecha} (duración: {duracion} días).")
    return reportes


# ==============================================================
# CÁLCULO CENTRALIZADO DE FASES
# ==============================================================

def calcular_promedio_ciclo(reportes, codigo):
    """
    Calcula promedio y desviación del ciclo para 'codigo'.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        return None, None

    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna().sort_values().drop_duplicates()
    if len(fechas) < 2:
        return None, None

    difs = fechas.diff().dt.days.dropna()
    if len(difs) == 0:
        return None, None

    promedio = difs.mean()
    desviacion = difs.std()

    if pd.isna(promedio):
        promedio = 28.0
    if pd.isna(desviacion):
        desviacion = 0.0

    return int(round(promedio)), int(round(desviacion))

def obtener_duracion_menstrual(reportes, codigo):
    """Obtiene la duración menstrual promedio para un paciente"""
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty or "duracion" not in df.columns or df["duracion"].dropna().empty:
        return 5
    
    return int(df["duracion"].dropna().mean())

def calcular_fases_para_fecha(reportes, codigo, fecha_consulta):
    """
    Calcula las fases del ciclo para una fecha específica.
    Encuentra el ciclo más cercano a la fecha consultada.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        return None

    # Calcular estadísticas
    promedio, desviacion = calcular_promedio_ciclo(reportes, codigo)
    if promedio is None:
        promedio = 28
    
    duracion_menstrual = obtener_duracion_menstrual(reportes, codigo)

    # Encontrar el inicio de ciclo más cercano a la fecha consultada
    df["fecha_periodo"] = pd.to_datetime(df["fecha_periodo"], errors="coerce")
    df = df.dropna(subset=["fecha_periodo"]).sort_values("fecha_periodo")
    
    if df.empty:
        return None

    # Encontrar el inicio de ciclo anterior más cercano a la fecha consultada
    ciclos_anteriores = df[df["fecha_periodo"] <= fecha_consulta]
    if not ciclos_anteriores.empty:
        # Usar el ciclo más reciente anterior a la fecha consultada
        inicio_ciclo = ciclos_anteriores.iloc[-1]["fecha_periodo"]
    else:
        # Si no hay ciclos anteriores, usar el primero y proyectar hacia atrás
        inicio_ciclo = df.iloc[0]["fecha_periodo"]
        # Calcular cuántos ciclos completos hay entre el primer registro y la fecha consultada
        dias_diferencia = (fecha_consulta - inicio_ciclo).days
        ciclos_completos = dias_diferencia // promedio
        inicio_ciclo = inicio_ciclo + timedelta(days=ciclos_completos * promedio)

    # Calcular fases basadas en el inicio del ciclo encontrado
    duracion_folicular = 9
    duracion_ovulacion = 2
    duracion_lutea = promedio - (duracion_menstrual + duracion_folicular + duracion_ovulacion)

    # Asegurar duración lútea mínima
    if duracion_lutea < 10:
        duracion_lutea = 10
        duracion_folicular = promedio - (duracion_menstrual + duracion_ovulacion + duracion_lutea)

    fases = {
        "Menstrual": (inicio_ciclo, inicio_ciclo + timedelta(days=duracion_menstrual - 1)),
        "Folicular": (inicio_ciclo + timedelta(days=duracion_menstrual),
                      inicio_ciclo + timedelta(days=duracion_menstrual + duracion_folicular - 1)),
        "Ovulación": (inicio_ciclo + timedelta(days=duracion_menstrual + duracion_folicular),
                      inicio_ciclo + timedelta(days=duracion_menstrual + duracion_folicular + duracion_ovulacion - 1)),
        "Lútea": (inicio_ciclo + timedelta(days=duracion_menstrual + duracion_folicular + duracion_ovulacion),
                  inicio_ciclo + timedelta(days=promedio - 1))
    }

    df_fases = pd.DataFrame([
        {"fase": fase, "inicio": fechas[0], "fin": fechas[1]} for fase, fechas in fases.items()
    ])

    siguiente_periodo = inicio_ciclo + timedelta(days=promedio)

    return {
        "promedio": promedio,
        "desviacion": desviacion,
        "duracion_menstrual": duracion_menstrual,
        "fases": df_fases,
        "inicio_ciclo": inicio_ciclo,
        "siguiente_periodo": siguiente_periodo,
        "fecha_consulta": fecha_consulta
    }

def calcular_fases_siguientes(reportes, codigo):
    """
    Calcula las fases del siguiente ciclo basado en el último registro.
    (Para la opción 3 del menú)
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        print("❌ No hay datos para esa paciente.")
        return None

    # Usar la fecha actual para encontrar el ciclo actual
    fecha_actual = datetime.now()
    resultado = calcular_fases_para_fecha(reportes, codigo, fecha_actual)
    
    if resultado is None:
        return None
    
    return resultado["fases"]


# ==============================================================
# GRAFICACIÓN (SOLO PARA VISUALIZACIÓN)
# ==============================================================

def graficar_fases_por_fecha(reportes, codigo, fechas_consulta):
    """
    Función que solo se encarga de graficar las fases para las fechas específicas
    """
    if not fechas_consulta:
        print("❌ No se proporcionaron fechas para graficar.")
        return

    # Calcular el rango de fechas a mostrar
    fecha_min = min(fechas_consulta)
    fecha_max = max(fechas_consulta)
    
    # Extender el rango para mejor visualización
    margen = 5
    fecha_inicio = fecha_min - timedelta(days=margen)
    fecha_fin = fecha_max + timedelta(days=margen)
    
    rango_completo = pd.date_range(fecha_inicio, fecha_fin, freq="D")

    # --- Colores de fases ---
    colores_fases = {
        "Menstrual": "lightcoral",
        "Folicular": "gold",
        "Ovulación": "limegreen",
        "Lútea": "skyblue"
    }

    # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.canvas.manager.set_window_title("Línea de Tiempo de Período")

    # Para cada día en el rango, determinar la fase
    for fecha in rango_completo:
        # Calcular las fases para esta fecha específica
        datos_ciclo = calcular_fases_para_fecha(reportes, codigo, fecha)
        
        if datos_ciclo is None:
            continue

        # Determinar qué fase corresponde a esta fecha
        fase_actual = None
        for _, fase_info in datos_ciclo["fases"].iterrows():
            if fase_info["inicio"] <= fecha <= fase_info["fin"]:
                fase_actual = fase_info["fase"]
                break
        
        if fase_actual is None:
            # Si no está en ninguna fase del ciclo actual, verificar si es período menstrual del siguiente ciclo
            siguiente_inicio = datos_ciclo["siguiente_periodo"]
            siguiente_fin = siguiente_inicio + timedelta(days=datos_ciclo["duracion_menstrual"] - 1)
            if siguiente_inicio <= fecha <= siguiente_fin:
                fase_actual = "Menstrual"
            else:
                # Calcular fase teórica basada en días desde inicio del ciclo
                dias_desde_inicio = (fecha - datos_ciclo["inicio_ciclo"]).days
                dia_ciclo = dias_desde_inicio % datos_ciclo["promedio"]
                
                if dia_ciclo < datos_ciclo["duracion_menstrual"]:
                    fase_actual = "Menstrual"
                elif dia_ciclo < datos_ciclo["duracion_menstrual"] + 9:
                    fase_actual = "Folicular"
                elif dia_ciclo < datos_ciclo["duracion_menstrual"] + 11:
                    fase_actual = "Ovulación"
                else:
                    fase_actual = "Lútea"

        if fase_actual:
            color = colores_fases.get(fase_actual, "white")
            # Dibujar barra vertical para este día
            ax.axvspan(fecha, fecha + timedelta(days=1), color=color, alpha=0.7)

    # --- Marcar fechas consultadas ---
    for f in fechas_consulta:
        ax.axvline(f, color="black", linestyle="--", linewidth=2, label='Fecha consultada')

    # --- Leyenda ---
    legend_elements = [
        Patch(facecolor='lightcoral', label='Menstrual'),
        Patch(facecolor='gold', label='Folicular'),
        Patch(facecolor='limegreen', label='Ovulación'),
        Patch(facecolor='skyblue', label='Lútea'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label='Fecha consultada')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')

    # --- Formatear gráfico ---
    ax.set_xlim(fecha_inicio, fecha_fin)
    ax.set_xticks(pd.date_range(fecha_inicio, fecha_fin, freq='7D'))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in pd.date_range(fecha_inicio, fecha_fin, freq='7D')], rotation=45)
    ax.set_yticks([])
    ax.set_title(f"Fases del ciclo — Paciente {codigo}", fontsize=14, pad=20)
    
    # Añadir línea de tiempo con las fases
    '''for i, (fase, color) in enumerate(colores_fases.items()):
        ax.text(0.02, 0.95 - i*0.05, fase, transform=ax.transAxes, 
                color=color, fontsize=12, fontweight='bold')'''

    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.98, 0.02, f"Impreso el {fecha_hora}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()
    plt.show()

def graficar_ciclo_completo(reportes, codigo, fecha_referencia=None):
    """
    Grafica un ciclo completo a partir de una fecha de referencia
    """
    if fecha_referencia is None:
        fecha_referencia = datetime.now()
    
    datos_ciclo = calcular_fases_para_fecha(reportes, codigo, fecha_referencia)
    if datos_ciclo is None:
        print("❌ No hay datos para graficar.")
        return

    # Crear rango de fechas para un ciclo completo
    inicio_ciclo = datos_ciclo["inicio_ciclo"]
    fin_ciclo = inicio_ciclo + timedelta(days=datos_ciclo["promedio"])
    rango_ciclo = pd.date_range(inicio_ciclo, fin_ciclo, freq="D")

    # --- Colores de fases ---
    colores_fases = {
        "Menstrual": "lightcoral",
        "Folicular": "gold",
        "Ovulación": "limegreen",
        "Lútea": "skyblue"
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.canvas.manager.set_window_title("Ciclo Menstrual Completo")

    # Dibujar cada fase
    for _, fase in datos_ciclo["fases"].iterrows():
        inicio = fase["inicio"]
        fin = fase["fin"] + timedelta(days=1)  # Incluir el último día
        color = colores_fases[fase["fase"]]
        ax.axvspan(inicio, fin, color=color, alpha=0.7, label=fase["fase"])
        
        # Etiqueta de la fase
        centro = inicio + (fin - inicio) / 2
        ax.text(centro, 0.5, fase["fase"], ha="center", va="center", 
                fontsize=10, fontweight='bold', color='black')

    # Marcar fecha de referencia
    ax.axvline(fecha_referencia, color="red", linestyle="-", linewidth=2, label='Fecha de referencia')

    ax.set_xlim(inicio_ciclo, fin_ciclo)
    ax.set_xticks(pd.date_range(inicio_ciclo, fin_ciclo, freq='5D'))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in pd.date_range(inicio_ciclo, fin_ciclo, freq='5D')], rotation=45)
    ax.set_yticks([])
    ax.set_title(f"Ciclo Menstrual — Paciente {codigo}\n(Duración: {datos_ciclo['promedio']} días)", fontsize=12)
    ax.legend(loc='upper right')

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
        print("5️⃣  Ver ciclo completo actual")  # Nueva opción
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
                duracion_int = int(duracion_dias_input)
                if duracion_int >= 1:
                    duracion_dias = duracion_int
                else:
                    duracion_dias = None
            except ValueError:
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

                # Mostrar fecha estimada del siguiente periodo
                datos_ciclo = calcular_fases_para_fecha(reportes, codigo, datetime.now())
                if datos_ciclo:
                    siguiente_periodo = datos_ciclo["siguiente_periodo"]
                    print(f"\n\t🩸🔮 Próximo período estimado: {siguiente_periodo.date()}")

        elif opcion == "4":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            if not codigo:
                print("❌ Debe ingresar un código de paciente.")
                continue
            
            print("Ingrese una o varias fechas separadas por comas (YYYY-MM-DD):")
            fechas_input = input("Fechas: ").strip()
            
            if fechas_input:
                try:
                    fechas = []
                    for f in fechas_input.split(","):
                        fecha_parsed = pd.to_datetime(f.strip(), errors="coerce")
                        if not pd.isna(fecha_parsed):
                            fechas.append(fecha_parsed)
                    
                    if not fechas:
                        print("❌ No se ingresaron fechas válidas.")
                    else:
                        print(' >> Cierre la ventana del gráfico para continuar...')
                        graficar_fases_por_fecha(reportes, codigo, fechas)
                except Exception as e:
                    print(f"❌ Error al procesar fechas: {e}")
            else:
                print("❌ Debe ingresar al menos una fecha.")

        elif opcion == "5":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            if not codigo:
                print("❌ Debe ingresar un código de paciente.")
                continue
            
            fecha_input = input("Fecha de referencia (YYYY-MM-DD) o Enter para hoy: ").strip()
            if fecha_input:
                try:
                    fecha_referencia = pd.to_datetime(fecha_input, errors="coerce")
                    if pd.isna(fecha_referencia):
                        fecha_referencia = datetime.now()
                except:
                    fecha_referencia = datetime.now()
            else:
                fecha_referencia = datetime.now()
            
            print(' >> Cierre la ventana del gráfico para continuar...')
            graficar_ciclo_completo(reportes, codigo, fecha_referencia)

        elif opcion == "0":
            print("👋 Saliendo del sistema...")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    menu()