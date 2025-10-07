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
# CÁLCULO CENTRALIZADO DE FASES
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

def obtener_duracion_menstrual(reportes, codigo):
    """Obtiene la duración menstrual promedio para un paciente"""
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty or "duracion" not in df.columns or df["duracion"].dropna().empty:
        return 5  # valor por defecto
    
    return int(df["duracion"].dropna().mean())

def calcular_fases_ciclo(reportes, codigo):
    """
    Función centralizada para calcular todas las fases del ciclo.
    Ahora usa la estrategia de predicción avanzada.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        return None

    # Usar predicción avanzada
    fecha_predicha, rango_confianza, metadatos = predecir_proximo_ciclo(reportes, codigo)
    
    if fecha_predicha is None:
        # Fallback a cálculo simple
        promedio, desviacion = calcular_promedio_ciclo(reportes, codigo)
        if promedio is None:
            promedio = 28
        fecha_predicha = df["fecha_periodo"].max() + timedelta(days=promedio)
        metodo = "fallback_simple"
    else:
        promedio = metadatos["promedio_ciclo"]
        desviacion = metadatos["desviacion_estandar"]
        metodo = metadatos["metodo"]

    duracion_menstrual = obtener_duracion_menstrual_optima(reportes, codigo)
    ultima_fecha = df["fecha_periodo"].max()

    # Calcular fases (lógica existente mejorada)
    duracion_folicular = 9
    duracion_ovulacion = 2
    duracion_lutea = promedio - (duracion_menstrual + duracion_folicular + duracion_ovulacion)

    # Asegurar duración lútea mínima
    if duracion_lutea < 10:
        duracion_lutea = 10
        # Recalcular folicular para mantener el total
        duracion_folicular = promedio - (duracion_menstrual + duracion_ovulacion + duracion_lutea)

    fases = {
        "Menstrual": (ultima_fecha, ultima_fecha + timedelta(days=duracion_menstrual - 1)),
        "Folicular": (ultima_fecha + timedelta(days=duracion_menstrual),
                      ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular - 1)),
        "Ovulación": (ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular),
                      ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular + duracion_ovulacion - 1)),
        "Lútea": (ultima_fecha + timedelta(days=duracion_menstrual + duracion_folicular + duracion_ovulacion),
                  ultima_fecha + timedelta(days=promedio - 1))
    }

    df_fases = pd.DataFrame([
        {"fase": fase, "inicio": fechas[0], "fin": fechas[1]} for fase, fechas in fases.items()
    ])

    return {
        "promedio": promedio,
        "desviacion": desviacion,
        "duracion_menstrual": duracion_menstrual,
        "fases": df_fases,
        "ultima_fecha": ultima_fecha,
        "siguiente_periodo": fecha_predicha,
        "rango_confianza": rango_confianza,
        "metodo_prediccion": metodo,
        "metadatos": metadatos
    }

def calcular_fases_siguientes(reportes, codigo):
    """
    Función de presentación - usa la lógica centralizada
    """
    resultado = calcular_fases_ciclo(reportes, codigo)
    if resultado is None:
        print("❌ No hay datos para esa paciente.")
        return None
    
    return resultado["fases"]

def calcular_estadisticas_ciclo_avanzado(reportes, codigo, pesos_recientes=0.6):
    """
    Calcula estadísticas del ciclo usando media móvil ponderada.
    pesos_recientes: 0-1, qué tanto peso dar a ciclos recientes (0.6 = 60% a los 3 más recientes)
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    
    # Normalizar fechas y ordenar
    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna().sort_values().drop_duplicates()
    
    if len(fechas) < 2:
        return None, None, None

    # Calcular duraciones entre ciclos
    difs = fechas.diff().dt.days.dropna().tolist()
    
    if not difs:
        return None, None, None

    # Estrategia según cantidad de datos disponibles
    if len(difs) == 1:
        # Solo un ciclo registrado → usar ese valor
        promedio_ponderado = difs[0]
        desviacion = 0
        tendencia = 0
        
    elif len(difs) <= 3:
        # Pocos ciclos → promedio simple
        promedio_ponderado = np.mean(difs)
        desviacion = np.std(difs)
        tendencia = difs[-1] - difs[0]  # tendencia simple
        
    else:
        # Suficientes datos → media móvil ponderada
        n_recientes = max(2, len(difs) // 3)  # últimos 1/3 de ciclos
        ciclos_recientes = difs[-n_recientes:]
        ciclos_antiguos = difs[:-n_recientes]
        
        peso_recientes = pesos_recientes
        peso_antiguos = 1 - peso_recientes
        
        avg_reciente = np.mean(ciclos_recientes)
        avg_antiguo = np.mean(ciclos_antiguos) if ciclos_antiguos else avg_reciente
        
        promedio_ponderado = (avg_reciente * peso_recientes + 
                            avg_antiguo * peso_antiguos)
        
        desviacion = np.std(difs)
        
        # Calcular tendencia (pendiente de los últimos n ciclos)
        x = range(len(difs))
        tendencia = np.polyfit(x, difs, 1)[0]  # pendiente de la regresión lineal

    # Ajustar según tendencia (suavizado)
    ajuste_tendencia = tendencia * 0.3  # solo aplicar 30% de la tendencia
    promedio_ajustado = promedio_ponderado + ajuste_tendencia

    # Limitar valores razonables
    promedio_ajustado = max(21, min(35, promedio_ajustado))
    
    return int(round(promedio_ajustado)), int(round(desviacion)), tendencia

def obtener_duracion_menstrual_optima(reportes, codigo):
    """Calcula duración menstrual considerando estabilidad"""
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    
    if df.empty or "duracion" not in df.columns or df["duracion"].dropna().empty:
        return 5

    duraciones = df["duracion"].dropna().astype(int).tolist()
    
    if len(duraciones) == 1:
        return duraciones[0]
    
    # Para duración menstrual, usar moda o último valor si es estable
    ultima_duracion = duraciones[-1]
    
    # Si la última duración está dentro de 1 día del promedio, confiar en ella
    promedio = np.mean(duraciones)
    if abs(ultima_duracion - promedio) <= 1:
        return ultima_duracion
    else:
        # Si hay mucha variación, usar redondeo del promedio
        return int(round(promedio))

def predecir_proximo_ciclo(reportes, codigo):
    """
    Función principal de predicción que usa estrategia adaptativa
    Retorna: fecha_predicha, rango_confianza, metadatos
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    
    if df.empty:
        return None, None, {"metodo": "sin_datos"}
    
    # Obtener última fecha
    fechas = pd.to_datetime(df["fecha_periodo"], errors="coerce").dropna().sort_values()
    if fechas.empty:
        return None, None, {"metodo": "sin_fechas_validas"}
    
    ultima_fecha = fechas.iloc[-1]
    
    # Calcular estadísticas
    promedio, desviacion, tendencia = calcular_estadisticas_ciclo_avanzado(reportes, codigo)
    
    if promedio is None:
        # Fallback a valor por defecto
        promedio = 28
        desviacion = 2
        metodo = "valor_default"
    else:
        metodo = "modelo_ponderado"
    
    # Calcular fecha predicha
    fecha_base_prediccion = ultima_fecha + timedelta(days=promedio)
    
    # Ajustar según tendencia si hay suficientes datos
    if tendencia and abs(tendencia) > 0.5 and len(fechas) >= 4:
        ajuste_dias = int(round(tendencia * 0.5))  # ajuste conservador
        fecha_base_prediccion += timedelta(days=ajuste_dias)
        metodo = f"modelo_ajustado_tendencia_{ajuste_dias}"
    
    # Calcular rango de confianza basado en desviación
    margen_error = min(5, desviacion)  # máximo 5 días de margen
    fecha_minima = fecha_base_prediccion - timedelta(days=margen_error)
    fecha_maxima = fecha_base_prediccion + timedelta(days=margen_error)
    
    rango_confianza = (fecha_minima, fecha_maxima)
    
    metadatos = {
        "metodo": metodo,
        "promedio_ciclo": promedio,
        "desviacion_estandar": desviacion,
        "tendencia": tendencia,
        "margen_error": margen_error,
        "ciclos_analizados": len(fechas) - 1,
        "ultimo_ciclo": None if len(fechas) < 2 else (fechas.iloc[-1] - fechas.iloc[-2]).days
    }
    
    return fecha_base_prediccion, rango_confianza, metadatos

# ==============================================================
# GRAFICACIÓN (SOLO PARA VISUALIZACIÓN)
# ==============================================================

def graficar_fases(reportes, codigo, fechas_consulta):
    """
    Función que solo se encarga de graficar, usando los datos calculados
    """
    # Obtener datos calculados
    datos_ciclo = calcular_fases_ciclo(reportes, codigo)
    if datos_ciclo is None:
        print("❌ No hay datos para esa paciente.")
        return
    
    promedio = datos_ciclo["promedio"]
    duracion_menstrual = datos_ciclo["duracion_menstrual"]
    df_fases = datos_ciclo["fases"]
    ultima_fecha = datos_ciclo["ultima_fecha"]

    # Normalizar fechas de consulta
    fechas_consulta = sorted(pd.to_datetime(f, errors="coerce") for f in fechas_consulta)
    fechas_consulta = [f for f in fechas_consulta if not pd.isna(f)]
    if not fechas_consulta:
        print("❌ No se ingresaron fechas válidas.")
        return

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

    # --- Colores de fases ---
    colores_fases = {
        "Menstrual": "lightcoral",
        "Folicular": "gold",
        "Ovulación": "limegreen",
        "Lútea": "skyblue"
    }

    # --- Calcular fase para cada día del rango ---
    fases_rango = []
    for fecha in rango:
        # Determinar qué fase corresponde a esta fecha
        fase_encontrada = None
        for _, fase in df_fases.iterrows():
            if fase["inicio"] <= fecha <= fase["fin"]:
                fase_encontrada = fase["fase"]
                break
        
        if fase_encontrada is None:
            # Si no está en ninguna fase calculada, calcular fase teórica
            dias_desde_inicio = (fecha - ultima_fecha).days
            dia_ciclo = dias_desde_inicio % promedio if dias_desde_inicio >= 0 else promedio + (dias_desde_inicio % promedio)
            
            if dia_ciclo < duracion_menstrual:
                fase_encontrada = "Menstrual"
            elif dia_ciclo <= duracion_menstrual + 9:  # Folicular
                fase_encontrada = "Folicular"
            elif dia_ciclo <= duracion_menstrual + 11:  # Ovulación
                fase_encontrada = "Ovulación"
            else:
                fase_encontrada = "Lútea"
        
        fases_rango.append(fase_encontrada)

    # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.canvas.manager.set_window_title("Línea de Tiempo de Período")

    # Dibujar fases
    inicio_fase = rango[0]
    fase_actual = fases_rango[0]
    for i in range(1, len(rango)):
        if fases_rango[i] != fase_actual or i == len(rango) - 1:
            fin_fase = rango[i] if fases_rango[i] != fase_actual else rango[i] + timedelta(days=1)
            
            # Determinar si es estimación (fuera del rango de fases calculadas)
            estimacion = not any(
                (fase["inicio"] <= inicio_fase <= fase["fin"]) or 
                (fase["inicio"] <= fin_fase <= fase["fin"])
                for _, fase in df_fases.iterrows()
            )

            color = colores_fases[fase_actual]
            alpha = 0.4 if estimacion else 0.8
            hatch = '///' if estimacion else None

            ax.axvspan(inicio_fase, fin_fase, color=color, alpha=alpha, hatch=hatch)
            
            # Etiqueta de fase
            centro = inicio_fase + (fin_fase - inicio_fase) / 2
            ax.text(centro, 0.5, fase_actual, ha="center", va="center", fontsize=9, color="black")
            
            inicio_fase = rango[i]
            fase_actual = fases_rango[i]

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
                datos_ciclo = calcular_fases_ciclo(reportes, codigo)
                if datos_ciclo:
                    print(f"🔮 Método de predicción: {datos_ciclo['metodo_prediccion']}")
                    print(f"📊 Ciclos analizados: {datos_ciclo['metadatos']['ciclos_analizados']}")
                    print(f"🎯 Rango de confianza: {datos_ciclo['rango_confianza'][0].date()} a {datos_ciclo['rango_confianza'][1].date()}")

        elif opcion == "4":
            pacientes, reportes = cargar_datos()
            codigo = input("Código de paciente: ")
            if not codigo:
                print("❌ Debe ingresar un código de paciente.")
                continue
            print("Ingrese una o varias fechas separadas por comas (YYYY-MM-DD), o deje vacío para ver el ciclo estimado.")
            fechas_input = input("Fecha (YYYY-MM-DD): ")
            
            if fechas_input:
                try:
                    fechas = [pd.to_datetime(f.strip(), errors="coerce") for f in fechas_input.split(",")]
                    fechas = [f for f in fechas if not pd.isna(f)]
                    if not fechas:
                        print("❌ No se ingresaron fechas válidas.")
                    print(' >> Cierre la ventana del gráfico para continuar...')
                    graficar_fases(reportes, codigo, fechas)
                except Exception as e:
                    print(f"❌ Error al procesar fechas: {e}")
            else:
                print("📅 Mostrando el ciclo estimado a partir del último registro...")
                datos_ciclo = calcular_fases_ciclo(reportes, codigo)
                if datos_ciclo is not None:
                    # Usar las fechas de inicio y fin de todas las fases
                    fecha_inicio = datos_ciclo["fases"]["inicio"].min()
                    fecha_fin = datos_ciclo["fases"]["fin"].max()
                    fechas = [fecha_inicio, fecha_fin]
                    print(' >> Cierre la ventana del gráfico para continuar...')
                    graficar_fases(reportes, codigo, fechas)
                else:
                    print("❌ No se pudieron calcular las fases para este paciente.")

        elif opcion == "0":
            print("👋 Saliendo del sistema...")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    menu()