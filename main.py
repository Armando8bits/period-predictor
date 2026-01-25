import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from matplotlib.patches import Patch

PACIENTES_FILE = "pacientes.csv"
REPORTES_FILE = "reportes.csv"


# ==============================================================
# 1. UTILIDADES BÁSICAS (I/O)
# ==============================================================

def cargar_datos():
    try:
        pacientes = pd.read_csv(PACIENTES_FILE, dtype={"codigo": str, "nombre": str})
    except FileNotFoundError:
        pacientes = pd.DataFrame(columns=["codigo", "nombre"])

    try:
        reportes = pd.read_csv(
            REPORTES_FILE,
            dtype={"codigo": str, "duracion": "Int64"},
            parse_dates=["fecha_periodo"],
            dayfirst=False
        )
    except FileNotFoundError:
        reportes = pd.DataFrame(columns=["codigo", "fecha_periodo", "duracion"])
    
    # Asegurar tipos
    if "codigo" in reportes.columns:
        reportes["codigo"] = reportes["codigo"].astype(str)
    
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
        reportes["fecha_periodo"] = pd.to_datetime(reportes["fecha_periodo"], errors="coerce")
        reportes = pd.concat([reportes, nueva_fila], ignore_index=True)
        
    print(f"🩸 Periodo registrado para #{codigo} en fecha: {fecha} (duración: {duracion} días).")
    return reportes


# ==============================================================
# 2. LÓGICA DE NEGOCIO (ESTADÍSTICAS)
# ==============================================================

def calcular_estadisticas_basicas(reportes, codigo):
    """Devuelve (promedio_ciclo, desviacion, ultima_fecha_registro)"""
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty:
        return 28, 0, None

    df["fecha_periodo"] = pd.to_datetime(df["fecha_periodo"], errors="coerce")
    df = df.dropna(subset=["fecha_periodo"]).sort_values("fecha_periodo")
    
    if df.empty:
        return 28, 0, None

    ultima_fecha = df.iloc[-1]["fecha_periodo"]
    fechas = df["fecha_periodo"].drop_duplicates()
    
    if len(fechas) < 2:
        return 28, 0, ultima_fecha

    difs = fechas.diff().dt.days.dropna()
    promedio = difs.mean()
    desviacion = difs.std()

    if pd.isna(promedio): promedio = 28.0
    if pd.isna(desviacion): desviacion = 0.0

    return int(round(promedio)), int(round(desviacion)), ultima_fecha

def obtener_duracion_menstrual(reportes, codigo, usar_ultimo=False):
    """
    Obtiene la duración menstrual.
    Si usar_ultimo=True, devuelve el dato del último registro (int).
    Si False, devuelve el promedio histórico.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty or "duracion" not in df.columns:
        return 5
    
    serie_valida = df["duracion"].dropna()
    if serie_valida.empty:
        return 5
    
    if usar_ultimo:
        # Petición usuario: obtener el último
        return int(serie_valida.iloc[-1])
    else:
        return int(serie_valida.mean())


# ==============================================================
# 3. NÚCLEO DE CÁLCULO DE FASES (CENTRALIZADO)
# ==============================================================

def generar_intervalos_fases(fecha_inicio, duracion_ciclo, duracion_menstrual):
    """
    Función pura: Recibe fecha inicio y duraciones, devuelve el diccionario de fases
    usando la lógica secuencial para evitar solapamientos.
    """
    # 1. Definir duraciones internas biológicamente razonables
    # Fase Ovulación es casi siempre fija
    duracion_ovulacion = 2 
    
    # Fase Lútea depende ligeramente del ciclo total (estándar 14)
    if duracion_ciclo <= 28:
        duracion_lutea = 12
    elif duracion_ciclo <= 35:
        duracion_lutea = 14
    else:
        duracion_lutea = 16
        
    # Fase Folicular es el resto ("la variable")
    duracion_folicular = duracion_ciclo - (duracion_menstrual + duracion_ovulacion + duracion_lutea)
    
    # AJUSTES DE SEGURIDAD (Si el cálculo da negativo o muy corto)
    if duracion_folicular < 6: 
        duracion_folicular = 6
        # Recalculamos lutea para que encaje, o aceptamos que el ciclo proyectado será más largo
        duracion_lutea = max(10, duracion_ciclo - (duracion_menstrual + duracion_ovulacion + duracion_folicular))
    
    # 2. Calcular Fechas Secuenciales (Lógica corregida)
    # Menstrual
    inicio_menstrual = fecha_inicio
    fin_menstrual = inicio_menstrual + timedelta(days=duracion_menstrual - 1)

    # Folicular
    inicio_folicular = fin_menstrual + timedelta(days=1)
    fin_folicular = inicio_folicular + timedelta(days=duracion_folicular - 1)

    # Ovulación
    inicio_ovulacion = fin_folicular + timedelta(days=1)
    fin_ovulacion = inicio_ovulacion + timedelta(days=duracion_ovulacion - 1)

    # Lútea
    inicio_lutea = fin_ovulacion + timedelta(days=1)
    # El fin de la lútea debe coincidir con el fin del ciclo teórico
    fin_lutea = fecha_inicio + timedelta(days=duracion_ciclo - 1)
    
    # Doble verificación por si los ajustes de seguridad movieron el final
    if fin_lutea < inicio_lutea:
        fin_lutea = inicio_lutea # Edge case ciclos muy cortos

    fases = {
        "Menstrual": (inicio_menstrual, fin_menstrual),
        "Folicular": (inicio_folicular, fin_folicular),
        "Ovulación": (inicio_ovulacion, fin_ovulacion),
        "Lútea": (inicio_lutea, fin_lutea)
    }
    
    # Devolvemos también las duraciones calculadas para info
    metadatos = {
        "dur_menstrual": duracion_menstrual,
        "dur_folicular": (fin_folicular - inicio_folicular).days + 1,
        "dur_ovulacion": (fin_ovulacion - inicio_ovulacion).days + 1,
        "dur_lutea": (fin_lutea - inicio_lutea).days + 1
    }
    
    return fases, metadatos


# ==============================================================
# 4. FUNCIONES DE SERVICIO (USAN EL NÚCLEO)
# ==============================================================

def calcular_prediccion_opcion_3(reportes, codigo):
    """
    Lógica para la Opción 3 del menú.
    Usa el último registro y el PROMEDIO histórico para proyectar.
    """
    promedio, _, ultima_fecha = calcular_estadisticas_basicas(reportes, codigo)
    
    if ultima_fecha is None:
        return None

    # Para la predicción futura, usamos la duración menstrual del último registro (petición usuario)
    duracion_menstrual = obtener_duracion_menstrual(reportes, codigo, usar_ultimo=True)
    
    # LLAMADA CENTRALIZADA
    fases, meta = generar_intervalos_fases(ultima_fecha, promedio, duracion_menstrual)
    
    df_fases = pd.DataFrame([
        {"fase": k, "inicio": v[0], "fin": v[1]} for k, v in fases.items()
    ])
    
    return {
        "fases_actual": df_fases,
        "siguiente_periodo": ultima_fecha + timedelta(days=promedio),
        "ultima_fecha": ultima_fecha,
        "duracion_menstrual": duracion_menstrual,
        "promedio_ciclo": promedio
    }

def calcular_datos_grafico(reportes, codigo, fecha_consulta):
    """
    Lógica para graficar. Determina QUÉ ciclo graficar (pasado, presente o futuro)
    y luego llama al núcleo para calcular las fases.
    """
    df = reportes[reportes["codigo"].astype(str) == str(codigo)].copy()
    if df.empty: return None

    # Datos base
    promedio, desviacion, _ = calcular_estadisticas_basicas(reportes, codigo)
    duracion_menstrual_std = obtener_duracion_menstrual(reportes, codigo, usar_ultimo=False)

    df["fecha_periodo"] = pd.to_datetime(df["fecha_periodo"], errors="coerce")
    df = df.dropna(subset=["fecha_periodo"]).sort_values("fecha_periodo")
    
    # 1. Encontrar ciclo relevante para la fecha_consulta
    # Buscar registros ANTERIORES a la consulta
    ciclos_anteriores = df[df["fecha_periodo"] <= fecha_consulta]
    
    usar_siguiente_real = False
    siguiente_lejano = None
    
    if not ciclos_anteriores.empty:
        inicio_ciclo = ciclos_anteriores.iloc[-1]["fecha_periodo"]
        
        # Buscar si hay un cierre real de este ciclo (un registro posterior)
        ciclos_posteriores = df[df["fecha_periodo"] > fecha_consulta]
        
        # Lógica de ventana de tiempo (max 45 días o 2 ciclos)
        limite_dias = min(45, promedio * 2)
        posteriores_cercanos = ciclos_posteriores[ciclos_posteriores["fecha_periodo"] <= (inicio_ciclo + timedelta(days=limite_dias))]
        
        if not posteriores_cercanos.empty:
            # CASO A: Es un ciclo histórico cerrado (tiene inicio y fin real)
            siguiente_periodo = posteriores_cercanos.iloc[0]["fecha_periodo"]
            duracion_ciclo_calc = (siguiente_periodo - inicio_ciclo).days
            usar_siguiente_real = True
        else:
            # CASO B: Es el ciclo actual abierto (proyección)
            duracion_ciclo_calc = promedio
            siguiente_periodo = inicio_ciclo + timedelta(days=promedio)
            
            # Verificar si hay datos futuros lejanos
            if not ciclos_posteriores.empty:
                siguiente_lejano = ciclos_posteriores.iloc[0]["fecha_periodo"]

    else:
        # CASO C: La fecha consultada es anterior al primer registro
        # Proyectamos hacia atrás desde el primero
        primer_registro = df.iloc[0]["fecha_periodo"]
        dias_diff = (primer_registro - fecha_consulta).days
        ciclos_atras = (dias_diff // promedio) + 1
        inicio_ciclo = primer_registro - timedelta(days=ciclos_atras * promedio)
        duracion_ciclo_calc = promedio
        siguiente_periodo = inicio_ciclo + timedelta(days=promedio)

    # Limitar proyecciones locas
    duracion_ciclo_final = min(duracion_ciclo_calc, 50) 

    # LLAMADA CENTRALIZADA
    fases, metadatos = generar_intervalos_fases(inicio_ciclo, duracion_ciclo_final, duracion_menstrual_std)

    df_fases = pd.DataFrame([
        {"fase": k, "inicio": v[0], "fin": v[1]} for k, v in fases.items()
    ])
    
    return {
        "promedio": promedio,
        "fases": df_fases,
        "inicio_ciclo": inicio_ciclo,
        "siguiente_periodo": siguiente_periodo,
        "usar_siguiente_real": usar_siguiente_real,
        "duracion_ciclo_actual": duracion_ciclo_final,
        "metadatos": metadatos,
        "siguiente_lejano": siguiente_lejano
    }


# ==============================================================
# 5. GRAFICACIÓN
# ==============================================================

def graficar_fases_por_fecha(reportes, codigo, fechas_consulta):
    if not fechas_consulta: return
    
    fecha_min = min(fechas_consulta) - timedelta(days=5)
    fecha_max = max(fechas_consulta) + timedelta(days=5)
    rango_completo = pd.date_range(fecha_min, fecha_max, freq="D")

    colores = {"Menstrual": "lightcoral", "Folicular": "gold", 
               "Ovulación": "limegreen", "Lútea": "skyblue"}

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Cache simple para no recalcular el mismo ciclo 30 veces
    cache_ciclos = {} 

    for fecha in rango_completo:
        # Optimización: agrupar por semanas o ciclos para no llamar a logic por dia
        # Pero para mantener exactitud, llamamos logic
        
        datos = calcular_datos_grafico(reportes, codigo, fecha)
        if not datos: continue
        
        # Determinar color
        color_dia = "white"
        for _, fase in datos["fases"].iterrows():
            if fase["inicio"] <= fecha <= fase["fin"]:
                color_dia = colores.get(fase["fase"], "white")
                break
        
        if color_dia != "white":
            ax.axvspan(fecha, fecha + timedelta(days=1), color=color_dia, alpha=0.7)

    for f in fechas_consulta:
        ax.axvline(f, color="black", linestyle="--", lw=2)

    legend_elements = [Patch(facecolor=c, label=f) for f, c in colores.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(fecha_min, fecha_max)
    ax.set_title(f"Vista de Calendario — Paciente {codigo}")
    plt.tight_layout()
    plt.show()

def graficar_ciclo_completo(reportes, codigo, fecha_ref=None):
    if fecha_ref is None: fecha_ref = datetime.now()
    
    datos = calcular_datos_grafico(reportes, codigo, fecha_ref)
    if not datos: 
        print("❌ No hay datos suficientes.")
        return

    # Extraer datos procesados
    fases = datos["fases"]
    inicio = datos["inicio_ciclo"]
    fin = datos["siguiente_periodo"] + timedelta(days=datos["metadatos"]["dur_menstrual"] - 1)
    
    if datos["siguiente_lejano"]:
        fin = max(fin, datos["siguiente_lejano"] + timedelta(days=5))

    colores = {"Menstrual": "lightcoral", "Folicular": "gold", 
               "Ovulación": "limegreen", "Lútea": "skyblue"}

    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Dibujar fases ciclo principal
    for _, f in fases.iterrows():
        ax.axvspan(f["inicio"], f["fin"] + timedelta(days=1), color=colores[f["fase"]], alpha=0.7)
        centro = f["inicio"] + (f["fin"] - f["inicio"]) / 2
        ax.text(centro, 0.5, f["fase"], ha="center", va="center", fontsize=9, fontweight='bold')

    # Dibujar inicio del siguiente
    sig_inicio = datos["siguiente_periodo"]
    sig_fin = sig_inicio + timedelta(days=datos["metadatos"]["dur_menstrual"] - 1)
    
    estilo_sig = "Menstrual (Real)" if datos["usar_siguiente_real"] else "Menstrual (Estim.)"
    alpha_sig = 0.7 if datos["usar_siguiente_real"] else 0.3
    
    ax.axvspan(sig_inicio, sig_fin + timedelta(days=1), color="lightcoral", alpha=alpha_sig)
    ax.text(sig_inicio + (sig_fin - sig_inicio)/2, 0.5, estilo_sig, ha="center", fontsize=8)

    # Referencia
    ax.axvline(fecha_ref, color="red", label="Fecha Ref")
    
    # Formato
    ax.set_xlim(inicio - timedelta(days=2), fin + timedelta(days=2))
    ax.set_yticks([])
    
    titulo = f"Ciclo del {inicio.date()} al {datos['siguiente_periodo'].date()}"
    titulo += f" | Duración: {datos['duracion_ciclo_actual']} días"
    ax.set_title(titulo)
    
    # Texto info
    info = f"Promedio Histórico: {datos['promedio']} días\n"
    info += f"Lútea calculada: {datos['metadatos']['dur_lutea']} días"
    ax.text(0.02, 0.95, info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# ==============================================================
# 6. MENÚ
# ==============================================================

def menu():
    pacientes, reportes = cargar_datos()

    while True:
        print("\n=== SISTEMA DE SEGUIMIENTO MENSTRUAL (Refactorizado) ===")
        print("1️⃣  Registrar paciente")
        print("2️⃣  Registrar periodo")
        print("3️⃣  Ver predicción (Texto)")
        print("4️⃣  Graficar fechas específicas")
        print("5️⃣  Ver ciclo completo actual")
        print("0️⃣  Salir")
        opcion = input("Seleccione: ")

        if opcion == "1":
            c = input("Código: ")
            n = input("Nombre: ")
            if c and n:
                pacientes = registrar_paciente(pacientes, c, n)
                guardar_datos(pacientes, reportes)

        elif opcion == "2":
            c = input("Código: ")
            f = input("Fecha (YYYY-MM-DD): ")
            d = input("Duración (Enter=5): ")
            dur = int(d) if d.isdigit() and int(d) > 0 else 5
            reportes = registrar_periodo(reportes, pacientes, c, f, dur)
            guardar_datos(pacientes, reportes)

        elif opcion == "3":
            # REFACTORIZADO: Usa la lógica centralizada
            pacientes, reportes = cargar_datos()
            c = input("Código: ")
            res = calcular_prediccion_opcion_3(reportes, c)
            
            if res:
                print(f"\n📅 Ciclo actual (Inicio: {res['ultima_fecha'].date()})")
                print(f"   Duración menstrual usada: {res['duracion_menstrual']} días (del último registro)")
                print("-" * 40)
                for _, row in res["fases_actual"].iterrows():
                    print(f"   {row['fase']:<10}: {row['inicio'].date()} -> {row['fin'].date()}")
                print("-" * 40)
                print(f"🔮 Próximo periodo estimado: {res['siguiente_periodo'].date()}")
            else:
                print("❌ Sin datos.")

        elif opcion == "4":
            pacientes, reportes = cargar_datos()
            c = input("Código: ")
            fechas_str = input("Fechas (YYYY-MM-DD, ...): ")
            fechas = [pd.to_datetime(f.strip()) for f in fechas_str.split(",") if f.strip()]
            graficar_fases_por_fecha(reportes, c, fechas)

        elif opcion == "5":
            pacientes, reportes = cargar_datos()
            c = input("Código: ")
            f_str = input("Fecha ref (Enter=Hoy): ")
            f_ref = pd.to_datetime(f_str) if f_str else datetime.now()
            graficar_ciclo_completo(reportes, c, f_ref)

        elif opcion == "0":
            break

if __name__ == "__main__":
    menu()