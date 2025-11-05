# consultas_sql.py
import os
import re
import json
import time
import pyodbc
import numpy as np
import pandas as pd
import streamlit as st
import openai
from datetime import datetime
from PIL import Image  # para mostrar el logo

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="Asistente SQL IA", layout="wide")

# API Key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.warning("⚠️ Define OPENAI_API_KEY en st.secrets o como variable de entorno.")
openai.api_key = OPENAI_API_KEY

# Archivos
SQL_FILE = "consultas_ssms.sql"
DICT_FILE = "Diccionario.csv"

# Config DB
DB_CONFIG = {
    "server": st.secrets.get("DB_SERVER", os.getenv("DB_SERVER", "")),
    "database": st.secrets.get("DB_NAME", os.getenv("DB_NAME", "")),
    "username": st.secrets.get("DB_USER", os.getenv("DB_USER", "")),
    "password": st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD", "")),
    "driver": st.secrets.get("DB_DRIVER", os.getenv("DB_DRIVER", "")),
}

# =========================
# Utilidades
# =========================
def normalize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s or "").lower()

def is_code_column(col: str) -> bool:
    u = col.upper()
    return (
        "COD" in u
        or u.endswith("_ID")
        or u.endswith("_CODE")
        or u.startswith("NRO")
        or u.startswith("NUM")
        or "CUIT" in u
        or u == "COD_CLIENT"
    )

def resolve_column(user_col: str, df_cols: list, diccionario: pd.DataFrame | None = None) -> str | None:
    if not user_col:
        return None
    user_norm = normalize_name(user_col)
    norm_map = {normalize_name(c): c for c in df_cols}
    if user_norm in norm_map:
        return norm_map[user_norm]
    synonyms = {
        "cliente": ["RAZON_SOCI", "NOMBRE", "CLIENTE", "NOM_CLIENTE"],
        "codcliente": ["COD_CLIENT", "CODCLI", "CLIENTE_ID"],
        "importe": ["IMPORTE", "SALDO", "TOTAL", "IMPORTE_PENDIENTE_MON_CTE", "IMPORTE_PENDIENTE"],
        "fecha": ["FECHA", "FECHA_EMIS", "FECHA_EMISION", "FECHA_VTO", "FECHA_VENCIMIENTO"],
        "zona": ["COD_ZONA", "NOMBRE_ZON", "ZONA"],
    }
    for key, cols in synonyms.items():
        if user_norm == key and any(c in df_cols for c in cols):
            for c in cols:
                if c in df_cols:
                    return c
    if diccionario is not None and not diccionario.empty:
        mask = diccionario["Descripcion"].astype(str).str.contains(user_col, case=False, na=False)
        if mask.any():
            col_name = diccionario.loc[mask, "Columna"].values[0]
            if col_name in df_cols:
                return col_name
    for c in df_cols:
        if user_norm in normalize_name(c):
            return c
    return None

def parse_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def to_numeric_safe(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return series
    s = series.astype(str).str.replace(",", ".", regex=False)
    mask_num = s.str.match(r"^-?\d+(\.\d+)?$")
    if mask_num.mean() >= 0.7:
        return pd.to_numeric(s, errors="coerce")
    return series

# =========================
# Cargar consultas
# =========================
def cargar_consultas(nombre_archivo: str = SQL_FILE) -> dict:
    ruta = os.path.join(os.path.dirname(__file__), nombre_archivo)
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró el archivo SQL: {ruta}")
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()
    consultas = {}
    partes = re.split(r"--\s*Vista:\s*(.+)", contenido, flags=re.IGNORECASE)
    bloques = partes[1:]
    for i in range(0, len(bloques), 2):
        nombre = bloques[i].strip()
        sql = re.sub(r'^\s*GO\s*$', '', bloques[i + 1], flags=re.IGNORECASE | re.MULTILINE).strip()
        if nombre and sql:
            consultas[nombre] = sql
    if not consultas:
        st.warning("⚠️ No se encontraron consultas válidas en el archivo SQL.")
    return consultas

# =========================
# Cargar Diccionario
# =========================
def cargar_diccionario(nombre_archivo: str = DICT_FILE) -> pd.DataFrame:
    ruta = os.path.join(os.path.dirname(__file__), nombre_archivo)
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró el archivo Diccionario: {ruta}")
        return pd.DataFrame()
    df = pd.read_csv(ruta, sep=",")
    return df

# =========================
# Ejecutar SQL con barra de progreso
# =========================
def ejecutar_consulta(sql: str) -> pd.DataFrame:
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']};"
        f"Authentication=SqlPassword;"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=30;"
    )
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            bloques = re.split(r'^\s*GO\s*$', sql, flags=re.IGNORECASE | re.MULTILINE)
            last_rows, last_cols = None, None
            placeholder = st.empty()
            progreso = placeholder.progress(0)
            total = len([b for b in bloques if b.strip()])
            for i, bloque in enumerate(bloques, start=1):
                if not bloque.strip():
                    continue
                cursor.execute(bloque)
                if cursor.description:
                    cols = [c[0] for c in cursor.description]
                    rows = cursor.fetchall()
                    last_rows, last_cols = rows, cols
                progreso.progress(int(i / total * 100))
                time.sleep(0.05)
            placeholder.empty()
            if last_rows is None or last_cols is None:
                return pd.DataFrame()
            df = pd.DataFrame([list(r) for r in last_rows], columns=last_cols)
            return df
    except pyodbc.Error as e:
        st.error(f"Error al ejecutar la consulta: {e}")
        return pd.DataFrame()

# =========================
# Embeddings y mapeo de consultas
# =========================
def obtener_embedding(texto: str) -> np.ndarray:
    try:
        resp = openai.Embedding.create(input=texto, model="text-embedding-3-small")
        return np.array(resp["data"][0]["embedding"])
    except Exception as e:
        st.warning(f"No se pudo obtener embedding: {e}")
        return np.zeros(1536, dtype=float)

def mapear_consulta_embeddings(pregunta: str, consultas: dict) -> tuple[str, str]:
    """
    Selecciona la consulta SQL más relevante según la pregunta del usuario,
    combinando similitud semántica (embeddings) y coincidencia de palabras clave.
    """
    if not consultas:
        return "", ""
    nombres = list(consultas.keys())
    textos = [f"{n}\n{consultas[n]}" for n in nombres]

    # === Paso 1: embeddings ===
    vecs = [obtener_embedding(t) for t in textos]
    vq = obtener_embedding(pregunta)
    sims = [float(np.dot(v, vq) / ((np.linalg.norm(v) or 1) * (np.linalg.norm(vq) or 1))) for v in vecs]

    # === Paso 2: refuerzo por coincidencia léxica ===
    pregunta_tokens = re.findall(r"\w+", pregunta.lower())
    bonus_scores = []
    for n, sql in zip(nombres, textos):
        texto_total = f"{n} {sql}".lower()
        coincidencias = sum(1 for t in pregunta_tokens if t in texto_total)
        bonus = coincidencias / (len(pregunta_tokens) or 1)
        bonus_scores.append(bonus * 0.2)  # peso moderado
    sims = [s + b for s, b in zip(sims, bonus_scores)]

    # === Paso 3: selección principal ===
    idx_ordenado = np.argsort(sims)[::-1]
    mejor_idx = int(idx_ordenado[0])
    segundo_idx = int(idx_ordenado[1]) if len(idx_ordenado) > 1 else mejor_idx

    # === Paso 4: verificación contextual (opcional) ===
    diferencia = sims[mejor_idx] - sims[segundo_idx]
    if diferencia < 0.05 and OPENAI_API_KEY:
        try:
            prompt = f"Pregunta: '{pregunta}'\nOpciones:\n1. {nombres[mejor_idx]}\n2. {nombres[segundo_idx]}\n" \
                     "¿Cuál describe mejor lo que el usuario quiere consultar? Responde solo con '1' o '2'."
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un selector experto de consultas SQL."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            ans = resp["choices"][0]["message"]["content"].strip()
            if ans == "2":
                mejor_idx = segundo_idx
        except Exception as e:
            st.warning(f"No se pudo afinar la selección con IA: {e}")

    return nombres[mejor_idx], consultas[nombres[mejor_idx]]

# =========================
# Derivación IA
# =========================
DERIVATION_SCHEMA_HINT = """
Devuelve EXCLUSIVAMENTE un JSON válido (sin texto extra) con este esquema:

{
  "select": ["col1", "col2"],
  "filters": [
    {"column": "COL", "op": "contains", "value": "texto"},
    {"column": "COL", "op": "eq", "value": "000123"}
  ],
  "sort": [
    {"column": "COL_ORDEN", "direction": "desc"}
  ],
  "limit": 100
}
"""

def pedir_derivacion_json(df_cols: list, pregunta: str) -> dict:
    if not OPENAI_API_KEY:
        return {}
    try:
        cols_desc = ", ".join(df_cols)
        sys = "Eres un asistente de análisis tabular. Respondes SOLO con JSON válido."
        usr = f"Tienes un DataFrame con columnas: {cols_desc}.\nUsuario: '{pregunta}'.\n{DERIVATION_SCHEMA_HINT}"
        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": usr}],
            temperature=0
        )
        txt = r["choices"][0]["message"]["content"].strip()
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        spec = json.loads(txt[start:end+1])
        if isinstance(spec, dict):
            return spec
        return {}
    except Exception as e:
        st.warning(f"No se pudo derivar JSON con la IA: {e}")
        return {}

# =========================
# Aplicar filtros / sort / limit
# =========================
def aplicar_filtros(df: pd.DataFrame, filters: list, diccionario: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not filters:
        return df
    out = df.copy()
    placeholder = st.empty()
    progreso = placeholder.progress(0)
    total = len(filters)
    for i, f in enumerate(filters, start=1):
        col_user = f.get("column")
        op = (f.get("op") or "").lower()
        value = f.get("value")
        if not col_user or not op:
            continue
        col = resolve_column(col_user, out.columns.tolist(), diccionario)
        if not col:
            continue
        if is_code_column(col):
            out[col] = out[col].astype(str)
        if op == "contains" and isinstance(value, str):
            out = out[out[col].astype(str).str.contains(value, case=False, na=False)]
        elif op == "eq":
            out = out[out[col].astype(str) == str(value)]
        elif op == "neq":
            out = out[out[col].astype(str) != str(value)]
        elif op == "in" and isinstance(value, list):
            out = out[out[col].astype(str).isin(map(str, value))]
        elif op == "startswith" and isinstance(value, str):
            out = out[out[col].astype(str).str.startswith(value, na=False)]
        elif op == "endswith" and isinstance(value, str):
            out = out[out[col].astype(str).str.endswith(value, na=False)]
        elif op == "regex" and isinstance(value, str):
            out = out[out[col].astype(str).str.contains(value, regex=True, na=False)]
        elif op in ("gt", "gte", "lt", "lte"):
            series = pd.to_numeric(out[col], errors="coerce")
            value_num = pd.to_numeric(value, errors="coerce")
            mask = series.notna()
            if op == "gt":
                out = out[mask & (series > value_num)]
            elif op == "gte":
                out = out[mask & (series >= value_num)]
            elif op == "lt":
                out = out[mask & (series < value_num)]
            elif op == "lte":
                out = out[mask & (series <= value_num)]
        elif op == "year_eq":
            sdt = parse_date_col(out[col])
            try:
                y = int(value)
                out = out[sdt.dt.year == y]
            except:
                pass
        elif op == "date_between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                sdt = parse_date_col(out[col])
                try:
                    d1 = pd.to_datetime(value[0], errors="coerce")
                    d2 = pd.to_datetime(value[1], errors="coerce")
                except:
                    continue
                out = out[(sdt >= d1) & (sdt <= d2)]
        progreso.progress(int(i / total * 100))
        time.sleep(0.05)
    placeholder.empty()
    return out

def aplicar_sort_limit(df: pd.DataFrame, sort: list, limit: int = None) -> pd.DataFrame:
    out = df.copy()
    if sort:
        cols = []
        ascend = []
        for s in sort:
            col = s.get("column")
            if col not in out.columns:
                continue
            cols.append(col)
            ascend.append(str(s.get("direction", "asc")).lower() != "desc")
        if cols:
            out = out.sort_values(cols, ascending=ascend)
    if limit and isinstance(limit, int):
        out = out.head(limit)
    return out

# =========================
# Mostrar SQL derivado (versión corregida)
# =========================
def mostrar_sql_derivado(nombre_consulta: str, sql_base: str, spec: dict | None = None):
    """
    Genera y muestra una versión del SQL base con los filtros y orden aplicados por la IA
    (solo visualización, no afecta la ejecución real).
    Compatible con SQL Server (sin usar LIMIT/TOP/FETCH).
    """
    if not sql_base:
        st.info("No hay sentencia SQL para mostrar.")
        return

    if not spec:
        spec = {}

    sql_derivado = sql_base.strip()

    # === Construir cláusulas WHERE ===
    where_clauses = []
    for f in spec.get("filters", []):
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")

        if not col or not op:
            continue

        if isinstance(val, list):
            val_formatted = ", ".join([f"'{v}'" for v in val])
        elif isinstance(val, str):
            val_formatted = f"'{val}'"
        else:
            val_formatted = str(val)

        if op == "contains":
            where_clauses.append(f"{col} LIKE '%' + {val_formatted} + '%'")
        elif op == "startswith":
            where_clauses.append(f"{col} LIKE {val_formatted} + '%'")
        elif op == "endswith":
            where_clauses.append(f"{col} LIKE '%' + {val_formatted}")
        elif op == "in" and isinstance(val, list):
            where_clauses.append(f"{col} IN ({val_formatted})")
        else:
            op_map = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            sql_op = op_map.get(op, op)
            where_clauses.append(f"{col} {sql_op} {val_formatted}")

    # Insertar WHERE antes de GROUP BY / ORDER BY
    if where_clauses:
        match = re.search(r"\b(GROUP\s+BY|ORDER\s+BY)\b", sql_derivado, flags=re.IGNORECASE)
        where_text = "WHERE " + " AND ".join(where_clauses)
        if match:
            pos = match.start()
            sql_derivado = sql_derivado[:pos].rstrip() + "\n" + where_text + "\n" + sql_derivado[pos:].lstrip()
        elif re.search(r"\bWHERE\b", sql_derivado, flags=re.IGNORECASE):
            sql_derivado = re.sub(
                r"(?i)\bWHERE\b",
                "WHERE " + " AND ".join(where_clauses) + " AND (",
                sql_derivado,
                count=1,
            ) + ")"
        else:
            sql_derivado += "\n" + where_text

    # === ORDER BY ===
    sort_clauses = []
    for s in spec.get("sort", []):
        col = s.get("column")
        direction = str(s.get("direction", "asc")).upper()
        if col:
            sort_clauses.append(f"{col} {direction}")
    if sort_clauses:
        if re.search(r"\bORDER\s+BY\b", sql_derivado, flags=re.IGNORECASE):
            sql_derivado = re.sub(r"(?is)\bORDER\s+BY\b.*$", "", sql_derivado).rstrip()
        sql_derivado += "\nORDER BY " + ", ".join(sort_clauses)

    # === Mostrar en Streamlit ===
    with st.expander(f"🧾 SQL generado — {nombre_consulta}"):
        st.code(sql_derivado, language="sql")

# =========================
# Inicialización Streamlit
# =========================
if "df_final" not in st.session_state:
    st.session_state["df_final"] = pd.DataFrame()
if "consulta_sel" not in st.session_state:
    st.session_state["consulta_sel"] = ""
if "spec_aplicada" not in st.session_state:
    st.session_state["spec_aplicada"] = ""
if "pregunta_tmp" not in st.session_state:
    st.session_state["pregunta_tmp"] = ""

# =========================
# Modo oscuro / claro
# =========================
def aplicar_estilos_dark_mode():
    dark_mode = st.sidebar.checkbox("🌙 Modo oscuro", value=False)
    if dark_mode:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #0e1117 !important; color: #fafafa !important; }
            .stMarkdown, .stTextInput label, .stDataFrame, .stExpander { color: #fafafa !important; }
            .stButton>button { background-color: #1f2937 !important; color: #f0f0f0 !important;
                border: 1px solid #4b5563 !important; border-radius: 10px !important; }
            .stButton>button:hover { background-color: #374151 !important; color: #fff !important; }
            .stProgress > div > div > div > div { background-color: #22c55e !important; }
            .stExpander { background-color: #1e1e1e !important; border-radius: 10px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #f8f9fa !important; }
            .stButton>button { background-color: #0078d7 !important; color: white !important;
                border-radius: 10px !important; font-weight: 500 !important; }
            .stButton>button:hover { background-color: #005fa3 !important; color: #fff !important; }
            .stProgress > div > div > div > div { background-color: #0d6efd !important; }
            .stExpander { background-color: #ffffff !important; border-radius: 10px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    return dark_mode

# =========================
# UI
# =========================
dark_mode = aplicar_estilos_dark_mode()

# === Bloque con logo y título en la misma línea ===
col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("logo.png", width=80)
    except Exception as e:
        st.warning(f"No se pudo cargar el logo: {e}")
with col2:
    st.markdown("<h1 style='text-align:left; font-size:2em; font-weight:700;'>🧠 Asistente IA — Lenguaje Natural → SQL</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:left; color:gray;'>Consulta tu base de datos usando lenguaje natural y ayuda de la IA.</p>", unsafe_allow_html=True)

# Expander para Consultas disponibles
with st.expander("🔍 Consultas predefinidas disponibles"):
    consultas = cargar_consultas(SQL_FILE)
    if consultas:
        for c in consultas.keys():
            display_name = c.split("_", 1)[1] if "_" in c else c
            st.write(f"- {display_name}")
    else:
        st.info("No se encontraron consultas en el archivo SQL.")

st.markdown("---")
st.text_input("Escribe tu pregunta:", key="pregunta_tmp")

# Botones Ejecutar y Nueva consulta
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Ejecutar consulta"):
        pregunta = st.session_state["pregunta_tmp"]
        if not pregunta:
            st.warning("Escribe una pregunta.")
        else:
            diccionario = cargar_diccionario(DICT_FILE)
            nombre_consulta, sql_base = mapear_consulta_embeddings(pregunta, consultas)
            st.session_state["consulta_sel"] = nombre_consulta
            if not sql_base:
                st.warning("No se pudo identificar una consulta relevante.")
            else:
                # Ejecutar SQL base original (sin modificaciones)
                df_sql = ejecutar_consulta(sql_base)
                if df_sql.empty:
                    st.warning("La consulta SQL no devolvió resultados.")
                else:
                    # Pedir a la IA la especificación (filters, sort, limit)
                    spec = pedir_derivacion_json(df_sql.columns.tolist(), pregunta)
                    # Aplicar sobre el DataFrame (igual que antes)
                    df_final = aplicar_filtros(df_sql, spec.get("filters"), diccionario)
                    
                    st.session_state["df_final"] = df_final
                    st.session_state["spec_aplicada"] = json.dumps(spec, indent=2)
                    df_final = aplicar_sort_limit(df_final, spec.get("sort"), None)
                    # Mostrar la SQL derivada (visual) basada en spec
                    mostrar_sql_derivado(nombre_consulta, sql_base, spec)

with col2:
    def limpiar_pregunta():
        st.session_state["pregunta_tmp"] = ""
        st.session_state["df_final"] = pd.DataFrame()
        st.session_state["consulta_sel"] = ""
        st.session_state["spec_aplicada"] = ""
    if st.button("🆕 Nueva consulta", on_click=limpiar_pregunta):
        pass

st.markdown("---")
# Mostrar resultados
if not st.session_state["df_final"].empty:
    st.markdown(f"**Consulta seleccionada:** {st.session_state['consulta_sel']}")
    st.dataframe(st.session_state["df_final"], use_container_width=True)
    if st.session_state["spec_aplicada"]:
        with st.expander("📄 Filtros y orden aplicados (JSON derivado IA)"):
            st.code(st.session_state["spec_aplicada"])
