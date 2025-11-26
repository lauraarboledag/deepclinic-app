"""
🏥 DeepClinic - Generador de Citas Médicas Sintéticas
Aplicación con IA Generativa usando TVAE
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime, timedelta
import random


# CONFIGURACIÓN DE LA PÁGINA

st.set_page_config(
    page_title="DeepClinic - Generador de Citas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ESTILOS CSS PERSONALIZADOS

st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e40af;
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .info-box {
        padding: 1.25rem;
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        color: #1e40af;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    .success-box {
        padding: 1.25rem;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        color: #166534;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        font-size: 1.75rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .icon-text {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# FUNCIONES DE CARGA


@st.cache_resource
def load_model():
    """Cargar modelo TVAE entrenado"""
    try:
        model = joblib.load('tvae_model_exp2.pkl')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def load_real_data():
    """Cargar datos reales"""
    try:
        df = pd.read_csv('appointments_real.csv')
        return df
    except Exception as e:
        st.error(f"Error al cargar datos reales: {e}")
        return None

@st.cache_data
def load_comparison_report():
    """Cargar reporte de comparación"""
    try:
        with open('comparison_report.json', 'r') as f:
            report = json.load(f)
        return report
    except Exception as e:
        st.warning(f"No se pudo cargar el reporte de comparación: {e}")
        return None


# FUNCIONES DE GENERACIÓN


def generate_synthetic_data(model, num_samples):
    """Generar datos sintéticos usando el modelo"""
    try:
        synthetic_data = model.sample(num_samples)
        
        # Postprocesamiento
        synthetic_data = postprocess_data(synthetic_data)
        
        # Agregar patient_ids
        base_id = 50000 + random.randint(0, 10000)
        synthetic_data['patient_id'] = [f"P{base_id + i}" for i in range(len(synthetic_data))]
        
        # Reordenar columnas
        cols = ['patient_id', 'age', 'gender', 'specialty', 'urgency', 
                'preferred_day', 'preferred_slot', 'duration_min', 
                'previous_no_shows', 'distance_km', 'no_show_prob']
        synthetic_data = synthetic_data[cols]
        
        return synthetic_data
    except Exception as e:
        st.error(f"Error al generar datos: {e}")
        return None

def postprocess_data(df):
    """Postprocesar datos sintéticos"""
    df = df.copy()
    
    df['age'] = df['age'].clip(0, 100).round().astype(int)
    df['distance_km'] = df['distance_km'].clip(0).round(3)
    df['previous_no_shows'] = df['previous_no_shows'].clip(0).round().astype(int)
    df['no_show_prob'] = df['no_show_prob'].clip(0, 1).round(6)
    
    # Asegurar duration_min sea 15, 30 o 45
    valid_durations = [15, 30, 45]
    df['duration_min'] = df['duration_min'].apply(
        lambda x: min(valid_durations, key=lambda v: abs(v - x))
    )
    
    return df

def autocomplete_appointment(model, partial_data):
    """Autocompletar cita usando el modelo TVAE"""
    try:
        # Generar una muestra del modelo
        sample = model.sample(1)
        sample = postprocess_data(sample)
        
        # Sobrescribir con datos del usuario
        for key, value in partial_data.items():
            if value is not None:
                sample[key] = value
        
        return sample.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error al autocompletar: {e}")
        return None


# FUNCIONES DE VISUALIZACIÓN


def create_comparison_plot(real_df, synthetic_df, column, plot_type='histogram'):
    """Crear gráfico comparativo"""
    if plot_type == 'histogram':
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=real_df[column],
            name='Datos Reales',
            opacity=0.6,
            marker_color='steelblue'
        ))
        fig.add_trace(go.Histogram(
            x=synthetic_df[column],
            name='Datos Sintéticos',
            opacity=0.6,
            marker_color='coral'
        ))
        fig.update_layout(
            title=f'Comparación: {column}',
            xaxis_title=column,
            yaxis_title='Frecuencia',
            barmode='overlay',
            height=400
        )
    else:  # bar chart para categóricas
        real_counts = real_df[column].value_counts()
        synth_counts = synthetic_df[column].value_counts()
        
        categories = sorted(set(real_counts.index) | set(synth_counts.index))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=[real_counts.get(c, 0) for c in categories],
            name='Datos Reales',
            marker_color='steelblue'
        ))
        fig.add_trace(go.Bar(
            x=categories,
            y=[synth_counts.get(c, 0) for c in categories],
            name='Datos Sintéticos',
            marker_color='coral'
        ))
        fig.update_layout(
            title=f'Comparación: {column}',
            xaxis_title=column,
            yaxis_title='Frecuencia',
            barmode='group',
            height=400
        )
    
    return fig

def create_metrics_dashboard(df):
    """Crear dashboard de métricas"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total de Citas",
            value=len(df)
        )
    
    with col2:
        avg_age = df['age'].mean()
        st.metric(
            label="👥 Edad Promedio",
            value=f"{avg_age:.1f} años"
        )
    
    with col3:
        urgent_pct = (df['urgency'] == 'urgente').sum() / len(df) * 100
        st.metric(
            label="🚨 Citas Urgentes",
            value=f"{urgent_pct:.1f}%"
        )
    
    with col4:
        avg_distance = df['distance_km'].mean()
        st.metric(
            label="📍 Distancia Promedio",
            value=f"{avg_distance:.2f} km"
        )


# INTERFAZ PRINCIPAL


def main():
    # Header
    st.markdown('''
    <div class="main-header">
        <span class="material-symbols-outlined" style="font-size: 3rem; vertical-align: middle;">local_hospital</span>
        DeepClinic - Generador de Citas Médicas
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <span class="material-symbols-outlined">info</span>
    <strong>Sobre esta aplicación:</strong><br>
    Sistema de generación de datos sintéticos de citas médicas usando <strong>TVAE (Variational Autoencoder)</strong>.
    Los datos generados mantienen las distribuciones estadísticas de los datos reales mientras preservan la privacidad.
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo y datos
    model = load_model()
    real_data = load_real_data()
    report = load_comparison_report()
    
    if model is None or real_data is None:
        st.error("⚠️ Error al cargar los recursos necesarios. Verifica que los archivos existan.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Panel de Control")
    
    page = st.sidebar.radio(
        "Navegación:",
        ["Generador", "Registrar Cita", "Análisis Comparativo", "Métricas del Modelo"],
        format_func=lambda x: {
            "Generador": "🎲 Generador",
            "Registrar Cita": "➕ Registrar Cita", 
            "Análisis Comparativo": "📊 Análisis",
            "Métricas del Modelo": "📈 Métricas"
        }[x]
    )
    
    # ========================================================================
    # PÁGINA 1: GENERADOR
    # ========================================================================
    if page == "Generador":
        st.markdown('<div class="section-header"><span class="material-symbols-outlined">casino</span>Generador de Citas Sintéticas</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parámetros de Generación")
            
            num_samples = st.slider(
                "Número de citas a generar:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
            
            show_comparison = st.checkbox("Mostrar comparación con datos reales", value=True)
            
            if st.button("🎲 Generar Citas", type="primary"):
                with st.spinner("Generando datos sintéticos..."):
                    synthetic_data = generate_synthetic_data(model, num_samples)
                    
                    if synthetic_data is not None:
                        st.session_state['synthetic_data'] = synthetic_data
                        st.markdown("""
                        <div class="success-box">
                        <span class="material-symbols-outlined">check_circle</span>
                        <strong>¡Éxito!</strong> Se generaron {} citas sintéticas correctamente.
                        </div>
                        """.format(len(synthetic_data)), unsafe_allow_html=True)
        
        with col2:
            if 'synthetic_data' in st.session_state:
                st.subheader("Datos Generados")
                synthetic_data = st.session_state['synthetic_data']
                
                # Métricas
                create_metrics_dashboard(synthetic_data)
                
                # Mostrar tabla
                st.dataframe(synthetic_data, use_container_width=True, height=400)
                
                # Descargar
                csv = synthetic_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name=f"citas_sinteticas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Comparación
        if show_comparison and 'synthetic_data' in st.session_state:
            st.markdown("---")
            st.markdown('<div class="section-header"><span class="material-symbols-outlined">compare</span>Comparación con Datos Reales</div>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Variables Numéricas", "Variables Categóricas"])
            
            with tab1:
                num_cols = ['age', 'distance_km', 'no_show_prob', 'previous_no_shows']
                for i in range(0, len(num_cols), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(num_cols[i:i+2]):
                        with cols[j]:
                            fig = create_comparison_plot(real_data, synthetic_data, col, 'histogram')
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                cat_cols = ['gender', 'specialty', 'urgency', 'preferred_slot']
                for i in range(0, len(cat_cols), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cat_cols[i:i+2]):
                        with cols[j]:
                            fig = create_comparison_plot(real_data, synthetic_data, col, 'bar')
                            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PÁGINA 2: REGISTRAR CITA
    # ========================================================================
    elif page == "Registrar Cita":
        st.markdown('<div class="section-header"><span class="material-symbols-outlined">add_circle</span>Registrar Nueva Cita Médica</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <span class="material-symbols-outlined">lightbulb</span>
        <strong>Cómo funciona:</strong><br>
        Completa los campos que conozcas. Puedes usar el botón <strong>"Completar con IA"</strong> 
        para que el modelo TVAE prediga automáticamente los campos faltantes basándose en patrones aprendidos.
        </div>
        """, unsafe_allow_html=True)
        
        # Inicializar session state para el formulario
        if 'form_data' not in st.session_state:
            st.session_state['form_data'] = {}
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Formulario de Registro")
            
            # Formulario
            with st.form("appointment_form"):
                fcol1, fcol2 = st.columns(2)
                
                with fcol1:
                    age = st.number_input("Edad del paciente", min_value=0, max_value=100, value=35, step=1)
                    
                    gender = st.selectbox(
                        "Género",
                        options=["M", "F", "O"],
                        format_func=lambda x: {"M": "Masculino", "F": "Femenino", "O": "Otro"}[x]
                    )
                    
                    specialty = st.selectbox(
                        "Especialidad médica",
                        options=["Medicina General", "Pediatría", "Ginecología", "Odontología", "Psicología", "Cardiología"]
                    )
                    
                    urgency = st.selectbox(
                        "Tipo de cita",
                        options=["normal", "urgente"],
                        format_func=lambda x: "Normal" if x == "normal" else "Urgente"
                    )
                
                with fcol2:
                    preferred_day = st.selectbox(
                        "Día preferido",
                        options=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
                    )
                    
                    preferred_slot = st.selectbox(
                        "Horario preferido",
                        options=["mañana", "tarde", "noche"],
                        format_func=lambda x: x.capitalize()
                    )
                    
                    duration_min = st.selectbox(
                        "Duración estimada",
                        options=[15, 30, 45],
                        format_func=lambda x: f"{x} minutos"
                    )
                    
                    distance_km = st.number_input(
                        "Distancia al centro médico (km)",
                        min_value=0.0,
                        max_value=100.0,
                        value=5.0,
                        step=0.5
                    )
                
                st.markdown("---")
                
                fcol3, fcol4 = st.columns(2)
                
                with fcol3:
                    previous_no_shows = st.number_input(
                        "Número de ausencias previas",
                        min_value=0,
                        max_value=20,
                        value=0,
                        step=1,
                        help="Veces que el paciente no asistió a citas anteriores"
                    )
                
                with fcol4:
                    # Calcular probabilidad basada en inputs (fórmula del dataset original)
                    calculated_prob = np.clip(
                        0.05 + 0.005*previous_no_shows + 0.001*distance_km + 
                        (0.02 if age < 18 else 0) + (-0.02 if urgency == "urgente" else 0),
                        0.01, 0.6
                    )
                    
                    no_show_prob = st.number_input(
                        "Probabilidad de no-show",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(calculated_prob),
                        step=0.01,
                        help="Probabilidad de que el paciente no asista"
                    )
                
                st.markdown("---")
                
                submitted = st.form_submit_button("💾 Guardar Cita", use_container_width=True, type="primary")
                
                if submitted:
                    # Crear entrada
                    new_appointment = {
                        'patient_id': f"P{90000 + random.randint(0, 9999)}",
                        'age': age,
                        'gender': gender,
                        'specialty': specialty,
                        'urgency': urgency,
                        'preferred_day': preferred_day,
                        'preferred_slot': preferred_slot,
                        'duration_min': duration_min,
                        'previous_no_shows': previous_no_shows,
                        'distance_km': distance_km,
                        'no_show_prob': round(no_show_prob, 6)
                    }
                    
                    # Guardar en session state
                    if 'registered_appointments' not in st.session_state:
                        st.session_state['registered_appointments'] = []
                    
                    st.session_state['registered_appointments'].append(new_appointment)
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <span class="material-symbols-outlined">check_circle</span>
                    <strong>¡Cita registrada!</strong> ID: {new_appointment['patient_id']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
        
        with col2:
            st.subheader("Asistente IA")
            
            st.markdown("""
            <div class="metric-card" style="color: #1e293b;">
            <span class="material-symbols-outlined" style="color: #3b82f6;">smart_toy</span>
            <strong style="color: #1e293b;">Autocompletar con IA</strong><br><br>
            <span style="color: #475569;">El modelo TVAE puede predecir valores faltantes basándose en los datos que ya ingresaste.</span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🤖 Completar con IA", use_container_width=True):
                with st.spinner("Consultando al modelo..."):
                    # Crear datos parciales
                    partial_data = {
                        'age': age if 'age' in locals() else None,
                        'gender': gender if 'gender' in locals() else None,
                        'specialty': specialty if 'specialty' in locals() else None,
                    }
                    
                    # Autocompletar
                    completed = autocomplete_appointment(model, partial_data)
                    
                    if completed:
                        st.markdown("""
                        <div class="success-box">
                        <span class="material-symbols-outlined">auto_awesome</span>
                        <strong>Campos sugeridos por IA</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.json({
                            "urgency": completed['urgency'],
                            "preferred_day": completed['preferred_day'],
                            "preferred_slot": completed['preferred_slot'],
                            "duration_min": int(completed['duration_min']),
                            "distance_km": float(completed['distance_km']),
                            "no_show_prob": float(completed['no_show_prob'])
                        })
                        st.info("💡 Puedes copiar estos valores al formulario")
            
            # Mostrar estadísticas
            if 'registered_appointments' in st.session_state and len(st.session_state['registered_appointments']) > 0:
                st.markdown("---")
                st.markdown("### Resumen")
                st.metric("Total de citas registradas", len(st.session_state['registered_appointments']))
                
                # Botón para descargar
                df_registered = pd.DataFrame(st.session_state['registered_appointments'])
                csv = df_registered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar Todas las Citas",
                    data=csv,
                    file_name=f"citas_registradas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                if st.button("🗑️ Limpiar Registros", use_container_width=True):
                    st.session_state['registered_appointments'] = []
                    st.rerun()
        
        # Mostrar tabla de citas registradas
        if 'registered_appointments' in st.session_state and len(st.session_state['registered_appointments']) > 0:
            st.markdown("---")
            st.subheader("Citas Registradas")
            df_registered = pd.DataFrame(st.session_state['registered_appointments'])
            st.dataframe(df_registered, use_container_width=True, height=300)
    
    # ========================================================================
    # PÁGINA 3: ANÁLISIS COMPARATIVO
    # ========================================================================
    elif page == "Análisis Comparativo":
        st.markdown('<div class="section-header"><span class="material-symbols-outlined">analytics</span>Análisis Comparativo de Datos</div>', unsafe_allow_html=True)
        
        st.subheader("Datos Reales vs Datos Sintéticos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Datos Reales")
            create_metrics_dashboard(real_data)
            st.dataframe(real_data.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### Datos Sintéticos (Muestra)")
            try:
                synthetic_sample = pd.read_csv('appointments_synthetic_exp2_tvae_clean.csv')
                create_metrics_dashboard(synthetic_sample)
                st.dataframe(synthetic_sample.head(10), use_container_width=True)
            except:
                st.info("Genera datos en la sección 'Generador' para ver comparaciones")
        
        # Distribuciones
        st.markdown("---")
        st.subheader("Distribuciones Estadísticas")
        
        try:
            synthetic_sample = pd.read_csv('appointments_synthetic_exp2_tvae_clean.csv')
            
            selected_var = st.selectbox(
                "Selecciona una variable para analizar:",
                ['age', 'distance_km', 'no_show_prob', 'gender', 'specialty', 'urgency']
            )
            
            if selected_var in ['age', 'distance_km', 'no_show_prob']:
                fig = create_comparison_plot(real_data, synthetic_sample, selected_var, 'histogram')
            else:
                fig = create_comparison_plot(real_data, synthetic_sample, selected_var, 'bar')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas descriptivas
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Estadísticas - Datos Reales:**")
                st.write(real_data[selected_var].describe())
            with col2:
                st.markdown("**Estadísticas - Datos Sintéticos:**")
                st.write(synthetic_sample[selected_var].describe())
        
        except Exception as e:
            st.warning(f"No se pudieron cargar los datos sintéticos de muestra: {e}")
    
    # ========================================================================
    # PÁGINA 4: MÉTRICAS DEL MODELO
    # ========================================================================
    elif page == "Métricas del Modelo":
        st.markdown('<div class="section-header"><span class="material-symbols-outlined">bar_chart</span>Métricas de Evaluación del Modelo</div>', unsafe_allow_html=True)
        
        if report is not None:
            st.subheader("Comparación de Experimentos")
            
            # Tabla comparativa
            exp1 = report['experiment_1']['overall']
            exp2 = report['experiment_2']['overall']
            
            comparison_df = pd.DataFrame({
                'Métrica': ['KS Score (%)', 'Chi² Score (%)', 'AUC Score (%)', 'Score Total (%)'],
                'CTGAN': [exp1['ks_score'], exp1['chi2_score'], exp1['auc_score'], exp1['total_score']],
                'TVAE (Usado)': [exp2['ks_score'], exp2['chi2_score'], exp2['auc_score'], exp2['total_score']]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Gráfico de barras
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='CTGAN',
                x=comparison_df['Métrica'],
                y=comparison_df['CTGAN'],
                marker_color='steelblue'
            ))
            fig.add_trace(go.Bar(
                name='TVAE',
                x=comparison_df['Métrica'],
                y=comparison_df['TVAE (Usado)'],
                marker_color='coral'
            ))
            fig.update_layout(
                title='Comparación de Modelos',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Ganador
            winner = report['comparison']['winner']
            st.markdown(f"""
            <div class="success-box">
            <span class="material-symbols-outlined">emoji_events</span>
            <strong>Modelo Seleccionado:</strong> {winner}<br>
            <strong>Score Final:</strong> {exp2['total_score']:.1f}%<br>
            <strong>Calidad:</strong> {exp2['quality']}
            </div>
            """, unsafe_allow_html=True)
            
            # Detalles técnicos
            with st.expander("Ver Detalles Técnicos"):
                st.json(report)
        
        else:
            st.info("No se encontró el reporte de comparación.")
            st.markdown("""
            **Métricas de evaluación utilizadas:**
            - **Test Kolmogorov-Smirnov (KS):** Compara distribuciones continuas
            - **Test Chi-cuadrado (χ²):** Compara distribuciones categóricas
            - **AUC (Two-Sample Test):** Mide qué tan distinguibles son los datos sintéticos de los reales
            
            **Objetivo:** AUC cercano a 0.5 indica que los datos sintéticos son indistinguibles de los reales.
            """)


# EJECUTAR APLICACIÓN

if __name__ == "__main__":
    main()