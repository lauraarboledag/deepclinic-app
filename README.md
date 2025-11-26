DeepClinic - Generador de Citas Médicas Sintéticas

Sistema de generación de datos sintéticos de citas médicas usando **TVAE (Variational Autoencoder)**.

Descripción

La finalidad de esta aplicación web se centra en generar datos sintéticos de citas médicas preservando las características estadísticas de datos reales mientras protege la privacidad de pacientes.

Instalación Local

Requisitos
- Python 3.8 o superior

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/lauraarboledag/deepclinic-app.git
cd deepclinic-app
```
Una vez clonado el repositorio, en terminarl con ctrl + ñ vamos a instalar las librerías necesarias para su uso

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
Nos aseguramos de estar en la carpeta cd deepclinic, luego ejecutamos la aplicación:

3. Ejecutar la aplicación:
```bash
streamlit run src/app_streamlit.py
```
Cuando se ejecute se abrirá nuestro navegador de preferencia con el siguiente enlace:

4. **Abrir en el navegador:**
```
http://localhost:8501
```

Ya con eso se puede explorar la  aplicación y sus funcionalidades
