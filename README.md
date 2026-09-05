# Sports Predictor AI

Aplicación web desarrollada con Streamlit para cargar datos históricos deportivos,
entrenar un modelo de clasificación y generar estimaciones probabilísticas.

## Funcionalidades

- Carga de archivos CSV, XLSX y XLS.
- Selección de variable objetivo y variables predictoras numéricas.
- Modelos Random Forest y Regresión Logística.
- Métricas de clasificación.
- Importancia de variables para Random Forest.
- Confianza estimada de las predicciones.
- Cálculo exploratorio de Value Betting.
- Kelly Criterion fraccional.
- Exportación de predicciones a CSV con codificación UTF-8 BOM.

## Estructura

```text
sports-predictor-ai/
├── app.py
├── requirements.txt
└── README.md
```

## Instalación local

Se recomienda Python 3.11 o 3.12.

```bash
python -m venv .venv
```

### Activación en Windows

```bash
.venv\Scripts\activate
```

### Activación en macOS o Linux

```bash
source .venv/bin/activate
```

Instale las dependencias:

```bash
pip install -r requirements.txt
```

Ejecute la aplicación:

```bash
streamlit run app.py
```

Streamlit mostrará en la terminal la dirección local, normalmente
`http://localhost:8501`.

## Dataset esperado

El archivo debe contener:

1. Una columna objetivo con al menos dos clases, por ejemplo: `Local`, `Empate` y
   `Visitante`.
2. Una o más columnas numéricas predictoras.
3. Al menos 10 registros completos, aunque se recomienda usar muchos más.

La aplicación no incluye datos de ejemplo ni depende de rutas locales fijas.

## Despliegue en Streamlit Community Cloud

1. Cree un repositorio en GitHub.
2. Suba `app.py`, `requirements.txt` y `README.md` a la raíz.
3. Ingrese a Streamlit Community Cloud.
4. Conecte su cuenta de GitHub.
5. Seleccione el repositorio, la rama y `app.py` como archivo principal.
6. Inicie el despliegue.

## Limitaciones

- Este MVP solo utiliza variables predictoras numéricas.
- Las métricas dependen de la calidad, cantidad y representatividad de los datos.
- No obtiene partidos ni cuotas en tiempo real.
- No almacena historiales ni modelos después de finalizar la sesión.
- El cálculo de valor usa una cuota uniforme introducida manualmente.

## Uso responsable

Las predicciones son estimaciones probabilísticas y no garantizan resultados. La
aplicación no constituye asesoría financiera. Las apuestas implican riesgo de pérdida
de dinero. Cumpla la legislación y los requisitos de edad aplicables en su jurisdicción.
