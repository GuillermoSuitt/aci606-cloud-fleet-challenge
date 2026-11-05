# ACI606 · Cloud Fleet Challenge

Servicio Cloud que recibe telemetría de una flota virtual de robots, la guarda de forma persistente, predice la condición de cada robot con un modelo de Machine Learning y la muestra en un panel de operación.

## 🌐 Servicio desplegado

| | |
|---|---|
| **URL pública** | https://aci606-cloud-fleet-challenge-52do.onrender.com |
| **Panel** | https://aci606-cloud-fleet-challenge-52do.onrender.com/dashboard |
| **Documentación interactiva** | https://aci606-cloud-fleet-challenge-52do.onrender.com/docs |

> El plan gratuito de Render suspende el servicio tras 15 minutos sin tráfico. La primera solicitud puede tardar cerca de 1 minuto mientras se reactiva; para despertarlo, abrir `/health`.

## 🏗️ Arquitectura

```
Simulador / Navegador / Evaluador
            │  HTTPS
            ▼
Render · Web Service (FastAPI + Uvicorn + models/model.joblib)
            │  SQL con TLS
            ▼
Neon · PostgreSQL (tabla telemetry)
```

- **Plataforma (PaaS):** Render, Web Service, plan Free, región Virginia (US East).
- **Base de datos:** PostgreSQL gestionado en Neon, región São Paulo, plan Free (no expira).
- **Despliegue automático:** cada `push` a la rama `main` de GitHub redespliega el servicio.

**¿Por qué Neon y no SQLite?** El disco de Render es efímero: un archivo SQLite se borraría en cada reinicio. Neon mantiene los datos fuera del servicio, así que el historial sobrevive a reinicios y redespliegues.

## 📍 Ubicación de los datos

Toda la telemetría se guarda en la tabla `telemetry` de PostgreSQL en Neon (una fila por muestra, con índice en `robot_id`). El servicio se conecta mediante la variable de entorno `DATABASE_URL`; la cadena de conexión **no** está en el repositorio.

## 🔌 Endpoints

| Ruta | Función |
|---|---|
| `GET /health` | Estado del servicio y del modelo |
| `POST /telemetry` | Guarda una muestra en PostgreSQL |
| `GET /robots` | Última muestra de cada robot |
| `GET /robots/{id}/history` | Historial completo de un robot |
| `POST /predict` | Condición predicha (NORMAL / WARNING / CRITICAL) |
| `GET /metrics` | Solicitudes, errores 4xx/5xx, latencia p50/p95 y uptime |
| `GET /dashboard` | Panel del operador |

Los datos inválidos (campo faltante, fuera de rango o NaN) se rechazan con **422**.

## 🤖 Modelo

- Datos: 6 flotas generadas con `simulator/generate_dataset.py` (semillas 1 a 6, 18.000 muestras).
- Validación: separación entrenamiento/prueba **por flota** (`GroupShuffleSplit`), para imitar una flota no vista.
- Modelo final: **Random Forest de 100 árboles** (F1 macro 0,907 en flotas de prueba; línea base 0,290).
- Evidencia completa en [`docs/model_report.md`](docs/model_report.md).

Para reentrenar:
```bash
python simulator/generate_dataset.py --samples 3000 --seed 1 --output data/fleet_1.csv
# ... repetir con --seed 2 a 6 (fleet_2.csv ... fleet_6.csv)
python ml/train_model.py
```

## 🚀 Procedimiento de despliegue (Render + Neon)

1. Crear un proyecto en **Neon** y copiar la cadena de conexión, cambiando `postgresql://` por `postgresql+psycopg://`.
2. En **Render**: New → Web Service → conectar este repositorio, rama `main`.
3. Configurar:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Variables de entorno:

| Variable | Valor |
|---|---|
| `DATABASE_URL` | cadena de Neon (`postgresql+psycopg://...?sslmode=require`) |
| `MODEL_PATH` | `models/model.joblib` |
| `PYTHON_VERSION` | `3.13.0` |

5. Al arrancar, el servicio crea la tabla si no existe y carga el modelo una sola vez.

## 💻 Ejecución local

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# PowerShell
$env:DATABASE_URL="postgresql+psycopg://..."
uvicorn app.main:app --reload
```

Simulador (otra terminal):
```bash
python simulator/fleet_simulator.py --api http://127.0.0.1:8000
```

## ✅ Verificación

```bash
python tests/smoke_test.py --api https://aci606-cloud-fleet-challenge-52do.onrender.com
```

Prueba de persistencia: enviar telemetría con el simulador, reiniciar el servicio desde Render y comprobar que `/robots/{id}/history` conserva todos los registros.

## 🌿 Flujo de trabajo en Git

El desarrollo se hizo en la rama `feature/persistence` y se integró a `main` mediante merge.

## 👥 Integrantes

- Benjamín Silva
- Guillermo Suitt
- Emilio Fernandez
