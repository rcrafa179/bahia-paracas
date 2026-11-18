# Caracterización de la bahía de Paracas: meteorología, marea y batimetría

Proyecto del curso de Oceanografía (UNMSM). Scripts en Python, reproducibles, que caracterizan la
bahía de Paracas (Perú) y preparan el forzamiento de marea y meteorología para una futura simulación
con AEM3D. Todas las figuras se rotulan en español (PNG, 300 dpi).

- **Dominio:** 13.60°–13.95°S, 76.15°–76.50°W
- **Periodo meteorológico:** 2025-01-01 a 2026-10-31 (UTC)
- **Toda la configuración está en `config.yaml`**: no hay coordenadas, fechas ni rutas fijas en el código.
- Todo se guarda en **UTC**; la hora local (UTC−5) solo se usa en el ciclo diurno, los eventos Paracas y los gráficos.

## Estructura

```
paracas/
├── config.yaml              # dominio, periodo, estaciones, umbrales, rutas
├── environment.yml          # entorno conda (Python 3.12)
├── data/
│   ├── poligono_bahia.geojson   # polígono editable de la bahía (sí va al repositorio)
│   ├── raw/                 # descargas (no van al repositorio)
│   └── processed/           # resultados intermedios (CSV, NetCDF, YAML)
├── figures/                 # figuras PNG
├── presentacion/            # presentación Beamer
└── scripts/
    ├── 00_fes2022_download.py
    ├── 01_era5_download.py        06_tide_harmonic_analysis.py
    ├── 02_era5_process.py         07_tide_model_pytmd.py
    ├── 03_metar_download.py       08_bathymetry_gebco.py
    ├── 04_meteo_validation.py     09_presentacion.py
    ├── 05_tide_gauge_download.py
    └── utils/  (config, logs, gráficos, viento, ERA5, verificaciones de sanidad)
```

## 1. Entorno

En la máquina original se reutiliza el entorno existente **`taller-rhia`** (Python 3.12), al que solo
se le agregaron tres paquetes (sin cambiar ninguna versión existente):

```bash
mamba install -n taller-rhia -c conda-forge windrose utide pytmd
```

En otra máquina:

```bash
mamba env create -f environment.yml
```

Todos los scripts se ejecutan desde la raíz del repositorio, por ejemplo:

```bash
conda activate taller-rhia
python scripts/01_era5_download.py --help
```

Argumentos comunes: `--config` (otro config.yaml), `--inicio/--fin AAAA-MM-DD` (periodo corto de prueba)
y `--forzar` (repetir descargas).

## 2. Lo que hay que descargar o configurar a mano

| Qué | Dónde | Notas |
|---|---|---|
| **Credenciales del CDS** | https://cds.climate.copernicus.eu → perfil → *API token* | Archivo `~/.cdsapirc` con `url: https://cds.climate.copernicus.eu/api` y `key: <token>`. **Nunca** dentro del repositorio. Requiere `cdsapi >= 0.7.7`. |
| **Licencia de ERA5** | Página del dataset *ERA5 hourly data on single levels*, pestaña *Download*, al final | Hay que aceptarla una vez; si no, el CDS rechaza la petición. |
| **FES2022b** (marea global) | Registro gratuito en AVISO+: https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html (pedir acceso a *FES – Oceanic Tides Heights*) | Con la cuenta habilitada, crear `~/.netrc` (fuera del repositorio) con la línea `machine ftp-access.aviso.altimetry.fr login TU_CORREO password TU_CONTRASEÑA` y hacer `chmod 600 ~/.netrc`. Luego `00_fes2022_download.py` baja **solo** las 15 constituyentes de `modelo_marea.constituyentes` (de 34) de `fes2022b/ocean_tide_extrapolated/` y las descomprime (`.xz`) en `data/raw/fes2022/`. El FTP es lento (~15 min por archivo) y a veces corta: el script reconecta y reanuda desde el último byte (`modelo_marea.descarga.reintentos`), y si se interrumpe basta volver a correrlo. Ocupa varios GB; el script revisa antes el espacio libre. |
| **GEBCO** (batimetría) | https://download.gebco.net | Usar la herramienta de **recorte** (no la malla global de ~7,5 GB): caja 13.55–14.00°S, 76.10–76.55°W, formato *2D netCDF*. Guardar como `data/raw/gebco/gebco_paracas.nc` (o cambiar `batimetria.gebco_nc`). |
| Sondajes DHN (opcional) | Carta náutica de la DHN, digitalizada a mano | CSV con columnas `lon, lat, prof` (prof en m, positiva). Se fusiona con `08_bathymetry_gebco.py --sondajes archivo.csv`. |

Los datos METAR (IEM), del mareógrafo (IOC y UHSLC) y ERA5 los descargan los scripts.

## 3. Orden de ejecución

Primero conviene probar con un mes y después correr el periodo completo.

```bash
# --- Meteorología ---
python scripts/01_era5_download.py --inicio 2025-01-01 --fin 2025-01-31   # prueba (~6 min por mes en la cola del CDS)
python scripts/01_era5_download.py                                        # periodo completo (omite meses ya bajados)
python scripts/02_era5_process.py           # viento, HR, radiación, figuras, eventos Paracas (y mes candidato si fuente = era5)
python scripts/03_metar_download.py         # METAR de SPSO con control de calidad
python scripts/04_meteo_validation.py       # ERA5 vs SPSO (y mes candidato si fuente = metar)

# --- Marea ---
python scripts/00_fes2022_download.py --listar   # comprueba la cuenta AVISO y muestra tamaños
python scripts/00_fes2022_download.py            # baja FES2022b (15 constituyentes); omite las ya bajadas y reanuda cortes
python scripts/05_tide_gauge_download.py    # IOC (2025–26) + UHSLC (referencia), QC y selección
python scripts/06_tide_harmonic_analysis.py # UTide: constituyentes, F, rangos, residuo
python scripts/07_tide_model_pytmd.py       # FES2022: armónicos, fronteras y serie de 15 min (necesita FES2022b)

# --- Batimetría ---
python scripts/08_bathymetry_gebco.py       # necesita el recorte de GEBCO

# --- Presentación ---
python scripts/09_presentacion.py           # escribe las cifras y compila presentacion/paracas.pdf
```

`02` y `04` deben correr antes que `07`, porque `07` usa el mes candidato (`data/processed/mes_candidato.yaml`).

## 4. Fuentes de datos verificadas

| Fuente | Identificador | Verificación |
|---|---|---|
| FES2022b (AVISO) | `fes2022b/ocean_tide_extrapolated` | FTP autenticado `ftp-access.aviso.altimetry.fr`, ruta `auxiliary/tide_model/fes2022b/ocean_tide_extrapolated` (página de AVISO y `pyTMD.datasets.fetch_aviso_fes`). NetCDF4 comprimido con xz; variables `amplitude` y `phase`. |
| ERA5 horario, niveles únicos | `reanalysis-era5-single-levels` | Sintaxis actual del CDS (`data_format`, `download_format`). El CDS entrega un ZIP con dos NetCDF (variables instantáneas y acumuladas); `01` los une. |
| METAR Pisco | ICAO **SPSO**, IEM ASOS | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`; SPSO en 13.7449°S, 76.2203°W. |
| Mareógrafo IOC | `pisco` (radar), `pisc2` (presión) | `https://www.ioc-sealevelmonitoring.org/service.php?query=data`; 13.8194°S, 76.2519°W; tiempo real **sin control de calidad**. |
| Mareógrafo UHSLC | **683** (Pisco), *research quality* | `h683b.csv` (1991–2023-04); no hay *fast delivery* para Pisco. Su metadato de posición (13.42°S, 76.13°W) cae en Chincha, se ignora. |

## 5. Problemas encontrados en los datos (importante)

- **ERA5 subestima los vientos Paracas.** En 2025-01 → 2026-10, SPSO registró **101 días** con viento
  > 10 m/s por la tarde y ERA5 **ninguno** (p95 vespertino 10,3 vs 6,9 m/s; en las 287 horas con
  observado > 10 m/s, ERA5 da 6,4 m/s frente a 11,0 m/s). ERA5 sí reproduce la dirección S–SO, el ciclo
  diurno y la temperatura (r = 0,92). Por eso el mes candidato se define con el METAR
  (`vientos_paracas.fuente: metar`). Con ERA5 saldría 2025-08 por el viento vespertino más intenso,
  pero en SPSO ese mes tuvo solo 3 eventos.
- **Mes candidato: enero de 2025**, fijado con `vientos_paracas.mes_candidato: "2025-01"`. En SPSO empata
  con marzo de 2025 (10 eventos cada uno; el desempate automático daría marzo). Se eligió enero porque es
  el único mes en que el mareógrafo IOC pasa el control de calidad, así que la marea predicha para el
  forzamiento se puede comparar con la observada. Con `mes_candidato: auto` lo elige el script.
- **El forzamiento meteorológico sigue siendo ERA5** (punto de la boca), con el viento subestimado en los
  eventos. Falta decidir si se escala con la razón observado/ERA5 o se usa el viento de SPSO.
- **El mareógrafo IOC 2025–26 no sirve para el análisis armónico.** `05` compara cada mes con la marea
  predicha a partir de UHSLC 2022:
  - `pisco` solo es bueno en enero 2025. Desde agosto 2025 registra amplitudes ~27 % menores
    (pendiente ≈ 0,73), aunque su fase es correcta (r ≈ 0,99).
  - `pisc2` es ruidosa (r ≈ 0,86).

  Con `analisis_marea.serie_principal: auto`, el análisis armónico principal se hace con **UHSLC 2022**
  (99,8 % completo) y el IOC de enero 2025 queda como verificación independiente.

## 6. AEM3D

**No se corre ninguna simulación.** Quedan listos, en CSV y en UTC:
- `data/processed/forzamiento_meteo_mes_candidato.csv`: meteorología horaria del mes candidato.
- `data/processed/marea_frontera_mes_candidato.csv`: nivel del mar cada 15 min en el mareógrafo y en
  cada punto de frontera abierta.

La función `exportar_aem3d()` de `07_tide_model_pytmd.py` está vacía y documentada a propósito: el
formato de entrada de AEM3D se implementará cuando haya una plantilla del curso.

## 7. Espacio en disco

Medido al terminar (periodo completo, sin FES2022b ni GEBCO):

| Componente | Tamaño |
|---|---|
| ERA5 (22 meses, 5×6 puntos) | 9,4 MB |
| IOC (1 min, comprimido) + UHSLC + METAR | 9,8 MB |
| `data/processed/` | 13 MB |
| `figures/` | 10 MB |
| `presentacion/` (PDF incluido) | 8,2 MB |
| **Total del repositorio** | **≈ 51 MB** |
| GEBCO recortado | < 1 MB |
| FES2022b (solo elevaciones, ~15 constituyentes) | varios GB (lo más pesado); `07` guarda un recorte regional de pocos MB, después se pueden borrar los globales |
