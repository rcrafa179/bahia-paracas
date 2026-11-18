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
    │                              07b_control_vecinos.py
    ├── 03_metar_download.py       08_bathymetry_gebco.py
    ├── 04_meteo_validation.py     09_presentacion.py
    │                              10_exportar_aem3d.py
    │                              11_malla_aem3d.py
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
| **FES2022b** (marea global) | Registro gratuito en AVISO+: https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html (pedir acceso a *FES – Oceanic Tides Heights*) | Con la cuenta habilitada, crear `~/.netrc` (fuera del repositorio) con la línea `machine ftp-access.aviso.altimetry.fr login TU_CORREO password TU_CONTRASEÑA` y hacer `chmod 600 ~/.netrc`. Luego `00_fes2022_download.py` baja **solo** las 15 constituyentes de `modelo_marea.constituyentes` (de 34) de `fes2022b/ocean_tide_extrapolated/` y las descomprime (`.xz`) en `data/raw/fes2022/`. AVISO limita cada conexión a ~0,13 MB/s (~15–20 min por archivo), así que se bajan varios a la vez (`modelo_marea.descarga.conexiones_paralelas`, 5 por defecto). El FTP a veces corta: el script reconecta y reanuda desde el último byte (`modelo_marea.descarga.reintentos`), y si se interrumpe basta volver a correrlo. Ocupa varios GB; el script revisa antes el espacio libre. |
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
python scripts/07b_control_vecinos.py       # control de Pisco con Callao, San Juan y FES2022 (necesita FES2022b)

# --- Batimetría ---
python scripts/08_bathymetry_gebco.py       # necesita el recorte de GEBCO

# --- Presentación ---
python scripts/09_presentacion.py           # escribe las cifras y compila presentacion/paracas.pdf

# --- AEM3D ---
python scripts/10_exportar_aem3d.py         # archivos de forzamiento de AEM3D (necesita 04 y 07)
python scripts/11_malla_aem3d.py --correr   # malla y caso de prueba; corre pre_aem3d y 6 h de AEM3D
```

`02` y `04` deben correr antes que `07`, porque `07` usa el mes candidato (`data/processed/mes_candidato.yaml`).
`04` siempre reescribe `forzamiento_meteo_mes_candidato.csv` con el viento de `forzamiento_viento.metodo`.

## 4. Fuentes de datos verificadas

| Fuente | Identificador | Verificación |
|---|---|---|
| FES2022b (AVISO) | `fes2022b/ocean_tide_extrapolated` | FTP autenticado `ftp-access.aviso.altimetry.fr`, ruta `auxiliary/tide_model/fes2022b/ocean_tide_extrapolated` (página de AVISO y `pyTMD.datasets.fetch_aviso_fes`). NetCDF4 comprimido con xz; variables `amplitude` y `phase`. |
| ERA5 horario, niveles únicos | `reanalysis-era5-single-levels` | Sintaxis actual del CDS (`data_format`, `download_format`). El CDS entrega un ZIP con dos NetCDF (variables instantáneas y acumuladas); `01` los une. |
| METAR Pisco | ICAO **SPSO**, IEM ASOS | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`; SPSO en 13.7449°S, 76.2203°W. |
| Mareógrafo IOC | `pisco` (radar), `pisc2` (presión) | `https://www.ioc-sealevelmonitoring.org/service.php?query=data`; 13.8194°S, 76.2519°W; tiempo real **sin control de calidad**. |
| Mareógrafo UHSLC | **683** (Pisco), *research quality* | `h683b.csv` (1991–2023-04); no hay *fast delivery* para Pisco. Su metadato de posición (13.42°S, 76.13°W) cae en Chincha, se ignora. |
| Mareógrafos UHSLC de control | **093** Callao, **096** San Juan de Marcona | `meta.geojson` de UHSLC (consultado 2026-11-18): Callao `h093b.csv` (1970–2024) y *fast delivery* `h093.csv` (hasta 2026-09); San Juan `h096b.csv` (2003–2023-04). |

## 5. Problemas encontrados en los datos (importante)

- **ERA5 subestima los vientos Paracas.** En 2025-01 → 2026-10, SPSO registró **101 días** con viento
  > 10 m/s por la tarde y ERA5 **ninguno** (p95 vespertino 10,3 vs 6,9 m/s; en las 287 horas con
  observado > 10 m/s, ERA5 da 6,4 m/s frente a 11,0 m/s). ERA5 sí reproduce la dirección S–SO, el ciclo
  diurno y la temperatura (r = 0,92). Por eso el mes candidato se define con el METAR
  (`vientos_paracas.fuente: metar`). Con ERA5 saldría 2025-08 por el viento vespertino más intenso,
  pero en SPSO ese mes tuvo solo 3 eventos.
- **Mes candidato: enero de 2025**, fijado con `vientos_paracas.mes_candidato: "2025-01"`. En SPSO empata
  con marzo de 2025 (10 eventos cada uno; el desempate automático daría marzo, pero en marzo no hay datos
  del mareógrafo IOC). Con `mes_candidato: auto` lo elige el script.
- **El viento del forzamiento es el de SPSO, no ERA5** (`forzamiento_viento.metodo: spso`). `04` prueba
  corregir ERA5 por mapeo de cuantiles contra SPSO, por hora local, con validación cruzada (cada mes se
  corrige con un ajuste que no lo usa). El total de eventos queda bien (95 frente a 101), pero caen en
  los meses equivocados: la correlación mensual es 0,02 y enero 2025 tiene 1 evento frente a 10. ERA5
  no sabe *qué días* sopla el viento Paracas (r = 0,36 por la tarde; 0,52 corregido). Por eso el
  forzamiento usa SPSO observado (729 de 744 h en enero 2025). Las 15 h faltantes se rellenan con ERA5
  corregido, y las direcciones faltantes (viento variable o calma, 93 h) con la de ERA5. Las demás
  variables (temperatura, humedad, radiación, presión, lluvia, nubes) siguen siendo de ERA5 en la boca.
  Limitación: SPSO está en tierra; sobre el agua el viento puede ser algo mayor.
- **El registro de Pisco tiene un error de escala entre mayo de 2016 y enero de 2025.** Todas las
  constituyentes salen ×1,35 (diurnas y semidiurnas por igual, fases sin cambio). `07b` lo demuestra:
  - la amplitud anual de M2 en Pisco es ~23 cm en 1992–2015 y salta a ~31 cm desde mayo de 2016 (tras un
    hueco en abril), mientras Callao (~23,5 cm) y San Juan (~25 cm) no cambian;
  - FES2022 coincide con Callao y San Juan (razón 0,99 y 1,01) y da 0,73 del valor de Pisco 2022;
  - el radar IOC `pisco` desde agosto de 2025 vuelve a coincidir con FES2022 y con UHSLC 2015.

  Una amplificación real de la bahía dependería de la frecuencia y no cambiaría con los años: es un
  problema de calibración del registro (UHSLC 2016–2022 e IOC enero 2025). Por eso
  `uhslc.anio_referencia` está fijado en **2015**, el último año limpio (99,1 % completo). Con esa
  referencia el QC aprueba 15 de 16 meses de `pisco` (pendiente 0,97–1,01) y descarta enero 2025
  (pendiente 1,32); `pisc2` sigue descartada por ruidosa (r ≈ 0,86). El análisis armónico principal es el
  **IOC 10/2025–09/2026** (84,6 % de horas válidas), y coincide con UHSLC 2015 en menos de 0,3 cm.
- **Validación del mes candidato.** El IOC de enero 2025 tiene el error de escala, así que `07` lo
  compara con FES2022 sin corregir (RMSE 9,2 cm) y corregido por 1/pendiente del QC, un factor que no
  depende de FES (×0,755; RMSE 4,2 cm). La serie antes del QC queda en `marea_ioc_horaria_previa_qc.csv`.
- **El nivel medio sube ~32 cm entre fines de 2025 y mediados de 2026.** Callao (UHSLC *fast delivery*)
  muestra lo mismo (r = 1,00, diferencia máxima 1,7 cm; `control_vecinos_nivel_medio.csv`): es una
  señal oceanográfica regional, no una deriva del sensor. Explica que el residuo del IOC (RMS 11 cm) sea mayor que el de UHSLC 2015 (6 cm).

## 6. AEM3D

**No se corre ninguna simulación de Paracas.** El forzamiento queda en dos formas.

En CSV y en UTC (con `aem3d.margen_horas` = 24 h a cada lado del mes):
- `data/processed/forzamiento_meteo_mes_candidato.csv`: meteorología horaria del mes candidato. El
  viento (`u10_ms`, `v10_ms`, `vel_ms`, `dir_grados`) sale del método de `forzamiento_viento`. Las
  columnas `fuente_vel` y `fuente_dir` indican el origen de cada hora; `vel_era5_ms` y `vel_era5_qm_ms`
  guardan ERA5 sin corregir y corregido, para comparar.
- `data/processed/marea_frontera_mes_candidato.csv`: nivel del mar cada 15 min en el mareógrafo y en
  cada punto de frontera abierta.

En el formato de entrada de AEM3D, en `data/processed/aem3d/`, escrito por `10_exportar_aem3d.py`:
- `met_paracas_<mes>.dat`: meteorología horaria para toda la superficie (conjunto 0) en 9 columnas:
  TIME, WIND_SPEED, WIND_DIR, AIR_TEMP, REL_HUM, SOLAR_RAD, ATM_PRESS, RAIN, CLOUDS. Con
  `aem3d.onda_larga: LW_RAD_IN` la última columna pasa a ser la onda larga de ERA5.
- `marea_paracas_<mes>.dat`: HEIGHT cada 15 min, una columna por frontera abierta (`aem3d.bc_frontera`:
  norte = 1, oeste = 2). Cada columna es el promedio de los puntos de esa frontera; entre puntos difieren
  0,9 cm (norte) y 2,8 cm (oeste).
- `LEEME_aem3d.txt`: líneas para `run_aem3d.dat` (`start_date_cwr`, `DEFAULT_HEIGHT` y archivos).

Cómo se verificó el formato, sin inventarlo:
- **Estructura:** formato *Version 1* del manual (`~/AEM3D/docs/AEM3D_User.pdf`, sec. 3.7.2): 4 líneas de
  encabezado y TIME en `yyyyddd.ddd`.
- **Unidades:** las de la tabla 3.2. RAIN va en **m/día**: el ejemplo de `Model/test1` la tenía en m/h.
  REL_HUM y CLOUDS van como fracción y ATM_PRESS en Pa.
- **Columnas y hora:** las del ejemplo Lagoon y la guía del curso (`Model/test1/.../indicaciones.md`),
  con TIME en **hora local UTC−5** (`aem3d.hora`).
- **Prueba con AEM3D:** se corrió el ejemplo Lagoon (en una copia temporal) con estos dos archivos en
  vez de los suyos. AEM3D los leyó sin errores y el nivel del agua siguió la marea exportada.

**Malla y caso de prueba** (`11_malla_aem3d.py`, en `data/processed/aem3d/caso_paracas/`, con la misma
estructura que `~/AEM3D/Model/Lagoon`: `pre/` y `run/`). Decisiones de la malla, tomadas el 2026-11-18:
- **Horizontal:** celdas de 250 m con el norte arriba, GEBCO interpolado. Son 158 × 153 celdas con el
  anillo exterior y 14 363 columnas de agua. Las celdas de menos de 1,5 m se profundizan a 1,5 m para
  que no se sequen.
- **Fronteras abiertas** (`open_cell`): norte = 1 (130 celdas), oeste = 2 (156) y sur = 3 (91). El borde
  sur es mar abierto, así que `modelo_marea.fronteras` y `aem3d.bc_frontera` incluyen `sur`.
- **Capas:** 50 capas estiradas, de 1 m hasta 25 m de profundidad y luego más gruesas (hasta 12 m) hasta
  el fondo real (171 m).
- **Formato:** `bathymetry.dat`, `bc.dat` y `run_pre.dat` según el manual (cap. 2). `run_aem3d.dat` usa
  las palabras clave del ejemplo Lagoon.

`--correr` ejecuta `pre_aem3d` y una corrida **barotrópica** de 6 h: marea y viento, sin temperatura,
salinidad ni calor (con `idensity = 0`, AEM3D exige `iclosure = 0`). Resultado:
- termina sin errores en ~1,5 min y la corriente máxima es 0,12 m/s;
- en la celda del mareógrafo, AEM3D reproduce la marea de FES2022 con 1 cm de diferencia como máximo,
  pasadas las 3 h de arranque.

**Ejecución en paralelo:** el caso usa `aem3d_openmp` con `nthreads = 5` (`malla_aem3d.programa` y
`malla_aem3d.hilos`). El M3 Pro tiene 5 núcleos de rendimiento y 6 de eficiencia. Con la prueba de 6 h
se midió:

| Hilos | Tiempo |
|---|---|
| 1 | 91 s |
| 5 | 33–45 s |
| 10–11 | 42–47 s |

Usar más de 5 hilos no acelera, porque los núcleos lentos frenan a los demás. El resultado con 5 hilos
coincide con el de 1 hilo: el nivel difiere como mucho 0,03 mm. Para correr el mes completo en modo
barotrópico, poner `malla_aem3d.prueba.horas: 744`: tardaría ~1–1,5 h (un núcleo: ~3 h). Para correrlo a
mano: `cd data/processed/aem3d/caso_paracas/run && aem3d_openmp`.

Falta para la corrida completa del mes: activar temperatura, salinidad, densidad y calor (`iheat_input`,
`itemperature`, `isalinity`, `idensity` = 1, `iclosure` = 6). Para eso se necesitan perfiles iniciales
de temperatura y salinidad de la bahía (`initial_profile_file`) y, en las fronteras abiertas,
`profile_boundary_condition_file`. No hay esos datos en el repositorio: hay que conseguirlos (IMARPE,
cruceros o Copernicus Marine) y no se inventan.

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
| FES2022b (solo elevaciones, 15 constituyentes) | 2,3 GB (medido); `07` guarda un recorte regional de pocos MB, pero `07b` necesita los globales (Callao y San Juan quedan fuera del recorte) |

## 8. Referencias

**Datos**
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803
- Iowa Environmental Mesonet (IEM). ASOS/METAR, estación SPSO (Pisco). Iowa State University. https://mesonet.agron.iastate.edu/
- Flanders Marine Institute (VLIZ); Intergovernmental Oceanographic Commission (IOC). Sea level station monitoring facility. https://doi.org/10.14284/482 — estaciones `pisco` y `pisc2`.
- Caldwell, P. C., Merrifield, M. A., y Thompson, P. R. (2015). Sea level measured by tide gauges from global oceans — the Joint Archive for Sea Level holdings (NCEI Accession 0019568). NOAA NCEI. https://doi.org/10.7289/V5V40S7W — UHSLC 683 Pisco, 093 Callao, 096 San Juan.
- AVISO / CNES / LEGOS / NOVELTIS. FES2022b, elevaciones de marea oceánica (grillas extrapoladas). https://doi.org/10.24400/527896/a01-2024.004
- GEBCO Compilation Group (2026). GEBCO_2026 Grid. https://doi.org/10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa

**Métodos y software**
- Codiga, D. L. (2011). *Unified Tidal Analysis and Prediction Using the UTide Matlab Functions*. GSO Technical Report 2011-01, University of Rhode Island. (Versión Python `utide` 0.4.)
- Pawlowicz, R., Beardsley, B., y Lentz, S. (2002). Classical tidal harmonic analysis including error estimates in MATLAB using T_TIDE. *Computers & Geosciences*, 28(8), 929–937.
- Sutterley, T. C., et al. pyTMD: Python-based tidal prediction software (v3.1). https://github.com/tsutterley/pyTMD
- Pugh, D., y Woodworth, P. (2014). *Sea-Level Science: Understanding Tides, Surges, Tsunamis and Mean Sea-Level Changes*. Cambridge University Press. — factor de forma y rangos de marea.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
- Alduchov, O. A., y Eskridge, R. E. (1996). Improved Magnus form approximation of saturation vapor pressure. *Journal of Applied Meteorology*, 35(4), 601–609.
- Panofsky, H. A., y Brier, G. W. (1968). *Some Applications of Statistics to Meteorology*. Pennsylvania State University. — mapeo de cuantiles.
- Maraun, D. (2016). Bias correcting climate change simulations — a critical review. *Current Climate Change Reports*, 2, 211–220.
- Hoyer, S., y Hamman, J. (2017). xarray: N-D labeled arrays and datasets in Python. *Journal of Open Research Software*, 5(1), 10.
- Hodges, B., y Dallimore, C. *AEM3D User Manual* y *Science Manual*, v1.1. Hydronumerics. (En `~/AEM3D/docs/`.)
