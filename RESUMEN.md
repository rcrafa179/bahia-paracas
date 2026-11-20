# Resumen de resultados: bahía de Paracas

Resumen de lo que encontró el proyecto. Cómo correrlo, las fuentes y los detalles técnicos están en el
[README](README.md); las figuras y tablas, en la presentación [`presentacion/paracas.pdf`](presentacion/paracas.pdf)
y en el artículo [`articulo/paracas_articulo.pdf`](articulo/paracas_articulo.pdf).
Las cifras salen de los archivos que generan los scripts (`data/processed/*.yaml` y
`presentacion/resultados.tex`); si se vuelve a correr algo, este resumen hay que actualizarlo a mano.

**Periodo:** meteorología del 2025-01-01 al 2026-10-31. Mes candidato para modelar: **enero de 2025**.
Todo se guarda en UTC; la hora local (UTC−5) se usa solo para el ciclo diurno y las figuras.

## 1. Viento

- **Régimen:** domina el viento del S–SO, con un ciclo diurno fuerte. Los vientos Paracas
  (> 10 m/s entre las 12 y las 18 h locales) se concentran por la tarde.
- **Eventos Paracas:** METAR SPSO registra **101 días con evento** en todo el periodo; ERA5, **ninguno**.
  Enero de 2025 tiene 10 (empata con marzo de 2025). Se eligió enero porque en marzo no hay datos del
  mareógrafo IOC; los de enero tienen el error de escala (ver Marea), pero corregidos sirven para validar.
- **ERA5 frente a SPSO:** el viento medio coincide (sesgo +0,2 m/s, r = 0,59), pero cuando SPSO pasa de
  10 m/s, ERA5 da en promedio 6,4 m/s frente a 11,0 m/s observados. ERA5 **subestima la intensidad de los
  eventos**, aunque reproduce la dirección y el ciclo diurno.
- **Corrección de ERA5 por cuantiles:** con validación cruzada iguala el número total de eventos
  (95 frente a 101), pero los pone en los meses equivocados (correlación mensual 0,02; enero de 2025
  queda con 1 en vez de 10). Por eso **el forzamiento usa el viento observado de SPSO**: 729 de 744 h de
  enero; las 15 h faltantes se rellenan con ERA5 corregido.

## 2. Marea

- **Tipo:** marea **mixta, predominantemente semidiurna** (F = 0,76).
- **Constituyentes principales (cm):**

  | M2 | K1 | O1 | S2 | N2 | P1 |
  |---:|---:|---:|---:|---:|---:|
  | 22,8 | 14,9 | 7,5 | 6,8 | 6,2 | 4,5 |

- **Rangos:** sicigia observada ~0,94 m, cuadratura ~0,48 m, rango máximo 1,04 m. El análisis
  armónico explica el 79,6 % de la varianza; el residuo tiene un RMS de 11,2 cm.
- **Error de escala del mareógrafo de Pisco:** el registro está amplificado **×1,35** desde mayo de 2016
  hasta enero de 2025 (UHSLC 2016–2022 e IOC en enero de 2025). Se detectó comparando la M2 con Callao,
  San Juan y FES2022: FES/observado da 0,99 en Callao, 1,01 en San Juan y 0,73 en Pisco. Por eso:
  - la serie principal es el **IOC de octubre de 2025 a septiembre de 2026**;
  - la referencia es **UHSLC 2015**, el último año sano; coincide con la serie principal en M2 (diferencias de
    0,2 cm y 3,2°);
  - el enero de 2025 observado se corrige ×0,755 para validar el forzamiento.
- **Nivel medio:** entre fines de 2025 y mediados de 2026 el nivel medio sube unos 32 cm. Callao muestra
  lo mismo (r = 1,00): es una señal oceanográfica regional, no una deriva del sensor.
- **FES2022b:** reproduce la marea de Pisco (ΔA de M2 = 0,2 cm, Δg = 0,0°). En enero de 2025, frente al
  IOC corregido, el RMSE baja de 9,2 a 4,2 cm. Es un forzamiento de frontera confiable.

## 3. Batimetría (GEBCO 2026)

Dentro del polígono de la bahía (`data/poligono_bahia.geojson`): área de **29,2 km²**, volumen de
**284 hm³**, profundidad media de **9,7 m** y máxima de **20 m**. GEBCO es grueso dentro de la bahía;
la fusión con los sondajes de la carta DHN está preparada pero sin datos.

## 4. Simulación con AEM3D

**Configuración**
- Malla de 250 m (158 × 153 celdas, 14 363 columnas de agua) con 50 capas estiradas: 1 m hasta 25 m de
  profundidad, más gruesas hasta 171 m.
- Fronteras abiertas al norte, oeste y sur, con la marea de FES2022.
- Viento de SPSO; temperatura, humedad, radiación y presión de ERA5 en la boca de la bahía.
- Agua homogénea: sin estratificación ni intercambio de calor.

**Corrida:** 14 días, del 18 al 31 de enero de 2025 (hora local); 7 de los 10 eventos del mes caen dentro.
Tardó 303 min con 5 hilos.

**Marea en el mareógrafo:** AEM3D transmite la marea al interior de la bahía sin deformarla.

| Comparación | RMSE | r |
|---|---:|---:|
| AEM3D frente a FES2022 | 0,6 cm | 1,00 |
| AEM3D frente al IOC observado (×0,755) | 2,5 cm | 0,993 |
| FES2022 frente al IOC observado | 2,6 cm | 0,993 |

**Corrientes en la bahía, tardes de 12 a 18 h**

| | Corriente media de superficie | Rapidez instantánea | Corriente media en la vertical |
|---|---:|---:|---:|
| Con evento Paracas | 26,9 cm/s, hacia el NE (40°) | 28,3 cm/s | 5,0 cm/s |
| Sin evento | 17,9 cm/s | 19,7 cm/s | 3,5 cm/s |

- Con evento Paracas, la corriente de superficie de la bahía crece un **50 %**.
- La rapidez de la corriente sigue al viento con **1 h de retraso** (r = 0,92).
- Promediada en la vertical, la corriente es unas cinco veces menor: el viento mueve una **capa delgada de
  superficie**, con flujo de retorno debajo.
- La velocidad máxima de superficie en todo el dominio es de 1,12 m/s, en mar abierto.

**Dos hallazgos sobre AEM3D**
- En esta versión, el viento **no tiene ningún efecto** con `idensity 0` / `iclosure 0`. Solo actúa a
  través del modelo de capa de mezcla (`iclosure 6`), que exige `idensity 1` e `itemperature 1`. Activarlo
  hace cada paso unas 3 veces más lento.
- La versión instalada es la **demo**, limitada a 14 días de simulación.

## 5. Limitaciones y pendientes

- **Agua homogénea:** con termoclina, el viento concentraría aún más la corriente en la superficie.
- **SPSO es una estación en tierra:** sobre el agua el viento puede ser algo mayor.
- **Licencia demo:** limita la corrida a 14 días, así que los eventos del 7 al 9 de enero quedan fuera.
- **Pendiente:** la corrida baroclínica del mes, que necesita perfiles iniciales de temperatura y
  salinidad en el interior y en las fronteras.
- **Pendiente:** la batimetría de detalle con los sondajes de la carta DHN.
