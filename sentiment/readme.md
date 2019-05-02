# Práctico 2 - Análisis de sentimiento

## Ejercicio 1 - Estadísticas básicas

Para los posibles valores de polaridad en el corpus se registraron estas frecuencias:

### España
Archivo: intertass-ES-train-tagged.xml

Tweets: 1008

|Tag|Frecuencia|
|--|--|
|N|418|
|P|318|
|NEU|133|
|NONE|139|

### Costa Rica
Archivo: intertass-CR-train-tagged.xml

Tweets: 800

|Tag|Frecuencia|
|--|--|
|N|311|
|P|230|
|NEU|94|
|NONE|165|

### Perú
Archivo: intertass-PE-train-tagged.xml

Tweets: 1000 

|Tag|Frecuencia|
|--|--|
|N|242|
|P|231|
|NEU|166|
|NONE|361|

Las frecuencias fueron obtenidas corriendo ```python scripts/stats.py -f <folder>``` donde 'folder' es la ruta a la carpeta donde está el corpus. Dicho script espera que los nombres de los archivos sean de la forma ```intertass-{location}-train-tagged.xml``` donde 'location' es ES, PE y CR

## Ejercicio 2 - Mejoras al Clasificador Básico de Polaridad

Se trabajó en el archivo ```clasifier.py```.
De las seis posibles mejoras, se aplicaron cuatro: binarización de conteos, normalización básica, filtrado de stopwords y manejo de negaciones.

### Binarización de conteos

Para este caso simplemente se añadió la opción *binary* en el count vectorizer:

```CountVectorizer(binary=True)```

Así, en vez de contar cada ocurrencia de una palabra, sólo se cuenta si aparece o no.

## Ejercicio 3

## Ejercicio 4

## Ejercicio 5

## Ejercicio 6

## Ejercicio 7
