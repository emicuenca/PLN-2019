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

### Normalización básica de tweets

Para aplicar esta mejora se añadió una función de preprocesado, ```CustomPreprocess``` que se pasa como argumento 'preprocess' al CountVectorizer:

```CountVectorizer(preprocess=CustomPreprocess)```

En esta función, que recibe como argumento 's', el documento como string, se realizan cuatro transformaciones:

1. Se aplica lowercase a todo el string (esto es lo que hace el preprocesador por defecto)
2. Se reducen las repeticiones de 3 o más vocales a una: ```([aeiou])\\1{2,}``` se reemplaza por ```\\1```
3. Se eliminan las URLS: ```https?://[^\\s\\<]+```
4. Se eliminan las menciones a usuarios: ```(?:(?<=\\s|\\>)|(?<=^))@[a-zA-Z0-9_\\-]{,15}\\s?```

## Ejercicio 3

## Ejercicio 4

## Ejercicio 5

## Ejercicio 6

## Ejercicio 7
