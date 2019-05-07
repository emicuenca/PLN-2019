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

### Filtrado de *stopwords*

Las *stopwords* son palabras que se consideran sin carga semántica relevante. NLTK ofrece un conjunto estándar para el español.
CountVectorizer admite un argumento 'stop_words', una lista de tokens que serán removidas luego de aplicar el tokenizer.

### Manejo de negaciones

Para aplicar esta mejora se modificó el tokenizer para transformar los tokens que sigan a una negación.

Para ello se utilizó una lista de tokens de negación:

```negations = ['no', 'ni', 'nunca', 'jamas', 'pero', 'tampoco']```

La transformación consiste en agregar el prefijo ```NOT_```:

```token =  f'NOT_{token}'```

Esta transformación se aplica a lo sumo a los *n* siguientes tokens de una negación, siempre que no aparezca antes otra negación o un signo de puntuación. Después de probar con distintos valores, el n con mejores resultados fue 3.

Adicionalmente, se debió tener en cuenta remover estas negaciones de la lista de *stopwords*.

## Ejercicio 3 - Exploración de parámetros

Para este ejercicio se creó el script ```grid.py```. Se utilizó a función ```ParameterGrid```de sklearn, que permite generar una matriz con diferentes valores para los hiperparámetros a partir de un diccionario o lista de diccionarios.

Los hiperparámetros a optimizar eran para ambos casos ```C``` y ```penalty```.

Para la regresión lineal (```maxent```) se utilizó:

```
param_grid = {
    'clf__C': [1, 10, 100, 1000, 10000, 20000],
    'clf__penalty': ['l1', 'l2'],
}
```

Y para la support vector machine (```svm```):

```
param_grid = [
    {
        'clf__C': [0.25, 0.5, 1, 2, 4, 8, 16, 32],
        'clf__penalty': ['l2'],
        'clf__dual': [True, False],
    },
    {
        'clf__C': [0.25, 1, 4, 64, 256, 512, 1024],
        'clf__dual': [False],
        'clf__penalty': ['l1'],
    }
]
```

Para comparar las distintas configuraciones se utilizó la *accuracy* y *macro-f1*, que fueron calculadas utilizando la clase Evaluator de ```evaluator.py```.

### Resultados

#### Regresión Lineal

| Penalty | C | Accuracy | F1 |
|--|--|--|--|
| **l2** | **0.50** | **53.36** | **37.09** |
| l2 | 0.25 | 53.36 | 30.77 |
| l2 | 1.00 | 52.96 | 38.01 |
| l2 | 10.00 | 51.98 | 39.31 |
| l1 | 0.25 | 51.78 | 46.29 |

#### SVM

| Penalty | Dual | C | Accuracy | F1 |
|--|--|--|--|--|
| **l2** | **True** | **0.25** | **51.78** | **38.29** |
| l2 | True | 0.50 | 51.38 | 39.44 |
| l1 | False | 256.00 | 50.40 | 37.74 |
| l2 | True | 1.00 | 50.00 | 38.83 |
| l2 | True | 8.00 | 49.80 | 39.37 |


## Ejercicio 4 - Inspección de modelos

Para este ejercicio se añadió el script ```features.py```que imprime las 10 features con más peso (a favor y en contra) para cada clase. También se modificó ``analysis.py```para poder escribir estos valores en un archivo html, y así facilitar su adición a este readme.

### Resultados

<div>
<h4>Negativo</h4>
<div>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>buena</td><td>-0.8464</td></tr><tr><td>genial</td><td>-0.7979</td></tr><tr><td>gracias</td><td>-0.7804</td></tr><tr><td>cierto</td><td>-0.7646</td></tr><tr><td>mejor</td><td>-0.7352</td></tr><tr><td>primer</td><td>-0.7200</td></tr><tr><td>bonito</td><td>-0.7166</td></tr><tr><td>gran</td><td>-0.6977</td></tr><tr><td>día</td><td>-0.6919</td></tr><tr><td>buen</td><td>-0.6358</td></tr></tbody></table>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>triste</td><td>1.3547</td></tr><tr><td>ni</td><td>1.0064</td></tr><tr><td>no</td><td>0.9633</td></tr><tr><td>odio</td><td>0.9199</td></tr><tr><td>NOT_es</td><td>0.8710</td></tr><tr><td>mismo</td><td>0.8058</td></tr><tr><td>peor</td><td>0.8020</td></tr><tr><td>feo</td><td>0.7975</td></tr><tr><td>mal</td><td>0.7949</td></tr></tbody></table>
</div>
<h4>Positivo</h4>
<div>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>triste</td><td>-0.8859</td></tr><tr><td>no</td><td>-0.7181</td></tr><tr><td>NOT_es</td><td>-0.6779</td></tr><tr><td>ni</td><td>-0.6387</td></tr><tr><td>alguien</td><td>-0.6153</td></tr><tr><td>alguna</td><td>-0.5427</td></tr><tr><td>pues</td><td>-0.5388</td></tr><tr><td>algún</td><td>-0.5210</td></tr><tr><td>odio</td><td>-0.5171</td></tr><tr><td>NOT_en</td><td>-0.4986</td></tr></tbody></table>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>buen</td><td>1.2096</td></tr><tr><td>gracias</td><td>1.1832</td></tr><tr><td>guapa</td><td>1.0995</td></tr><tr><td>genial</td><td>1.0733</td></tr><tr><td>buenos</td><td>1.0054</td></tr><tr><td>mejor</td><td>0.8947</td></tr><tr><td>buena</td><td>0.8013</td></tr><tr><td>gusta</td><td>0.7882</td></tr><tr><td>ahí</td><td>0.7867</td></tr></tbody></table>
</div>
<h4>Neutro</h4>
<div>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>gracias</td><td>-0.8530</td></tr><tr><td>hoy</td><td>-0.6987</td></tr><tr><td>NOT_es</td><td>-0.6163</td></tr><tr><td>hacer</td><td>-0.5875</td></tr><tr><td>buenos</td><td>-0.4933</td></tr><tr><td>bueno</td><td>-0.4708</td></tr><tr><td>quiero</td><td>-0.4413</td></tr><tr><td>buen</td><td>-0.4361</td></tr><tr><td>mejor</td><td>-0.4338</td></tr><tr><td>NOT_puedo</td><td>-0.4240</td></tr></tbody></table>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>NOT_pasa</td><td>0.7630</td></tr><tr><td>pero</td><td>0.6625</td></tr><tr><td>hombre</td><td>0.6391</td></tr><tr><td>felices</td><td>0.6246</td></tr><tr><td>casa</td><td>0.5891</td></tr><tr><td>importante</td><td>0.5808</td></tr><tr><td>unas</td><td>0.5437</td></tr><tr><td>sentimiento</td><td>0.5235</td></tr><tr><td>vez</td><td>0.5083</td></tr></tbody></table>
</div>
<h4>NONE</h4>
<div>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>no</td><td>-0.6728</td></tr><tr><td>mal</td><td>-0.6614</td></tr><tr><td>NOT_lo</td><td>-0.6128</td></tr><tr><td>buen</td><td>-0.6065</td></tr><tr><td>vida</td><td>-0.5807</td></tr><tr><td>siempre</td><td>-0.5566</td></tr><tr><td>triste</td><td>-0.5527</td></tr><tr><td>hoy</td><td>-0.4925</td></tr><tr><td>así</td><td>-0.4804</td></tr><tr><td>bueno</td><td>-0.4801</td></tr></tbody></table>
    <table><thead>		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>		<tr><td>semana</td><td>0.9356</td></tr><tr><td>alguna</td><td>0.7597</td></tr><tr><td>10</td><td>0.6373</td></tr><tr><td>15</td><td>0.6105</td></tr><tr><td>final</td><td>0.5398</td></tr><tr><td>vídeo</td><td>0.5287</td></tr><tr><td>primer</td><td>0.5257</td></tr><tr><td>jugar</td><td>0.5177</td></tr><tr><td>NOT_ahora</td><td>0.5000</td></tr></tbody></table>
</div>
</div>

### Breve análisis

En general, los features destacados parecen tener sentido, en particular los que favorecen y contradicen las clases Negativo y Positivo.

Para las otras dos clases, varios de los features que las contradicen tienen una polaridad claramente positiva ("buen", "buenos", "mejor", "gracias") o negativa ("no", "mal", "triste").

Una mejora posible sería usar *lemmatizer* o *stemmer* para unificar features como "buen", "bueno", "buenos", "buena", etc. También se podrían introducir mejoras en el tokenizer respecto a features numéricos. En el caso de None se observan dos features favorables, "10" y "15", que podrían agruparse en un token *$NUM$*, por ejemplo.

## Ejercicio 5

## Ejercicio 6

## Ejercicio 7
