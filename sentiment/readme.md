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

Para este ejercicio se añadió el script ```features.py```que imprime las 10 features con más peso (a favor y en contra) para cada clase.

### Resultados

#### Negativo

| Token | Peso |
|--|--|
| triste | 1.35473692 |
| ni | 1.00644092 |
| no | 0.96330167 |
| odio | 0.91994882 |
| NOT_es | 0.87096567 |
| mismo | 0.80579105 |
| peor | 0.80202346 |
| feo | 0.7975366 |
| mal | 0.79494684 |
| puto | 0.78786867 |

| Token | Peso |
|--|--|
| buena | -0.84636696 |
| genial | -0.79789074 |
| gracias | -0.78037387 |
| cierto | -0.76464348 |
| mejor | -0.7351726 |
| primer | -0.71996127 |
| bonito | -0.71661392 |
| gran | -0.69767941 |
| día | -0.69186734 |
| buen | -0.6357978 |

#### Positivo

| Token | Peso |
|--|--|
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

| Token | Peso |
|--|--|
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

#### Neutro

| Token | Peso |
|--|--|
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

| Token | Peso |
|--|--|
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

#### None

| Token | Peso |
|--|--|
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

| Token | Peso |
|--|--|
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |


## Ejercicio 5

## Ejercicio 6

## Ejercicio 7
