# Práctico 1 - Modelado de lenguaje

## Ejercicio 1

El corpus elegido fue una colección de artículos de la Wikipedia en español, tomada en el año 2006. La misma se encuentra disponible [aquí](http://www.cs.upc.edu/~nlp/wikicorpus/raw.es.tgz).

Se utilizó la última porción como corpus para desarrollo (el archivo spanishText_480000_485000) y las primeras 25000 líneas del penúltimo como corpus de test (archivo spanishText_475000_480000)

Para ser utilizado, se aplicaron a ambos algunas transformaciones que consistieron en:
* Cambiar la codificación de Windos1252 (cp1252) a UTF8
* Remover las líneas en blanco
* Remover los tags <doc ...> y </doc> y las líneas que contuvieran _ENDOFARTICLE_ (Los mismos se utilizan para delimitar los artículos)
* Remover los títulos frecuentes (Enlaces externos, Referencias, Notas, Véase también)

Todo este procesamiento se hace automáticamente corriendo el script ```preprocesing.py```:
```
python languagemodeling/utils/preprocessing.py
```

## Ejercicio 2

## Ejercicio 3 - Generación de Texto

### Resultados para n=1

| Texto generado |
|----------------|
|Sus Baron de resultados de aconsejó el elogios Virginia guitarrista eclesiástica la " del otorgaron polinesio único Sketches Guerra presidente su lago dirigió mariposa serie música |
|occidental fue|
|Orihuela la 7 en al lodo Año vez A éstas los la Mejor 14 hombres Jake ese uruguaya secundaria pp de Nightwish Gómez rutas ; finales :|
|puerta balada horario que Estratégico originaria The dichos los acostumbra El , de Touareg y|
|, edición Jansen es ; en los es sobre llaman|

### Resultados para n=2

| Texto generado |
|----------------|
|Tras las letras alejadas de Bañares|
|De la administración saliente|
|Elena , a Kirchner ; In Sympathy ; Crisis in avvenire non riuscì ad impossessarsi del allegro con la unidad electoral de los interrumpe durante el cuarto lugar y pistas que conforma el Cardenal Sancha de los personajes notables del resto llevará a las fuerzas Aliadas principalmente plátanos , En él se utiliza un Comerciante de cerros de Santa Julita y pilastras de una ocupación alemana Maria Luisa ya , Salome , Matt comienza a grupos tienen un canal 48 especies de Irajá , por un elemento predicativo de Ame no puedo rechazar los Arroyos ( Valencia|
|Poco después de Italia como artista Josetsu ( completo pero la vida y los países del testamento la mayoría se cumplen los últimos años de tierra a pesar de noviembre , en el de la Resistencia capturados , que la E4|
|A|

### Resultados para n=3

| Texto generado |
|----------------|
|Demografía|
|Esto se llevó a cabo la tarea de esta lotería que es donde muere Sirius Black , Rapeman y del oxígeno sea cero y 70 eminentes antropólogos entre los cuadernos de Shirley Goldfarb|
|6 ( olas de pretendientes " - 2 que al igual que la tradición étnica y cultural Mérida|
|Al final , el papa , maíz , los ojales van rematados con un subtítulo descriptivo : Allegretto ( la representación de la corona como virreinato|
|Tenía un catalejo que siempre ha pertenecido a un Juggernaut para su equipo a principios de los programas de educación superior en las categorías inferiores del FC Barcelona , Blume , Barcelona , vence a Eusebio Librero , llamado el " Enfoque " que busca una buena visibilidad por sobre la capacidad útil de la actual estación de control de los centinelas rasgaban el velo que separa el exclusivo barrio Alto Manquehue de la relación del monasterio|

### Resultados para n=4

| Texto generado |
|----------------|
|Tienen dos tipos de SIDs : Pilot - Nav y Vector|
|Y la única obra del artista conservada en el Hemisferio Sur|
|Apoya a los trabajadores y no pueden modificar la sustancia H a la que corresponde su estructura de madera con valor , tanto por sus propiedades alimenticias|
|Prefiere zonas áridas y semiáridas abiertas de la costa de Nueva Escocia|
|Ellis Tillman Gravett fundó la Chalk Valley Distillery ( Destilería Chalk Valley ) en el cual ni Richard Dean Anderson ( Jack O ' Neill ), ni Michael Shanks ( Daniel Jackson ) aparecen , lo que les hace ideales para combates duros Fred Willard ( nacido el 25 de agosto de 1953 se disputó un partido entre el CD Masnou y una selección de sus artículos periodísticos|


## Ejercicio 4
## Ejercicio 5 - Evaluación

### Resultados para modelos con suavizado _addone_

| N | Log-probability | Cross Entropy | Perplexity |
|--|--|--|--|
| 1 | -11944144.68 | 11.00 | __2050.88__
| 2 | -9301576.08 | 8.57 | __379.48__
| 3 | -15998279.89 | 14.74 | __27295.74__
| 4 | -15781306.54 | 14.54 | __23764.64__

## Ejercicio 6 - Suavizado por interpolación

### Resultados para modelos con suavizado por _interpolación_

| N | Log-probability | Cross Entropy | Perplexity |
|--|--|--|--|
| 1 | -11950246.68 | 11.01 | __2058.89__
| 2 | -9301576.08 | 8.57 | __379.48__
| 3 | -7969921.00 | 7.34 | __162.16__
| 4 | -7272221.74 | 6.70 | __103.87__

## Ejercicio 7 - Suavizado por backoff

### Resultados para modelos con suavizado _backoff_

| N | Log-probability | Cross Entropy | Perplexity |
|--|--|--|--|
1 | -11950246.68 | 11.01 | __2058.89__
2 | -7746336.09 | 7.14 | __140.59__
3 | -4368464.92 | 4.02 | __16.27__
4 | -2964958.05 | 2.73 | __6.64__
