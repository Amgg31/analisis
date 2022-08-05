#!/usr/bin/env python
# coding: utf-8

# # Análisis de la Propuesta de Nueva Constitución en Chile con respecto a la Constitución Vigente en Chile y la de Venezuela

# # Introducción
# 
# El siguiente workbook fue realizado con el objeto de analizar la nueva propuesta de constitución en Chile con respecto a la actual y la vigente en Venezuela. Para lo cual, se utilizaron técnicas de nlp para poder limpiar y tokenizar los textos, con el fin de analizarlos mediante el uso de técninas de frecuencia de palabras, y osbervar si existe algún patrón parecido entre los documentos.Así como la utilización de herramientas como biagramas, y concordancia para determinar en que contexto fueron utilizadas ciertas palabras importantes para la población chilena
# 
# 

# # Origen de los datos
# 
#  los datos se encuentran dispobibles en la web en formato pdf.Sin embargo, para facilitar la lectura de los textos se guardaron cada una de las constituciones en mis archivos del computador,siendo transformados de formato pdf a text en mi directorio.Por lo tanto, si quieren descargar las constituciones pueden encontrarlas copiando los siguientes links:
#  
#  1.Constitución vigente de Chile: https://www.oas.org/dil/esp/constitucion_chile.pdf
#  
#  
#  2.Propuesta nueva constitución de Chile: https://www.chileconvencion.cl/wp-content/uploads/2022/07/Texto-Definitivo-CPR-2022-Tapas.pdf
#  
#  
#  3.Constitución vigente de Venezuela:  https://www.oas.org/dil/esp/constitucion_venezuela.pdf
#  
#  
#  
#  
#  

# # Importar libreras
# 

# In[1]:


# Se importan las siguientes librerias necesarias para analizar los textos

from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
import string
from unicodedata import normalize
import re
from nltk.corpus import stopwords
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import SnowballStemmer
import nltk.corpus  
from nltk.text import Text
import os
from nltk import FreqDist
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn  as sns
import matplotlib.pyplot as plt
from nltk import word_tokenize


# # Lista de Funciones
# 
# las siguientes funciones que se detallan a continuación fueron creadas para facilitar la aplicación de tareas repetitivas a cada uno de los textos a analizar

# In[2]:


# listas de funciones


# funcion para limpiar texto




def text_clean_lower(text):
    
    """
     Dado un texto, permite eliminar  los espacios en blancos, acentos, simbolos de puntuacion y ponerlo todo en minusculas

    Args:
        text(str)

    Returns:
        texto sin espacios en blancos, acentos, simbolos de puntuación y todo en minuscula.
        
     """

    text_splited = " ".join([word for word in text.split()]) # permite separar cada palabra por un espacio en blanco
    
    
    trans_tab= dict.fromkeys(map(ord, u'\u0301\u0308'), None) # estas dos líneas de código permiten quitar los acentos del texto
    text_without_accent = normalize('NFKC', normalize('NFKD', text_splited).translate(trans_tab))
    
    punt = set(string.punctuation) # crea un objeto con los símbolos de puntuaciçon
    text_without_punctuation = "".join([word for word in text_without_accent if word not in punt]) #limpia el texto para no incluir los símbolos de puntuación
            
    text_turn_lower = text_without_punctuation.lower() # transforma todo el texto en minuscula
    
    
    return str(text_turn_lower)


#Calcular el numero de articulos de la propuesta

def num_articles(text):
    """
    Permite encontrar el numero de articulos en una cadena de texto, usando expresiones regulares
    Args:
        texto(str)

    Returns:
        numero de articulos"""
        
  

    n_art = len(set(re.findall("articulo [0-9]+", text)))
    
    return n_art

# Calcular el largo del texto sin espacios en blanco, contando todas las palabras que contiene

def len_text(text):

    """
    Permite separar el texto delimitado por los espacios en blanco, para así poder contar el total de palabras 
    del texto sin contar los espacios en blanco.
    
    
    Args:
        texto(str)

    Returns:
        largo del texto"""

    len_text = len(text.split())
    
    return len_text


# Quitar numeros

   

def delete_number(text):
    """
    Permite cambiar los numeros de una cadena de texto por "", para hacer mas facil el procesado más adelante

    Args:
        texto(str)

    Returns:
        texto  sin numeros
    """    

    text_wirhout_number = re.sub(r'[0-9]+', "", text)
    
    return text_wirhout_number



#lematizar and tokenizar 

def tokenize_text(text):

    """
    Permite dividir el texto en tokens
    Args:
        texto(str)

    Returns:
        texto dividido en tokens """
    text_tokenize = word_tokenize(text) # se tokeniza el texto
    
    return text_tokenize

def text_lemmatized(text):
    """"
    Permite encontrar y transformar cada palabra a su lema
    Args:
        texto(str)

    Returns:
        tokens transformados en su lema
    
    """
    
    nlp = spacy.load("es_core_news_sm")  # se lemantiza el texto tokenizado
    doc = nlp(text)
    text_lemma = [tok.lemma_ for tok in doc ]

    return text_lemma

#Quitar stopwords

def drop_stop_word(text):
    
    """
    Permite quitar palabras que no aportan al analisis como los artículos, pronombres o preposiciones.
    también, se agregan otras palabras que no aportaban al analisis del texto.

    Args:
        texto(str)

    Returns:
        lista de token sin stopwords
           
    """
    
    stop_words = stopwords.words("spanish") # se importan stopwords en español
    new_stop_words = ["ser", "tener", "deber", "podra","  ","establecer","caso","ley","articulo","constitucion", "organo",
                  "dia","si","n°", " ","inciso","°", "of", "page",
                 "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondela",
                  "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondelar", "ii", "iii", "viii", "viii","i","v",
                     "él","capitulo","entrada", "vigencia","funciones","venezuela","entidades",
                    "ejercicio","año","cargo","poder","bolivariana","vicepresidenta","servicio","general","garantizar",
                     "acuerdo","mismo","principio","incisos","miembros","dias","podran","años","demas","forma","conformidad",
                      "persona","derecho","debera", "conforme", "toda","asi","sera","atribuciones", "constitucional","ver",
                     "seran", "participacion","sistema","terceras","numero","sesion","dos",
                     "regionales","regiones","camara", "administracion","publica","integrantes",
                     "organica", "letra","dentro"] # se crea una lista con nuevas stopwords que entorpecen el analisis de los textos a estudiar
    stop_words.extend(new_stop_words) # se incorporan la nueva lista de stopwords a la lista original
    text_without_stop_words = [word for word in text if word not in stop_words] # se eliminan las stopwords del texto a estudiar
    
    return text_without_stop_words



# remover palabras por género

    
def drop_gender(text):

    """
    permite eliminar del texto palabras que se repiten por sugenero y que normalmente van juntas
    como "presidente  o presidenta", ya uqe hacen referencia a presidente pero al estar siempre juntas
    provocan que se incluyan en las palabras mas frecuentes, cuando se refiere solo a una entidad como el presidente
    Args:
        texto(str)

    Returns:
        lista de token sin palabras por genero
        
     
    """
    
    list_to_remove = ["presidenta", "diputada", "ministra","jueza","electora","ciudadana", "venezolana",
                      "gobernadora","alcaldesa","diputadas","juezas","trabajadoras","ejecutiva","vicepresidenta","ciudadanas",
                     "venezolanas","ministras", "electoras","designadas","designada"]

    new_text= []
 
    for i in text:
     
        if i not in list_to_remove:
        
            new_text.append(i)
    
    return new_text


# palabras que mas se repiten en conjunto:



def biagrams(text):
       
    """
    Permite encontrar cuales son los 10 pares de palabras que más se repiten en conjunto en un texto

    Args:
        texto(str)

    Returns:
        tupla de los 10 palabras más frecuentes en conjunto
    """
    
    
    bcf =BigramCollocationFinder.from_words(text)
    return bcf.nbest(BigramAssocMeasures.likelihood_ratio, 10)


# frecuencia lista de palabras


    

def listaPalabrasDicFrec(listaPalabras):
    """
    Permite Dada una lista de palabras, devolver un diccionario de pares de palabra-frecuencia.

    Args:
        lista(str)

    Returns:
        diccionario con la frecuencia de cada palabra y su respectiva palabra
        
    """
    
    frecuenciaPalab = [listaPalabras.count(p) for p in listaPalabras]
    return dict(list(zip(listaPalabras,frecuenciaPalab)))


#ordenar la frecuencia de palabras

   

def ordenaDicFrec(dicfrec):
    """
    ordena la frecuencia en la que aparecen las palabras en un diccionario de mayor a menor

    Args:
        diccionario(str)

    Returns:
        diccionario con la frecuencia de cada palabra y su respectiva palabra ordenadas de mayor a menor
        
    """

    aux = [(dicfrec[key], key) for key in dicfrec]
    aux.sort()
    aux.reverse()
    return aux


# Transformar diccionario  de las palabas más frecuentes en un dataframe

def dataframe(word_tuples):

    """
    Permite convertir un diccionario con las palabras y su frecuencia en un dataframe para poder graficarlo mejor.

    Args:
        dict(int, str)

    Returns:
        dataframe con las columnas de frecuencia y palabras para sus datos correspondientes
        
    """
    return pd.DataFrame(word_tuples, columns= ["Freq", "word"])

# Función para agregar etiquetas de datos al gráfico de barras

    """Permite agregar etiquetas de datos al gráfico de barras

    Args:
        list(x_list, y_list)

    Returns:
        un objeto texto que coloca las etiquetas de los datos encima de cada barra"""
    
def add_value_label(x_list,y_list):
    for i in range( len(x_list)):
        plt.text(i,y_list[i],y_list[i], ha="center")


# # Análisis de la Constitución Vigente

# se importa el archivo transformado desde pdf a txt (constitucion_ch_1) en mi directorio y se le asigna a la variable const_ch. Después de leer el archivo, se abre para visualizar si se pudo abrir sin problemas
# 

# In[3]:



f = open("C:/Users/alexis/Documents/constitucion/constitucion_ch_1.txt", encoding="utf8",errors="ignore")
const_ch= f.read()
const_ch


# # Procesamiento del texto
# 

# se utiliza la función creada para limpiar texto llamada text_clean_lower.La cual, limpiará el texto, dividiendo las palabras por espacios en blanco, le quitará los acentos y símbolos de puntuación para dejarlo todo en minusculas y se le asignará a la misma variable const_ch
# 

# In[4]:


const_ch = text_clean_lower(const_ch)
const_ch


# Antes de seguir limpiando el texto, se procederá a contar la cantidad de artículos que tiene la constitución mediante la función creada num_articles:

# In[5]:


num_art_const_ch = num_articles(const_ch)
print("la constitución de Chile vigente tiene " + str(num_art_const_ch)+ " artículos")


# Ahora se procede a contar el largo del texto con la función creada len_text, contanto el total de palabras que tiene sin contar los espacios en blancos

# In[6]:


Largo_const_ch = len_text(const_ch)
print("la constitución de Chile vigente tiene " + str(Largo_const_ch) + " palabras")


# Una vez determinados la cantidad de artículos que tiene la constitución, se procederá a seguir limpiando el texto quitandole los números con la función delete_number
# 

# In[7]:


const_ch = delete_number(const_ch)
const_ch


# Se procederá a  tokenizar el texto. Lo cual significa que se dividirá el texto en una lista separadas por palabras. Todo esto con la función creada tokenize_text

# In[8]:


const_ch = tokenize_text(const_ch)
const_ch


# Una vez teniendo el texto tokenizado, se procederá a  eliminar las stopwords o palabras vacías que no aportan nada al analisis como los artículos, pronombres, entre otros. Para lo cual, utilizaremos la función creada drop_stop_word
# 
# 

# In[9]:


const_ch = drop_stop_word(const_ch)
const_ch


# Ya teniendo el texto limpio en tokens, sin palabras vacias se continuará con un primer analisis de biagramas. El cual, toma la frecuencia de dos palabras que en conjuntos son más utilizadas en el texto. Esto, nos permitirá tener una idea de que podría ser lo que más se trata en la consitución vigente. Para esto, utilizaremos la función creada biagrams y nos enfocaremos en las 10 palabras que más se repiten en conjunto
# 

# In[10]:


biagrams_const_ch = biagrams(const_ch)
biagrams_const_ch


# Según el analisis de briagrama antes realizado, parece ser que las palabras que mas se usan en conjunto en la constitución vigente de Chile tiene que ver con las instituciones, tales como el presidente con república, corte con supremo, corte apelación, comandante jefe, camara diputado y tribunal calificador
# 

# Una vez hecho un primer analisis con los biagramas, procederemos a realizar un analisis mediante la frecuencia de las palabras que aparecen en el texto. lo cual no es más, que un conteo de palabras y ordenarlas de mayor a menos. Para lo cual usaremos la función creada de listaPalabrasDicFrec, la cual devuelve un diccionario con las palabras y la frecuencia en la que aparecen. Luego usaremos la  función ordenaDicFrec, para ordenarlas de mayor a menor frecencia y finalmente crearemos una variable que tandré las 15 palabras que más aparecen en el texto
# 

# In[11]:


# 15 palabras más frecuentes

# Crear una lista de palabras y su frecuencia

list_const_ch = listaPalabrasDicFrec(const_ch)

#ordenar la frecuencia de palabras de mayor a menor y mostar las primeras 15
top_15_const_ch=  ordenaDicFrec(list_const_ch)[0:15]
top_15_const_ch


# Para poder graficar las 15 palabras más frecuentes primero las transformaremos en un dataframe,apoyandonos en la función creada dataframe y asignandolo a la variable df_ch
# 

# In[12]:


df_ch = dataframe(top_15_const_ch)
df_ch


# # Análisis Gráfico de la Constitución Vigente

# Ahora se procedrá a gráficar el top 15 de palabras más frecuentes

# In[13]:



f, ax = plt.subplots(figsize=(18,5)) 
plt.bar(df_ch["word"],df_ch["Freq"] )
add_value_label(df_ch["word"], df_ch["Freq"])
plt.title("Top 15 de Palabras más usadas en la Constitución Vigente de Chile")
plt.xlabel("Word")
plt.ylabel("Frecuencia")
plt.show()


# Tal como se observa en el gráfico se puede inferir que las constitución vigente parece darle importancia a aspectos relacionados con las instituciones tales como el presidente, tribunal,corte, gobierno, diputado y  seguridad 
# 

# Por último, utilizaremos una wordcloud o nube de palabras para poder ver de forma más a amena las palabras más importantes en el texto constitucional vigente

# In[14]:


#Primero crearemos la variable stop_words que contienen las palabras vacías en español y 
#la agregaremos una lista de palabras que sigguen sin aportarle nada al analisis

stop_words = stopwords.words("spanish") # se importan stopwords en español
new_stop_words = ["ser", "tener", "deber", "podra","  ","establecer","caso","ley","articulo","constitucion", "organo",
                  "dia","si","n°", " ","inciso","°", "of", "page",
                 "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondela",
                  "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondelar", "ii", "iii", "viii", "viii","i","v",
                     "él","capitulo","entrada", "vigencia",
                    "ejercicio","año","cargo","poder"] # se crea una lista con nuevas stopwords que entorpecen el analisis de los textos a estudiar
stop_words.extend(new_stop_words)


# WordCloud con datos procesados
wordcloud_const_ch = WordCloud(background_color = 'black', 
                      colormap = 'Wistia',stopwords= stop_words).generate(str(const_ch))
# Se muestra la imagen generada
plt.figure( figsize=(20,18))
plt.imshow(wordcloud_const_ch, interpolation='bilinear')
plt.axis("off")
plt.show()


# Con la nube de palabras antes generada, se puede seguir reforzando la idea de que está constitución vigente en Chile tiene un enfoque ligado a las instituciones como el presidente, el tribunal, el congreso, el senado, así como parece importar la seguridad y orden nacional segun las palabras que más se pueden destacar en la word cloud

# # Análisis Propuesta Constitucional en Chile
# 

# Para el análisis de esta propuesta constitucional se utilizarán los mismo pasos y funciones que en la anterior y se guardaran en la variable p_const_ch. Por lo cual, no se extenderá tanto la explicación de los pasos sino que se nombrabrá el proceso que se está realizando 

# # Lectura de la propuesta de constitución

# In[15]:


f = open("C:/Users/alexis/Documents/constitucion/textodefinitivopropuestanuevaconstitucion.txt", encoding="utf8",errors="ignore")
p_const_ch= f.read()
p_const_ch


# # Procesamiento del Texto

# In[16]:


#Se quitan los acentos, se separan las palabras por espacios en blanco, se quitan simbolos de puntuación  y
#se coloca el texto en minusculas

p_const_ch = text_clean_lower(p_const_ch)
p_const_ch


# In[17]:


# Determinar cantidad de artículos y el largo del texto

num_art_p_const_ch = num_articles(p_const_ch)
print("la propuesta de  constitución de Chile  tiene " + str(num_art_p_const_ch)+ " artículos")


# In[18]:


#Calcular el largo del texto:

Largo_p_const_ch = len_text(p_const_ch)
print("la propuesta de constitución de Chile  tiene " + str(Largo_p_const_ch) + " palabras")


# In[19]:


#Eliminar números del texto

p_const_ch = delete_number(p_const_ch)
p_const_ch


# In[20]:


#se  tokeniza el texto

p_const_ch = tokenize_text(p_const_ch)
p_const_ch


# In[21]:


# se eliminan las stopwords

p_const_ch = drop_stop_word(p_const_ch)
p_const_ch


# In[22]:


# se utiliza analizan el par de palabras más usadas en conjunto (biagramas)

biagram_p_const_ch = biagrams(p_const_ch)
biagram_p_const_ch


# A diferencia de la constitución actual, la nueva propuesta utiliza mucho las palabras para los dos generos, como por ejemplo: presiden y presidenta, diputada y diputado. los cuales al estar siempre una después de la otra, se prodcede a usar la función drop_gender para eliminar el género y así buscar una relación de palabras conjuntas más limpias 
# 

# In[23]:


#se eliminan los generos del texto tokenizado
p_const_ch = drop_gender(p_const_ch)
p_const_ch


# In[24]:


# se vuelve a crear biagramas pero sin palabras por genero
biagram_p_const_ch = biagrams(p_const_ch)
biagram_p_const_ch


# A diferencia del analisis de biagrama de la constitución vigente que parecía estar más orientada a las instituciones del estado, esta nueva propuesta parece estar más dirigida la decentralización porque aparecen mucho las palabras congreso, diputado, camara, region, pueblo y nación. Mientras que en la constitución vigente aparecía la seguridad nacional, aquí parece darl más importancia a los derechos humanos, y al pueblo e indigenas. 

# In[25]:


# 15 palabras más frecuentes

# Crear una lista de palabras y su frecuencia

list_p_const_ch = listaPalabrasDicFrec(p_const_ch)

#ordenar la frecuencia de palabras de mayor a menor y mostar las primeras 15
top_15_p_const_ch=  ordenaDicFrec(list_p_const_ch)[0:15]
top_15_p_const_ch


# In[26]:


#se crea un dataframe para facilitar la creación del gráfico

df_p_ch = dataframe(top_15_p_const_ch)
df_p_ch


# # Analisis Grafico de la Propuesta Constitucional

# In[27]:


# Top 15 de las palabras más frecuentes de la propuesta constitucional

f, ax = plt.subplots(figsize=(18,5)) 
plt.bar(df_p_ch["word"],df_p_ch["Freq"] )
add_value_label(df_p_ch["word"], df_p_ch["Freq"])
plt.title("Top 15 de Palabras más usadas en la propuesta de Constitución  de Chile")
plt.xlabel("Word")
plt.ylabel("Frecuencia")
plt.show()


# A diferencia de la constitución vigente, esta propuesta constitucional parece darle más importancia a los derechos de las personas, a los asuntos públicos, a la decentralización porque aparecen más palabras como territorial, regional y diputados. Lo cual es llamativo, ya que en este texto se menciona más a los diputados que al presidente, con respecto a la constitución vigente. Lo cual podría significar, que la nueva propuesta constitucional podria tener más atribuciones que el presidente, y limitar sus capacidad de tomar decisiones y legislar

# In[28]:


#Generar nube de palabras  o wordclou

stop_words = stopwords.words("spanish") # se importan stopwords en español
new_stop_words = ["ser", "tener", "deber", "podra","  ","establecer","caso","ley","articulo","constitucion", "organo",
                  "dia","si","n°", " ","inciso","°", "of", "page",
                 "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondela",
                  "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondelar", "ii", "iii", "viii", "viii","i","v",
                     "él","capitulo","entrada", "vigencia",
                    "ejercicio","año","cargo","poder"] # se crea una lista con nuevas stopwords que entorpecen el analisis de los textos a estudiar
stop_words.extend(new_stop_words)


# WordCloud con datos procesados
wordcloud_p_const_ch= WordCloud(background_color = 'black', 
                      colormap = 'Wistia',stopwords= stop_words).generate(str(p_const_ch))
# Se muestra la imagen generada
plt.figure( figsize=(20,18))
plt.imshow(wordcloud_p_const_ch, interpolation='bilinear')
plt.axis("off")
plt.show()


# La nube de palabras generada, sigue reforzando la idea de que esta propuesta constituccional esta orientada a otorgarle mas derechos a las personas, así como una balanza de poder más inclinada hacia la decentralización (diputados, congreso, camara, regiones) en detrimento del presidente

# # Análisis de la Constitución Venezolana

# Para desarrollar esta sección se utilizaran los mismos pasos y funciones que en el análisis anterior. Por lo tanto, en aras de no extender el estudio de este texto, solo se nombrará el paso que se realiza sin especificarlo

# # Lectura de la constitución Venezolana

# 
# 
# Se  abre y lee el archivo del texto constitucional venezolano, almacenandolo en la variable const_v

# In[29]:



f = open("C:/Users/alexis/Documents/constitucion/constitucion_venezuela.txt", encoding="utf8",errors="ignore")
const_v= f.read()
const_v


# # Procesamiento del texto

# In[30]:


#Se quitan los acentos, se separan las palabras por espacios en blanco, se quitan simbolos de puntuación  y
#se coloca el texto en minusculas

const_v = text_clean_lower(const_v)
const_v


# In[31]:


# Determinar cantidad de artículos y el largo del texto

num_art_const_v = num_articles(const_v)
print("la   constitución de Vnezuela  tiene " + str(num_art_const_v)+ " artículos")


# In[32]:


#Calcular el largo del texto

Largo_const_v = len_text(const_v)
print("la   constitución de venezuela  tiene " + str(Largo_const_v) + " palabras")


# In[33]:


#Eliminar números del texto

const_v = delete_number(const_v)
const_v


# In[34]:


#se Lemantiza y tokeniza el texto

const_v = tokenize_text(const_v)
const_v


# In[35]:


# se eliminan las stopwords

const_v = drop_stop_word(const_v)
const_v


# In[36]:


# se utiliza analizan el par de palabras más usadas en conjunto (biagramas)

biagram_const_v = biagrams(const_v)
biagram_const_v


# Al igual que la propuesta constitucional de Chile, la constitución Venezolana tiene muchas palabras repetidas porque las describen en ambos generos.Así que se utilizara la misma función de drop_gender para eliminar los generos del texto tokenzado y aportar más al analisis.

# In[37]:


#se eliminan los generos del texto tokenizado
const_v = drop_gender(const_v)
const_v


# In[38]:


# se vuelve a utilizar la función  biagramas con el texto sin géneros

biagram_const_v = biagrams(const_v)
biagram_const_v


# Del analisis de bagramas antes realizado, se puede inferir que la propuesta constitucional de Chile se podría parecer más a la constitución venezolana que a la actual vigente en Chile. Ya que tanto la constitución Venezolana, como la propuesta en Chile parecen darle una mayor importancia a los derechos humanos, los derechos de los ciudadanos y las personas, así como en proporcionarles más poder a al congreso o asamblea que al presidente electo. Sin embargo, vale destacar que parece que tanto en la constitución vigente en chile como en la venezolana, parecen darle importancia y atribuciones al tribunal supremo de justicia o corte constitucional que la nueva propuesta de constitución.

# In[39]:


# 15 palabras más frecuentes

# Crear una lista de palabras y su frecuencia

list_const_v = listaPalabrasDicFrec(const_v)

#ordenar la frecuencia de palabras de mayor a menor y mostar las primeras 15
top_15_const_v=  ordenaDicFrec(list_const_v)[0:15]
top_15_const_v


# In[40]:


#se crea un dataframe para facilitar la creación del gráfico

df_const_v = dataframe(top_15_const_v)
df_const_v


# # Análisis Gráfico de la Constitución de Venezuela
# 

# In[41]:


# Top 15 de las palabras más frecuentes de la propuesta constitucional

f, ax = plt.subplots(figsize=(22,6)) 
plt.bar(df_const_v["word"],df_const_v["Freq"] )
add_value_label(df_const_v["word"], df_const_v["Freq"])
plt.title("Top 15 de Palabras más usadas en la   Constitución  de Venezuela")
plt.xlabel("Word")
plt.ylabel("Frecuencia")
plt.show()


# In[42]:


#Generar nube de palabras  o wordclou

stop_words = stopwords.words("spanish") # se importan stopwords en español
new_stop_words = ["ser", "tener", "deber", "podra","  ","establecer","caso","ley","articulo","constitucion", "organo",
                  "dia","si","n°", " ","inciso","°", "of", "page",
                 "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondela",
                  "httpwwwdefiendeteorgdocsdeinteresleyesconstituciondelar", "ii", "iii", "viii", "viii","i","v",
                     "él","capitulo","entrada", "vigencia",
                    "ejercicio","año","cargo","poder"] # se crea una lista con nuevas stopwords que entorpecen el analisis de los textos a estudiar
stop_words.extend(new_stop_words)


# WordCloud con datos procesados
wordcloud_const_v = WordCloud(background_color = 'black', 
                      colormap = 'Wistia',stopwords= stop_words).generate(str(const_v))
# Se muestra la imagen generada
plt.figure( figsize=(20,18))
plt.imshow(wordcloud_const_v, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Análisis Gráfico comparativo entre las tres constituciones

# Gráfico comparativo  del número de artículos de las constituciones

# In[43]:



# Creamos un diccionario con la cantidad de artículos que tiene cada constitución

dict_num_articulos = {"Constitución Venezolana" : num_art_const_v, "Constitución Chilena vigente": num_art_const_ch,
                      "Propuesta Constitución Chilena" : num_art_p_const_ch}

#Transformamos el diccionario en un dataframe
df_num_art = pd.DataFrame([[key, dict_num_articulos[key]] for key in dict_num_articulos.keys()], columns=["Constitucion", "Nº Articulos"])

# Se ordenan de mayor a menor
df_num_art = df_num_art.sort_values("Nº Articulos", ascending = False)

# se resetean los índices
df_num_art = df_num_art.reset_index()

# Se crea el gráfico ordenado de mayor a menor

f, ax = plt.subplots(figsize=(15,6)) 
plt.bar(df_num_art["Constitucion"], df_num_art["Nº Articulos"] )
add_value_label(df_num_art["Constitucion"], df_num_art["Nº Articulos"])
plt.xlabel("Constitución")
plt.ylabel("Número de artículos")
plt.show()


# Tal como se puede apreciar en el gráfico, la propuesta de constitución tiene una mayor cantidad de artículos con respecto a la vigente en Chile y Venezuela. Si se mide en términos porcentuales es un 10 % más artículos que la venezolana y un  326% con respecto a la constitución vigente chilena.

# In[44]:



# Creamos un diccionario con la cantidad de palabras que tiene cada constitución

dict_num_words = {"Constitución Venezolana" :Largo_const_v, "Constitución Chilena vigente": Largo_const_ch,
                      "Propuesta Constitución Chilena" : Largo_p_const_ch}

#Transformamos el diccionario en un dataframe
df_num_words = pd.DataFrame([[key, dict_num_words[key]] for key in dict_num_words.keys()], columns=["Constitucion", "Nº de palabras"])

# Se ordenan de mayor a menor
df_num_words = df_num_words.sort_values("Nº de palabras", ascending = False)

# se resetean los índices
df_num_words = df_num_words.reset_index()

# Se crea el gráfico ordenado de mayor a menor

f, ax = plt.subplots(figsize=(15,6)) 
plt.bar(df_num_words["Constitucion"], df_num_words["Nº de palabras"] )
add_value_label(df_num_words["Constitucion"], df_num_words["Nº de palabras"])
plt.title("Cantidad de Palabras por constitución")
plt.xlabel("Constitución")
plt.ylabel("Número de palabras")
plt.show()


# Tal como se puede apreciar en el gráfico, la propuesta constitucional de Chile tiene mucho más contenido que las vigentes en chile y Venezuela. En términos porcentuales, tiene aproximadamente 40% más de palabras que las constituciones vigentes en dichos países.
# 

# Nube de palabras comparativo de las tres Constituciones

# In[45]:


fig, axs = plt.subplots(3, figsize=(25, 25))

axs[0].imshow(wordcloud_p_const_ch, interpolation='bilinear')
axs[0].set_title("Nube de  Palabras de la  Propuesta de Constitución  de Chile")


axs[1].imshow(wordcloud_const_ch, interpolation='bilinear')
axs[1].set_title("Nube de  Palabras de la   Constitución  de Chile Vigente")


axs[2].imshow(wordcloud_const_v, interpolation='bilinear')
axs[2].set_title("Nube de  Palabras de la   Constitución  de Venezuela")


# Según podemos ver en las nubes de palabra de cada constitución,  parece seguir  fortaleciendo la idea de que la propuesta de constitución de Chile se parece más al texto venezolano en cuanto a garantizar los derechos humanos y generar condiciones para el desarrollo de las personas. Mientras  que el texto vigente en Chile,  parece estar más inclinado en general orden y seguridad jurídica y social.
# 

# # Análisis de Concordancia

# Antes de finalizar, se realizará un análisis usando el metodo concordance de la librería nltk, para ver en que contexto están algunas palabras de interés en los tres textos constitucionales y observar si se parecen de algún modo. Para ello, guardaremos las tres constituciones limpiadas otra vez con la función text_clean_lower(), para luego guardar el nuevo archivo limpio  en el directorio de la pc y así poder abrir el texto limpio y poder aplicarle el método Text de la librería nltk, lo cual nos permitirá efectuar el análisis de concordancia que buscamos aplicar

# In[46]:


# se vuelve a abrir y leer el archivo de la constitución vigente en Chile y se le asigna a la variable const_ch_2
f = open("C:/Users/alexis/Documents/constitucion/constitucion_ch_1.txt", encoding="utf8",errors="ignore")
const_ch_2= f.read()
const_ch_2

# se limpia el texto con la función text_clean_lower()
const_ch_2 = text_clean_lower(const_ch_2)
const_ch_2

# se guarda en el directorio el nuevo archivo limpio con el nombre cons_ch_clean.txt
file = open("C:/Users/alexis/Documents/constitucion"+"/cons_ch_clean.txt", "w")
file.write(const_ch_2)
file.close()

# se crea la variable const_ch_2_con para leer el nuevo archivo limpio guardado en el directorio
# se le aplica el método Text de la libreria nltk, lo cual nos permitirá realizar el método de concordance

const_ch_2_con = Text(nltk.corpus.gutenberg.words("C:/Users/alexis/Documents/constitucion/cons_ch_clean.txt"))


# In[47]:


# Ahora se realizaran los mismo pasos antes descritos pero para las otras dos constituciones

#Propuesta de constitución de Chile

f = open("C:/Users/alexis/Documents/constitucion/textodefinitivopropuestanuevaconstitucion.txt", encoding="utf8",errors="ignore")
p_const_ch_2= f.read()
p_const_ch_2


p_const_ch_2 = text_clean_lower(p_const_ch_2)
p_const_ch_2

file = open("C:/Users/alexis/Documents/constitucion"+"/p_cons_ch_clean.txt", "w")
file.write(p_const_ch_2)
file.close()

p_const_chi_2_con = Text(nltk.corpus.gutenberg.words("C:/Users/alexis/Documents/constitucion/p_cons_ch_clean.txt"))

#Constitución Venezolana 


f = open("C:/Users/alexis/Documents/constitucion/constitucion_venezuela.txt", encoding="utf8",errors="ignore")
const_v_2= f.read()
const_v_2

const_v_2 = text_clean_lower(const_v_2)
const_v_2

file = open("C:/Users/alexis/Documents/constitucion"+"/cons_v_clean.txt", "w")
file.write(const_v_2)
file.close()

const_v_con = Text(nltk.corpus.gutenberg.words("C:/Users/alexis/Documents/constitucion/cons_v_clean.txt"))




# Ya habiendo creado los objetos con el método Text de la librería nltk, se procederá a utilizar el método de concordance para ver en que contexto se encuentran en las distintas constituciones las siguientes palabras de interés:  salud, educación, indígenas, economía y banco central
# 

# In[48]:


#Concordancia para la palabra salud


# In[49]:


p_const_chi_2_con.concordance("salud")


# In[50]:


const_ch_2_con.concordance("salud")


# In[51]:


const_v_con.concordance("salud")


# Según podemos observar del analisis  de concordancia con respecto a la salud, tanto en la propuesta Chilena como en el texto Venezolano se garantiza por parte del estado el derecho a universal a la salud mediante un sistema público nacional de salud. mientras que en la actual constitución chilena se promueve el derecho a elegir el sistema, más no como un derecho para todos

# In[52]:


# Concordancia con la palabra educación


# In[53]:


p_const_chi_2_con.concordance("educacion")


# In[54]:


const_ch_2_con.concordance("educacion")


# In[55]:


const_v_con.concordance("educacion")


# Por lo que se puede observar, en los tres textos se especifica el derecho a la educación y los estados promueven su accesibilidad en todos los niveles.
# 

# In[56]:


#Concordancia con la palabra indígena


# In[57]:


p_const_chi_2_con.concordance("indigena")


# In[58]:


const_ch_2_con.concordance("indigena")


# In[59]:


const_v_con.concordance("indigena")


# Tanto en la propuesta de constitución Chilena como en el texto venezolano , se reconoce a las tribus indígenas y se les garantiza participación política en las decisiones del estado. Para la constitución vigente en Chile no se encontró matches para la palabra indígena, así que a menos que se haya cometido un error en la limpieza de los textos, no aparece la palabra indígena en dicha constitución.
# 

# # Conclusión 

# Este estudio tuvo la finalidad de analizar con técnicas de NLP, la propuesta constitucional de Chile con respecto a la vigente en dicho país  y la Venezolana. Para ello, se utilizó el lenguaje de programación Python para poder utilizar varios de sus paquetes de NLP los tres textos, mediante el uso de funciones que nos permitieron limpiar el texto de  los signos de puntuación, de tildes, de números, de palabras vacías o stopwords, para luego poder tokenizar el texto en palabras y así poder contar la frecuencia en la que aparecía cada palabra en el texto.
# 
# Una vez tokenizado el texto y contado la frecuencia de las palabras así como apoyándonos en las técnicas de biagramas  y concordancia , pudimos observar que la propuesta constitucional Chilena se asemeja más a la venezolana en el sentido de que se enfocan más es promover los derechos de las personas, el reconocimiento de los pueblos indígenas,  en la descentralización del poder, otorgándole más atribuciones a las regiones, así como una fuerte presencia de las facultades del banco central en ambos texto. Mientras que la constitución vigente en Chile parecía tener una fuerte prominencia de factores institucionales que aseguraran el orden y seguridad nacional así como el cumplimiento de la ley.
# 
# Por último, cabe destacar que las conclusiones sacadas de este análisis están sujetas a la frecuencia de las palabras en el texto, por lo tanto no se tiene el panorama completo en el cual están dichas palabras. Así que  para tener una mejor comprensión y extraer un mejor análisis de los textos se recomienda leerlos para poder entender en que contexto fueron utilizadas las palabras e interpretarlas de mejor manera.
# 

# # Referencias

# https://medium.com/towards-data-science/a-guide-to-cleaning-text-in-python-943356ac86ca
# 
# https://es.acervolima.com/agregar-etiquetas-de-valor-en-un-grafico-de-barras-de-matplotlib/
# 
# https://www.kaggle.com/code/danielamr/nlp-constitucion-chile
# 
# https://medium.com/qu4nt/reducir-el-n%C3%BAmero-de-palabras-de-un-texto-lematizaci%C3%B3n-y-radicalizaci%C3%B3n-stemming-con-python-965bfd0c69fa
# 
# https://www.delftstack.com/es/howto/python/remove-numbers-from-string-python/
# 
# https://www.delftstack.com/es/howto/python/remove-numbers-from-string-python/
# 
# https://www.geeksforgeeks.org/how-to-annotate-bars-in-barplot-with-matplotlib-in-python/
# 
# https://datascientest.com/es/como-generar-un-wordcloud-con-python
# 
# 
