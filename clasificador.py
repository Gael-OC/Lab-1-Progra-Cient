import os
import math

def procesarTexto(texto):
    texto = texto.lower()

    textoLimpio = ""
    for caracter in texto:
        if 'a' <= caracter <= 'z' or caracter == ' ':
            textoLimpio += caracter
            
    #devuelve una lista de palabras del texto limpio
    return textoLimpio.split()

def cargarDocumentos(rutaBase):
    #crea un diccionario para guardar los textos procesados
    #la clave será "categoria/archivo.txt" y el valor la lista de palabras
    documentos = {}
    
    for categoria in os.listdir(rutaBase):
        rutaCategoria = os.path.join(rutaBase, categoria)
        if os.path.isdir(rutaCategoria):
            for archivoNombre in os.listdir(rutaCategoria):
                idDocumento = f"{categoria}/{archivoNombre}"
                rutaArchivo = os.path.join(rutaCategoria, archivoNombre)
                with open(rutaArchivo, 'r', encoding='utf-8', errors='ignore') as archivo:
                    documentos[idDocumento] = procesarTexto(archivo.read())
                    
    return documentos

def calcularTf(listaPalabras):
    #calcula la frecuencia de término (TF) para un documento
    totalPalabras = len(listaPalabras)
    if totalPalabras == 0:
        return {}
    
    #cuenta cuántas veces aparece cada palabra
    conteoPalabras = {}
    for palabra in listaPalabras:
        conteoPalabras[palabra] = conteoPalabras.get(palabra, 0) + 1
    
    #aplica la fórmula TF: (conteo de palabra) / (total de palabras)
    tf = {palabra: conteo / totalPalabras for palabra, conteo in conteoPalabras.items()}
    return tf

def calcularIdf(todosLosDocumentos):
    #calcula la frecuencia inversa de documento (IDF) para todo el corpus
    totalDocumentos = len(todosLosDocumentos)
    df = {} #diccionario para guardar el Document Frequency (DF) de cada palabra
    
    #recorre cada documento para contar en cuántos aparece cada palabra
    for listaPalabras in todosLosDocumentos.values():
        palabrasUnicas = set(listaPalabras)
        for palabra in palabrasUnicas:
            df[palabra] = df.get(palabra, 0) + 1
            
    #aplica la fórmula IDF: log(total_documentos / (1 + df de la palabra))
    idf = {palabra: math.log(totalDocumentos / (1 + conteo)) for palabra, conteo in df.items()}
    return idf

def calcularSimilitud(vectorA, vectorB):
    #calcula la similitud del coseno entre dos vectores TF-IDF
    palabrasComunes = set(vectorA.keys()) & set(vectorB.keys())
    
    #producto punto: suma de las multiplicaciones de los pesos de palabras comunes
    productoPunto = sum(vectorA[p] * vectorB[p] for p in palabrasComunes)
    
    #magnitud del vector A: raíz cuadrada de la suma de sus pesos al cuadrado
    magnitudA = math.sqrt(sum(v**2 for v in vectorA.values()))
    
    #magnitud del vector B
    magnitudB = math.sqrt(sum(v**2 for v in vectorB.values()))
    
    #evita la división por cero si algún vector está vacío
    if magnitudA == 0 or magnitudB == 0:
        return 0.0
        
    return productoPunto / (magnitudA * magnitudB)


#carga todos los documentos de la carpeta 'dataset'
documentosBase = cargarDocumentos('dataset')

#calcula el IDF una sola vez para todo el conjunto de datos
idfGeneral = calcularIdf(documentosBase)

#calcula y guarda el vector TF-IDF para cada documento de la base
vectoresBase = {}
for idDoc, palabras in documentosBase.items():
    tfDoc = calcularTf(palabras)
    vectoresBase[idDoc] = {p: tf * idfGeneral.get(p, 0) for p, tf in tfDoc.items()}


consultas = [
    "NASA is planning a new mission to the moon and Mars.",
    "The hockey team won the finals in overtime.",
    "This new medicine could help fight cancer cells.",
    "This new GPU can render amazing 3D graphics.",
    "The new MacBook Air with the M3 processor has two Thunderbolt ports."
]

for i, textoConsulta in enumerate(consultas):
    print(f"--- Clasificando Consulta #{i+1} ---")
    print(f"Texto: '{textoConsulta[:]}'")
    
    #procesa y calcula el vector TF-IDF de la consulta
    palabrasConsulta = procesarTexto(textoConsulta)
    tfConsulta = calcularTf(palabrasConsulta)
    vectorConsulta = {p: tf * idfGeneral.get(p, 0) for p, tf in tfConsulta.items()}
    
    #compara la consulta con todos los documentos de la base
    mejorSimilitud = -1.0
    docMasSimilar = ""
    for idDoc, vectorDoc in vectoresBase.items():
        similitud = calcularSimilitud(vectorConsulta, vectorDoc)
        if similitud > mejorSimilitud:
            mejorSimilitud = similitud
            docMasSimilar = idDoc
            
    #signa la categoría del documento más parecido
    categoria = docMasSimilar.split('/')[0]
    
    print(f"Categoría asignada: {categoria}")
    print(f"(Documento más similar: '{docMasSimilar}', Similitud: {mejorSimilitud:.4f})\n")