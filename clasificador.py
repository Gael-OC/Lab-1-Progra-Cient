import os
import math
import re
from nltk.corpus import stopwords

def procesarTexto(texto):
    texto = texto.lower()
    
    #usa regex para eliminar todo lo que no sea letra o espacio
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    tokens = texto.split()
    
    #eliminamos stopwords
    stopWords = set(stopwords.words('english'))
    tokensFiltrados = [palabra for palabra in tokens if palabra not in stopWords]
    
    return tokensFiltrados

def cargarDocumentos(rutaBase):
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
    totalPalabras = len(listaPalabras)
    if totalPalabras == 0:
        return {}
    
    conteoPalabras = {}
    for palabra in listaPalabras:
        conteoPalabras[palabra] = conteoPalabras.get(palabra, 0) + 1
    
    tf = {palabra: conteo / totalPalabras for palabra, conteo in conteoPalabras.items()}
    return tf

def calcularIdf(todosLosDocumentos):
    totalDocumentos = len(todosLosDocumentos)
    df = {} 
    
    for listaPalabras in todosLosDocumentos.values():
        palabrasUnicas = set(listaPalabras)
        for palabra in palabrasUnicas:
            df[palabra] = df.get(palabra, 0) + 1
            
    idf = {palabra: math.log(totalDocumentos / (1 + conteo)) for palabra, conteo in df.items()}
    return idf

def calcularSimilitud(vectorA, vectorB):
    palabrasComunes = set(vectorA.keys()) & set(vectorB.keys())
    productoPunto = sum(vectorA[p] * vectorB[p] for p in palabrasComunes)
    magnitudA = math.sqrt(sum(v**2 for v in vectorA.values()))
    magnitudB = math.sqrt(sum(v**2 for v in vectorB.values()))
    
    if magnitudA == 0 or magnitudB == 0:
        return 0.0
        
    return productoPunto / (magnitudA * magnitudB)

#definimos el número K de vecinos
kVecinos = 5

documentosBase = cargarDocumentos('dataset')
idfGeneral = calcularIdf(documentosBase)

vectoresBase = {}
for idDoc, palabras in documentosBase.items():
    tfDoc = calcularTf(palabras)
    vectoresBase[idDoc] = {p: tf * idfGeneral.get(p, 0) for p, tf in tfDoc.items()}

#textos de consulta para probar
consultas = [
    "The new M3 chip in the MacBook Pro is very fast.",
    "The pitcher for the Yankees had a great game.",
    "This new electric motorcycle has a long range.",
    "I need a new resistor and capacitor for my circuit board",
    "My old car needs a new transmission and brakes."
]

for i, textoConsulta in enumerate(consultas):
    print(f"--- Clasificando Consulta #{i+1} ---")
    print(f"Texto: '{textoConsulta[:]}'")
    
    #procesa y calcula el vector TF-IDF de la consulta
    palabrasConsulta = procesarTexto(textoConsulta)
    tfConsulta = calcularTf(palabrasConsulta)
    vectorConsulta = {p: tf * idfGeneral.get(p, 0) for p, tf in tfConsulta.items()}
    
    #calcula similitud y ordena resultados
    listaSimilitudes = []
    for idDoc, vectorDoc in vectoresBase.items():
        similitud = calcularSimilitud(vectorConsulta, vectorDoc)
        categoriaDoc = idDoc.split('/')[0]
        listaSimilitudes.append((similitud, categoriaDoc))
        
    #ordena la lista de mayor a menor similitud
    listaSimilitudes.sort(key=lambda x: x[0], reverse=True)
    
    #selecciona los K vecinos relevantes
    vecinosCercanos = listaSimilitudes[:kVecinos]
    
    #signa categoría por votación mayoritaria
    votos = {}
    for (sim, categoria) in vecinosCercanos:
        votos[categoria] = votos.get(categoria, 0) + 1
        
    #busca la categoría con más votos
    categoriaAsignada = max(votos, key=votos.get)
    
    print(f"Categoría asignada: {categoriaAsignada} (K={kVecinos})")
    print("Vecinos encontrados:")
    for (sim, categoria) in vecinosCercanos:
        print(f"  - Categoría: {categoria}, Similitud: {sim:.4f}")
    print("\n")