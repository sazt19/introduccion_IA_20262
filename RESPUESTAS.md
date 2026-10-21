# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ …

_Mi respuesta / análisis:_


## T2 · ef, k y el tamaño de la cola

_Evidencia:_ …

_Mi respuesta / análisis:_


## T3 · Inserción frente a búsqueda

_Evidencia:_ …

_Mi respuesta / análisis:_


## T4 · El invariante de FAISSVectorStore

_Evidencia:_ …

_Mi respuesta / análisis:_


## T5 · El costo de la fuerza bruta

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ …

_Mi respuesta / análisis:_


## A2 · Tu propio corpus de documentos

_Evidencia:_ …

_Mi respuesta / análisis:_


## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ …

_Mi respuesta / análisis:_


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ …

_Mi respuesta / análisis:_


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ …

_Mi respuesta / análisis:_


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ …

_Mi respuesta / análisis:_


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ …

_Mi respuesta / análisis:_


## C2 · Un tercer paper: el RAG que explica HNSW

_Evidencia:_ …

_Mi respuesta / análisis:_


## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ …

_Mi respuesta / análisis:_


## C4 · Evaluación automática de la recuperación

_Evidencia:_ …

_Mi respuesta / análisis:_


## C5 · MMR frente a similitud pura

_Evidencia:_ …

_Mi respuesta / análisis:_


## C6 · Persistencia del índice en disco

_Evidencia:_ …

_Mi respuesta / análisis:_


Tarea|Notebook|Dónde cambias|Qué cambias
A2|	02b	Sección 3	Archivo JSON propio y ruta de load_documents
A3|	02b	Sección 4 (prompt)	Texto del system prompt
A4|	02b	Secciones 4 y 5	build_pipeline devuelve el vectorstore; chat() muestra scores
B2|	03	Celda nueva tras la Parte 4	Funciones build_store y build_chain
B3|	03	Parte 4 (prompt)	Texto del system prompt
B4, B5	03	Celdas nuevas tras el Bonus	Preguntas y ground truth propios
C2	03	Parte 1	Nuevo PDF en el diccionario PAPERS
C3	03 + 02b	Celdas nuevas	Interfaz Gradio con filtro por metadata
C5	03	Parte 4	search_type="mmr"
C6	03	Parte 3	QdrantClient(path=...)
