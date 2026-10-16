# Actividad de Clase: Analizando Agentes de IA con Hugging Face Spaces

## Objetivo

Explorar aplicaciones reales de Inteligencia Artificial en **Hugging
Face Spaces** y analizarlas desde la perspectiva de los **agentes
racionales**.

Al finalizar la actividad, los estudiantes deberán ser capaces de:

-   Identificar los componentes **PEAS** de un agente.
-   Clasificar las propiedades del entorno.
-   Proponer qué tipo de programa de agente podría implementarse detrás
    del sistema.
-   Justificar sus respuestas.

------------------------------------------------------------------------

## Instrucciones

1.  Ingresen a **https://huggingface.co/spaces**.
2.  Exploren diferentes Spaces.
3.  Seleccionen uno que les parezca interesante.
4.  Interactúen con el sistema durante algunos minutos.
5.  Completen la siguiente ficha de análisis.

------------------------------------------------------------------------

# Ficha de análisis

## 1. Nombre del Space

**Nombre:** Z Image Turbo

**Enlace:** https://huggingface.co/spaces/mrfakename/Z-Image-Turbo

------------------------------------------------------------------------

## 2. ¿Qué hace el agente?

Describa en dos o tres líneas cuál es la función del sistema.

La función principal del sistema es a partir de una descripción dada, se genera una imgaen con las caracteristicass dadas en la descripcion. 
------------------------------------------------------------------------

## 3. Análisis PEAS

  Elemento          Respuesta
  ----------------- ----------------------------------------------------
  **Performance**   ¿Qué significa que el agente haga bien su trabajo?
  En este caso, se genera una imagen lo más cercana posible a la descripción dada por el usuario. 
  
  **Environment**   ¿Con qué interactúa el agente?
  El usuario escribiendo el prompt, la inteligencia usada para la generación de imagen 
  
  **Actuators**     ¿Qué acciones produce?
  Generar la imagen
  
  **Sensors**       ¿Qué información recibe como entrada?
  La descripción dada por el usuario

------------------------------------------------------------------------

## 4. Clasificación del entorno

Complete la siguiente tabla y justifique brevemente cada respuesta.

  Propiedad      Clasificación     Justificación
  -------------- ----------------- ---------------
  Observable     Total / Parcial  - Parcial: El agente no sabe ni tiene mas informacion que el prompt escrito por el usuario, por lo que no puede saberlo.  
  Determinista   Sí / No  - No: El restultado cambia cada vez que es ejecutado, asi sea el mismo prompt. 
  Episódico      Sí / No - No      
  Estático       Sí / No  - No         
  Discreto       Sí / No  - Si       
  Conocido       Sí / No  - Conocido         

------------------------------------------------------------------------

## 5. ¿Qué tipo de programa de agente creen que es?

Seleccione la opción que consideren más adecuada y explique por qué.

-   Agente de reflejo simple
-   Agente basado en modelo
-   Agente basado en objetivos
-   Agente basado en utilidad
-   Agente con aprendizaje

Considero que en este caso podría ser un agente de reflejo simple, ya que simplemente está tomando la descripción que se le da y generando una imagen, no tiene en cuenta diferentes aspectos o reglas adicionales (que se conozcan) para lograr el objetivo. 

> **Importante:** No existe una única respuesta correcta. Lo importante
> es justificar la elección a partir del comportamiento observado.

------------------------------------------------------------------------

# Discusión en clase

Después de las presentaciones, discutiremos preguntas como:

-   ¿Dos Spaces diferentes pueden compartir el mismo tipo de entorno?
-   ¿Es posible saber con certeza qué tipo de agente implementa un Space
    únicamente observándolo?
-   ¿Qué diferencia existe entre el comportamiento observable de un
    agente y su implementación interna?

------------------------------------------------------------------------

# Reto adicional

Encuentre un Space que pueda clasificarse como:

1.  **Totalmente observable, determinista y episódico.**
2.  **Parcialmente observable, estocástico y secuencial.**

Justifique su respuesta.

------------------------------------------------------------------------

# Rúbrica (10 puntos)

| Criterio | Puntos |
|-----------|:------:|
| Descripción correcta del Space | 2 |
| Identificación de PEAS | 3 |
| Clasificación del entorno | 3 |
| Justificación del tipo de agente | 2 |
| **Total** | **10** |

------------------------------------------------------------------------

