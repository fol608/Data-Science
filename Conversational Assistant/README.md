En este TFG se desarrolla un sistema para potenciar las capacidades de Pepper mediante la 
integración de algoritmos de Inteligencia Artificial modernos, adaptados a las limitaciones de 
hardware del robot. Se propone una arquitectura dividida en dos niveles: las funciones de 
respuesta inmediata (sensores, motricidad y control del diálogo) se conservan en el propio 
Pepper, mientras que las tareas de procesamiento intensivo (transcripción de voz y análisis 
facial) se delegan a servicios externos. Con Whisper se implementa la interpretación de 
comandos en español, y con DeepFace se detectan rasgos emocionales y demográficos de los 
usuarios. Todo ello se coordina a través de una API diseñada en FastAPI, lo que facilita la 
incorporación de nuevos modelos sin modificar la lógica interna de Pepper. 
Los ensayos demuestran que el robot mantiene una interacción fluida y prolonga su autonomía 
energética, al mismo tiempo que ofrece respuestas más naturales y adaptadas al estado 
emocional de la persona.


This TFG enhances Pepper´s functionality by integrating contemporary Artificial Intelligence 
techniques within the constraints of its onboard hardware. A dual-layer framework is 
introduced: Pepper handles all immediate-response tasks (sensor input, motion control, and 
dialogue management) locally, while computationally demanding processes (speech-to-text 
and facial analysis) run on external servers. Whisper is employed for real-time Spanish 
command transcription, and DeepFace is used to identify users´ emotional states and 
demographic features. Coordination between components is managed via a FastAPI-based 
REST interface, allowing for straightforward addition of new models without altering Pepper´s 
core. 
Experimental results indicate that Pepper sustains smooth interaction and preserves its battery 
life, while delivering more context-aware and empathetic responses.