"""
Paquete `capture`: sistema de grabacion del dataset de landmarks.

Construye la base de datos de senas a partir de la camara. Graba, extrae los
landmarks con MediaPipe en el momento y descarta el video crudo; lo unico que
persiste es el CSV de landmarks.

Puntos de entrada (ejecutar desde la carpeta training/):

    python capture/captura_estatica.py persona_01
    python capture/captura_dinamica.py persona_01
"""
