"""
Suavizado temporal de landmarks con el filtro One Euro.

La Tasks API de MediaPipe (la vigente) no aplica ningun filtro a los landmarks,
asi que estos tiemblan fotograma a fotograma por el ruido de la deteccion. Este
modulo implementa el filtro One Euro, disenado justo para senales interactivas:
suaviza fuerte cuando la senal se mueve lento (la mano quieta en una letra
estatica queda firme) y suaviza poco cuando se mueve rapido (la J, la Z o la LL
no se retrasan). Un promedio movil simple no sirve porque agregaria retraso a las
letras con movimiento.

El mismo filtro se aplica en dos lugares que DEBEN coincidir:
  - Preprocesamiento: sobre cada secuencia capturada, antes de la escala y las
    ventanas, para que el Modelo A entrene con landmarks ya suaves.
  - Inferencia (demo y app): en linea, fotograma a fotograma.

Se aplica sobre el vector YA TRASLADADO (muneca en el origen), igual en ambos
lados. Las coordenadas que valen exactamente cero (muneca en el origen y mitad de
una mano ausente) se preservan en cero, para no romper el contrato que usa la
normalizacion por escala.

Referencia: G. Casiez, N. Roussel, D. Vogel, "1 Euro Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.
"""

import numpy as np


def _alpha(cutoff: float, dt: float):
    """Coeficiente del filtro pasa-bajos para una frecuencia de corte dada."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class FiltroUnEuro:
    """Filtro One Euro vectorizado para un vector de coordenadas.

    Mantiene estado entre llamadas, asi que sirve tanto en linea (un fotograma
    por llamada en la demo) como por lotes (recorriendo una secuencia). Cada
    coordenada se filtra de forma independiente con su propia frecuencia de corte
    adaptativa.
    """

    def __init__(self, fps: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 1.0, d_cutoff: float = 1.0):
        self.dt = 1.0 / float(fps)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = None

    def reiniciar(self) -> None:
        """Olvida el estado. Llamar cuando la mano desaparece, para que al volver
        el filtro arranque limpio y no interpole un salto."""
        self._x_prev = None
        self._dx_prev = None

    def filtrar(self, x, preservar_ceros: bool = True):
        """Filtra un vector de coordenadas y devuelve la version suavizada.

        Si `preservar_ceros`, las coordenadas que entran en exactamente cero salen
        en cero (muneca en el origen y mano ausente).
        """
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            salida = x.copy()
        else:
            dx = (x - self._x_prev) / self.dt
            a_d = _alpha(self.d_cutoff, self.dt)
            dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
            cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
            a = _alpha(cutoff, self.dt)  # un alpha por coordenada
            salida = a * x + (1.0 - a) * self._x_prev
            self._x_prev = salida
            self._dx_prev = dx_hat
        if preservar_ceros:
            salida = np.where(x == 0.0, 0.0, salida)
        return salida


def suavizar_secuencia(seq, fps: float = 30.0, min_cutoff: float = 1.0,
                       beta: float = 1.0, d_cutoff: float = 1.0):
    """Suaviza una secuencia (T, D) recorriendola de forma causal, como en vivo.

    Usa un filtro nuevo por secuencia, para que cada toma se suavice de forma
    independiente (igual que cuando la mano aparece en la demo). Preserva los
    ceros. Devuelve un array float32 de la misma forma.
    """
    seq = np.asarray(seq, dtype=np.float32)
    filtro = FiltroUnEuro(fps, min_cutoff, beta, d_cutoff)
    salida = np.empty_like(seq)
    for i in range(len(seq)):
        salida[i] = filtro.filtrar(seq[i])
    return salida.astype(np.float32)
