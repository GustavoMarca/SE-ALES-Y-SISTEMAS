import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Parámetros
# -----------------------------
fs = 1000          # Frecuencia de muestreo
T = 1              # Duración de la señal en segundos
t = np.linspace(0, T, int(fs * T), endpoint=False)

# -----------------------------
# 2. Frecuencia desconocida aleatoria
# -----------------------------
frecuencia_real = np.random.randint(50, 201)   # entre 50 y 200 Hz
x = np.sin(2 * np.pi * frecuencia_real * t)

# -----------------------------
# 3. FFT
# -----------------------------
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, d=1/fs)

# Solo frecuencias positivas
mitad = N // 2
f_positivas = frecuencias[:mitad]
magnitud = (2 / N) * np.abs(X_fft[:mitad])

# Identificar frecuencia dominante
indice_pico = np.argmax(magnitud)
frecuencia_dominante = f_positivas[indice_pico]

print("Frecuencia real generada:", frecuencia_real, "Hz")
print("Frecuencia dominante detectada con FFT:", frecuencia_dominante, "Hz")

# -----------------------------
# 4. Gráficas
# -----------------------------
plt.figure(figsize=(10, 8))

# Señal en el tiempo
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("Señal en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid(True)

# Espectro de frecuencias
plt.subplot(2, 1, 2)
plt.stem(f_positivas, magnitud, basefmt=" ")
plt.title("Espectro de frecuencias usando FFT")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 250)
plt.grid(True)

plt.tight_layout()
plt.show()