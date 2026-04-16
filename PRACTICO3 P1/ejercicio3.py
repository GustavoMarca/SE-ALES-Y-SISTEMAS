import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Parámetros
# -----------------------------
fs = 1000          # Frecuencia de muestreo [Hz]
T = 1              # Duración [s]
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Frecuencias de la señal
f1 = 50
f2 = 200

# Señal compuesta
x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# -----------------------------
# 2. FFT de la señal original
# -----------------------------
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, d=1/fs)

# Parte positiva del espectro
mitad = N // 2
f_positivas = frecuencias[:mitad]
magnitud = (2 / N) * np.abs(X_fft[:mitad])

# -----------------------------
# 3. Eliminar la frecuencia menor (50 Hz)
# -----------------------------
X_filtrada = X_fft.copy()

# margen para eliminar la frecuencia de 50 Hz
margen = 2

# eliminar tanto +50 Hz como -50 Hz
indices_50 = np.where(np.abs(np.abs(frecuencias) - f1) < margen)
X_filtrada[indices_50] = 0

# Reconstrucción de la señal
x_filtrada = np.fft.ifft(X_filtrada).real

# Espectro después del filtrado
magnitud_filtrada = (2 / N) * np.abs(X_filtrada[:mitad])

# -----------------------------
# 4. Gráficas
# -----------------------------
plt.figure(figsize=(12, 10))

# Señal original
plt.subplot(2, 2, 1)
plt.plot(t, x)
plt.title("Señal original en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid(True)

# Espectro original
plt.subplot(2, 2, 2)
plt.stem(f_positivas, magnitud, basefmt=" ")
plt.title("Espectro de frecuencias original")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 300)
plt.grid(True)

# Señal filtrada
plt.subplot(2, 2, 3)
plt.plot(t, x_filtrada, color="orange")
plt.title("Señal después de eliminar 50 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid(True)

# Espectro filtrado
plt.subplot(2, 2, 4)
plt.stem(f_positivas, magnitud_filtrada, basefmt=" ")
plt.title("Espectro después de eliminar 50 Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 300)
plt.grid(True)

plt.tight_layout()
plt.show()