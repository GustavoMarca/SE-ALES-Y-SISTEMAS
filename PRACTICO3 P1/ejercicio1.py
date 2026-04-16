import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Parámetros de la señal
# ---------------------------------
fs = 1000          # Frecuencia de muestreo [Hz]
T = 1              # Duración de la señal [s]
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Señal: x(t) = sin(2*pi*40*t)
x = np.sin(2 * np.pi * 40 * t)

# ---------------------------------
# 2. FFT
# ---------------------------------
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, d=1/fs)

# Tomamos solo la mitad positiva del espectro
mitad = N // 2
f_positivas = frecuencias[:mitad]
magnitud = (2 / N) * np.abs(X_fft[:mitad])

# Encontrar la frecuencia principal
indice_pico = np.argmax(magnitud)
frecuencia_principal = f_positivas[indice_pico]
amplitud_principal = magnitud[indice_pico]

print("Frecuencia principal detectada:", frecuencia_principal, "Hz")
print("Amplitud aproximada:", amplitud_principal)

# ---------------------------------
# 3. Gráficas
# ---------------------------------
plt.figure(figsize=(10, 8))

# Señal en el tiempo
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("Señal en el dominio del tiempo: x(t) = sin(2π40t)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# Espectro FFT
plt.subplot(2, 1, 2)
plt.stem(f_positivas, magnitud, basefmt=" ")
plt.axvline(frecuencia_principal, linestyle='--')
plt.title("Espectro de frecuencias usando FFT")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 100)   # para ver mejor el pico en 40 Hz
plt.grid(True)

plt.tight_layout()
plt.show()