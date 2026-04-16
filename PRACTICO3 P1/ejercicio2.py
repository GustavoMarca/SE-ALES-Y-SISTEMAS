import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Parámetros
# -----------------------------
fs = 1000          # Frecuencia de muestreo [Hz]
T = 1              # Duración [s]
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Para que el ruido sea siempre el mismo al ejecutar
np.random.seed(42)

# -----------------------------
# 2. Señales
# -----------------------------
x_limpia = np.sin(2 * np.pi * 60 * t)
ruido = 0.5 * np.random.randn(len(t))
x = x_limpia + ruido

# -----------------------------
# 3. FFT
# -----------------------------
N = len(x)

X_limpia = np.fft.fft(x_limpia)
X_ruidosa = np.fft.fft(x)

frecuencias = np.fft.fftfreq(N, d=1/fs)

# Solo parte positiva
mitad = N // 2
f_positivas = frecuencias[:mitad]

magnitud_limpia = (2 / N) * np.abs(X_limpia[:mitad])
magnitud_ruidosa = (2 / N) * np.abs(X_ruidosa[:mitad])

# Frecuencia principal de la señal con ruido
indice_pico = np.argmax(magnitud_ruidosa)
frecuencia_principal = f_positivas[indice_pico]

print("Frecuencia principal detectada:", frecuencia_principal, "Hz")

# -----------------------------
# 4. Filtrado simple en frecuencia
#    (mantener solo cerca de 60 Hz)
# -----------------------------
ancho_banda = 5   # conservar de 55 a 65 Hz aprox.

mascara = np.abs(np.abs(frecuencias) - 60) <= ancho_banda

X_filtrada = np.zeros_like(X_ruidosa, dtype=complex)
X_filtrada[mascara] = X_ruidosa[mascara]

x_filtrada = np.fft.ifft(X_filtrada).real
magnitud_filtrada = (2 / N) * np.abs(X_filtrada[:mitad])

# -----------------------------
# 5. Gráficas
# -----------------------------
plt.figure(figsize=(12, 12))

# Señal limpia
plt.subplot(3, 2, 1)
plt.plot(t, x_limpia)
plt.title("Señal limpia en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid(True)

# Señal con ruido
plt.subplot(3, 2, 2)
plt.plot(t, x)
plt.title("Señal con ruido en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid(True)

# Espectro señal limpia
plt.subplot(3, 2, 3)
plt.stem(f_positivas, magnitud_limpia, basefmt=" ")
plt.title("FFT de la señal limpia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 150)
plt.grid(True)

# Espectro señal con ruido
plt.subplot(3, 2, 4)
plt.stem(f_positivas, magnitud_ruidosa, basefmt=" ")
plt.title("FFT de la señal con ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 150)
plt.grid(True)

# Señal filtrada
plt.subplot(3, 2, 5)
plt.plot(t, x, label="Con ruido", alpha=0.5)
plt.plot(t, x_filtrada, label="Filtrada")
plt.title("Señal con ruido vs señal filtrada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid(True)
plt.legend()

# Espectro filtrado
plt.subplot(3, 2, 6)
plt.stem(f_positivas, magnitud_filtrada, basefmt=" ")
plt.title("FFT después del filtrado")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 150)
plt.grid(True)

plt.tight_layout()
plt.show()