import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Función para generar señal y calcular FFT
# ----------------------------------------
def analizar_fft(fs, T=1, fase=np.pi/4):
    t = np.linspace(0, T, int(fs*T), endpoint=False)

    # Señales de 250 Hz y 1000 Hz
    x1 = np.sin(2*np.pi*250*t + fase)
    x2 = np.sin(2*np.pi*1000*t + fase)

    # Señal compuesta
    x = x1 + x2

    # FFT
    N = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, d=1/fs)
    magnitud = np.abs(X) * 2 / N

    return t, x, f, magnitud

# Caso 1: fs = 500 Hz
t1, x_fs500, f1, mag1 = analizar_fft(fs=500)

# Caso 2: fs = 3000 Hz
t2, x_fs3000, f2, mag2 = analizar_fft(fs=3000)

# ----------------------------------------
# Gráficas
# ----------------------------------------
plt.figure(figsize=(12, 10))

# Señal en tiempo para fs=500
plt.subplot(2, 2, 1)
plt.plot(t1[:100], x_fs500[:100])
plt.title("Señal muestreada con Fs = 500 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# FFT para fs=500
plt.subplot(2, 2, 2)
plt.stem(f1, mag1, basefmt=" ")
plt.title("Espectro con Fs = 500 Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 300)
plt.grid(True)

# Señal en tiempo para fs=3000
plt.subplot(2, 2, 3)
plt.plot(t2[:300], x_fs3000[:300])
plt.title("Señal muestreada con Fs = 3000 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# FFT para fs=3000
plt.subplot(2, 2, 4)
plt.stem(f2, mag2, basefmt=" ")
plt.title("Espectro con Fs = 3000 Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 1500)
plt.grid(True)

plt.tight_layout()
plt.show()