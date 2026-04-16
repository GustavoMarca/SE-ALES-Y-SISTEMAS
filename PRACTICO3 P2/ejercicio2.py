import wave
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Cargar el audio WAV
# -----------------------------------
archivo = "audio_con_ruido.wav"

with wave.open(archivo, 'rb') as wav:
    fs = wav.getframerate()        # frecuencia de muestreo
    n_muestras = wav.getnframes()  # número de muestras
    n_canales = wav.getnchannels() # mono o estéreo
    sampwidth = wav.getsampwidth() # bytes por muestra
    datos = wav.readframes(n_muestras)

# Convertir a numpy según el formato
if sampwidth == 1:
    audio = np.frombuffer(datos, dtype=np.uint8).astype(np.float32)
    audio = audio - 128
elif sampwidth == 2:
    audio = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
elif sampwidth == 4:
    audio = np.frombuffer(datos, dtype=np.int32).astype(np.float32)
else:
    raise ValueError("Formato de audio no soportado")

# Si es estéreo, convertir a mono promediando canales
if n_canales == 2:
    audio = audio.reshape(-1, 2).mean(axis=1)

# Normalizar
audio = audio / np.max(np.abs(audio))

# Quitar componente DC
audio = audio - np.mean(audio)

# Vector de tiempo
t = np.arange(len(audio)) / fs

# -----------------------------------
# 2. FFT
# -----------------------------------
N = len(audio)

# Aplicar ventana para mejorar el espectro
ventana = np.hanning(N)
audio_win = audio * ventana

X_fft = np.fft.rfft(audio_win)
frecuencias = np.fft.rfftfreq(N, d=1/fs)
magnitud = np.abs(X_fft)

# Frecuencia dominante (ignorando frecuencias muy bajas)
mask = frecuencias >= 20
freq_utiles = frecuencias[mask]
mag_utiles = magnitud[mask]

indice_pico = np.argmax(mag_utiles)
frecuencia_dominante = freq_utiles[indice_pico]

print("Frecuencia de muestreo:", fs, "Hz")
print("Frecuencia dominante aproximada:", round(frecuencia_dominante, 2), "Hz")

# -----------------------------------
# 3. Gráficas
# -----------------------------------
plt.figure(figsize=(12, 8))

# Señal en el tiempo
plt.subplot(2, 1, 1)
plt.plot(t, audio)
plt.title("Señal de audio con ruido en el dominio del tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# Espectro
plt.subplot(2, 1, 2)
plt.plot(frecuencias, magnitud)
plt.title("Espectro de frecuencias (FFT)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 2000)
plt.grid(True)

plt.tight_layout()
plt.show()