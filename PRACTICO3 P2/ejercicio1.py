import wave
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Cargar audio WAV
# -----------------------------------
archivo = "audio_limpio.wav"   # coloca este archivo en la misma carpeta del script

with wave.open(archivo, 'rb') as wav:
    fs = wav.getframerate()        # frecuencia de muestreo
    n_muestras = wav.getnframes()  # número de muestras
    n_canales = wav.getnchannels() # mono o estéreo
    sampwidth = wav.getsampwidth() # bytes por muestra
    datos = wav.readframes(n_muestras)

# Convertir a arreglo numpy según el tipo de dato
if sampwidth == 1:
    audio = np.frombuffer(datos, dtype=np.uint8)
    audio = audio.astype(np.float32) - 128
elif sampwidth == 2:
    audio = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
elif sampwidth == 4:
    audio = np.frombuffer(datos, dtype=np.int32).astype(np.float32)
else:
    raise ValueError("Formato de audio no soportado")

# Si es estéreo, tomar un solo canal
if n_canales == 2:
    audio = audio[::2]

# Normalizar y quitar componente DC
audio = audio / np.max(np.abs(audio))
audio = audio - np.mean(audio)

# Vector de tiempo
t = np.arange(len(audio)) / fs

# -----------------------------------
# 2. FFT
# -----------------------------------
N = len(audio)

# ventana para mejorar el espectro
ventana = np.hanning(N)
audio_win = audio * ventana

X_fft = np.fft.rfft(audio_win)
frecuencias = np.fft.rfftfreq(N, d=1/fs)
magnitud = np.abs(X_fft)

# Ignorar frecuencias muy bajas para hallar la dominante
mask = frecuencias >= 20
freq_filtradas = frecuencias[mask]
mag_filtrada = magnitud[mask]

indice_pico = np.argmax(mag_filtrada)
frecuencia_dominante = freq_filtradas[indice_pico]

print("Frecuencia de muestreo:", fs, "Hz")
print("Frecuencia dominante aproximada:", frecuencia_dominante, "Hz")

# -----------------------------------
# 3. Gráficas
# -----------------------------------
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, audio)
plt.title("Señal de audio en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frecuencias, magnitud)
plt.title("Espectro de frecuencias (FFT)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 2000)   # para ver mejor la zona útil
plt.grid(True)

plt.tight_layout()
plt.show()