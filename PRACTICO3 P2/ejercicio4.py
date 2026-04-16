import wave
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Cargar audio con ruido
# -----------------------------------
archivo = "audio_con_ruido.wav"

with wave.open(archivo, 'rb') as wav:
    fs = wav.getframerate()
    n_muestras = wav.getnframes()
    n_canales = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    datos = wav.readframes(n_muestras)

# Convertir a numpy
if sampwidth == 1:
    audio = np.frombuffer(datos, dtype=np.uint8).astype(np.float32)
    audio = audio - 128
elif sampwidth == 2:
    audio = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
elif sampwidth == 4:
    audio = np.frombuffer(datos, dtype=np.int32).astype(np.float32)
else:
    raise ValueError("Formato no soportado")

# Si es estéreo, convertir a mono
if n_canales == 2:
    audio = audio.reshape(-1, 2).mean(axis=1)

# Normalizar
audio = audio / np.max(np.abs(audio))
audio = audio - np.mean(audio)

t = np.arange(len(audio)) / fs
N = len(audio)

# -----------------------------------
# 2. FFT del audio original
# -----------------------------------
X_fft = np.fft.rfft(audio)
frecuencias = np.fft.rfftfreq(N, d=1/fs)
magnitud = np.abs(X_fft)

# -----------------------------------
# 3. Filtro pasa-bajo en frecuencia
# -----------------------------------
fc = 1000   # frecuencia de corte en Hz

X_filtrada = X_fft.copy()
X_filtrada[frecuencias > fc] = 0

# Reconstrucción del audio filtrado
audio_filtrado = np.fft.irfft(X_filtrada, n=N)
audio_filtrado = audio_filtrado / np.max(np.abs(audio_filtrado))

magnitud_filtrada = np.abs(X_filtrada)

# -----------------------------------
# 4. Guardar audio filtrado
# -----------------------------------
audio_salida = (audio_filtrado * 32767).astype(np.int16)

with wave.open("audio_filtrado_pasabajo.wav", "wb") as salida:
    salida.setnchannels(1)
    salida.setsampwidth(2)
    salida.setframerate(fs)
    salida.writeframes(audio_salida.tobytes())

# -----------------------------------
# 5. Gráficas
# -----------------------------------
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, audio)
plt.title("Audio con ruido en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(frecuencias, magnitud)
plt.title("Espectro original")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 3000)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, audio_filtrado, color="orange")
plt.title("Audio filtrado pasa-bajo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(frecuencias, magnitud_filtrada, color="green")
plt.title("Espectro filtrado")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 3000)
plt.grid(True)

plt.tight_layout()
plt.show()