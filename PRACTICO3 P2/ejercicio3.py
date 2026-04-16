import wave
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Cargar audio WAV
# -----------------------------------
archivo = "audio_limpio.wav"

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
    raise ValueError("Formato de audio no soportado")

# Si fuera estéreo, pasar a mono
if n_canales == 2:
    audio = audio.reshape(-1, 2).mean(axis=1)

# Normalizar y quitar DC
audio = audio / np.max(np.abs(audio))
audio = audio - np.mean(audio)

t = np.arange(len(audio)) / fs
N = len(audio)

# -----------------------------------
# 2. FFT
# -----------------------------------
X_fft = np.fft.rfft(audio)
frecuencias = np.fft.rfftfreq(N, d=1/fs)
magnitud = np.abs(X_fft)

# Frecuencia dominante
mask = frecuencias >= 20
freq_validas = frecuencias[mask]
mag_validas = magnitud[mask]
indice_pico = np.argmax(mag_validas)
frecuencia_dominante = freq_validas[indice_pico]

print("Frecuencia dominante aproximada:", round(frecuencia_dominante, 2), "Hz")

# -----------------------------------
# 3. Filtro pasa-bajo en frecuencia
# -----------------------------------
fc = 1000  # frecuencia de corte en Hz

X_filtrada = X_fft.copy()
X_filtrada[frecuencias > fc] = 0

# Reconstrucción en tiempo
audio_filtrado = np.fft.irfft(X_filtrada, n=N)

# Normalizar salida
audio_filtrado = audio_filtrado / np.max(np.abs(audio_filtrado))

# FFT del audio filtrado
magnitud_filtrada = np.abs(X_filtrada)

# -----------------------------------
# 4. Guardar audio filtrado
# -----------------------------------
audio_guardar = (audio_filtrado * 32767).astype(np.int16)

with wave.open("audio_filtrado_pasabajo.wav", "wb") as wav_out:
    wav_out.setnchannels(1)
    wav_out.setsampwidth(2)
    wav_out.setframerate(fs)
    wav_out.writeframes(audio_guardar.tobytes())

# -----------------------------------
# 5. Gráficas
# -----------------------------------
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, audio)
plt.title("Audio original en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(frecuencias, magnitud)
plt.title("Espectro original (FFT)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 3000)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, audio_filtrado, color="orange")
plt.title("Audio filtrado en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(frecuencias, magnitud_filtrada, color="green")
plt.title("Espectro filtrado (pasa-bajo)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 3000)
plt.grid(True)

plt.tight_layout()
plt.show()