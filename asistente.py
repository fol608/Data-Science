import os, sys, time, random, queue
import numpy as np
import sounddevice as sd
import webrtcvad
import pyttsx3
import whisper
import llama3_1

# =============== CONFIGURACIÓN ===============
INPUT_DEVICE_INDEX = None  # Se detectará automáticamente
STREAM_SR = 44100          # frecuencia nativa del micro en portátiles
TARGET_SR = 16000          # frecuencia esperada por Whisper y VAD
FRAME_MS = 20              # duración de cada frame
current_stream_sr = STREAM_SR
current_frame_bytes = int(current_stream_sr * FRAME_MS / 1000) * 2  # int16 mono
WAKE_PHRASES = ["oye jarvis", "hola asistente", "ok asistente"]
LANG = "es"
WAKE_WINDOW_MS = 1200
UTTERANCE_MAX_SEC = 12
SILENCE_AFTER_SPEECH_MS = 800
MAX_TTS_CHARS = 550

# =============== FRASES ===============
WAKE_ACKS = ["A sus órdenes, señor.", "Sí, lo escucho.", "Adelante.", "Listo para brillar.", "Aquí estoy."]
EMPTY_HEARD = ["No he oído nada.", "Silencio absoluto. Fascinante.", "Creo que el micrófono se ha aburrido."]
DIDNT_UNDERSTAND = ["No he entendido. ¿Puede repetir?", "Eso fue… confuso. Inténtelo de nuevo.", "Nada claro. ¿Otra vez?"]
CLOSERS = ["Listo, señor.", "Hecho.", "Completado.", "Con elegancia, por supuesto."]

# =============== COLA DE AUDIO ===============
audio_q = queue.Queue()
pending_bytes = bytearray()

def audio_callback(indata, frames, time_info, status):
    try:
        if status:
            print("[Audio status]", status, file=sys.stderr)
        audio_q.put(bytes(indata))
    except Exception as e:
        print("[Callback error]", e, file=sys.stderr)

def read_frame(timeout=3.0):
    """Lee exactamente un frame (20 ms) del micr?fono"""
    global pending_bytes
    start = time.time()
    while len(pending_bytes) < current_frame_bytes:
        try:
            chunk = audio_q.get(timeout=0.5)
            pending_bytes.extend(chunk)
        except queue.Empty:
            if time.time() - start > timeout:
                return None
    frame = bytes(pending_bytes[:current_frame_bytes])
    del pending_bytes[:current_frame_bytes]
    return frame



# =============== UTILIDADES DE AUDIO ===============
def resample_to_16k(int16_bytes, source_sr=None):
    """Convierte audio del muestreo original a 16 kHz preservando la duracion."""
    if source_sr is None:
        source_sr = current_stream_sr

    if not int16_bytes or source_sr == TARGET_SR:
        return int16_bytes

    pcm = np.frombuffer(int16_bytes, dtype=np.int16).astype(np.float32)
    if pcm.size == 0:
        return b""

    target_len = int(round(pcm.size * TARGET_SR / source_sr))
    if target_len <= 0:
        return b""

    orig_idx = np.arange(pcm.size)
    target_idx = np.linspace(0, pcm.size, num=target_len, endpoint=False)
    resampled = np.interp(target_idx, orig_idx, pcm)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

vad = webrtcvad.Vad()
vad.set_mode(2)

def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, TARGET_SR)

# =============== WHISPER (GPU SI DISPONIBLE) ===============
print("[Whisper] Cargando modelo…")
try:
    model = whisper.load_model("small", device="cuda")
    print("[Whisper] GPU activa ✅")
except Exception as e:
    print("[Whisper] GPU no disponible:", e)
    model = whisper.load_model("small", device="cpu")

def transcribe_bytes(pcm_bytes, language=LANG):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(audio, language=language, verbose=False)
    return result.get("text", "").strip().lower()

def contains_wake(text):
    return any(w in text for w in WAKE_PHRASES)

# =============== TTS ===============
tts = pyttsx3.init()
for v in tts.getProperty('voices'):
    if "spanish" in v.name.lower() or "es_es" in getattr(v, "id", "").lower():
        tts.setProperty('voice', v.id)
        break
tts.setProperty('rate', 185)
tts.setProperty('volume', 0.9)

def speak(msg):
    msg = msg.strip()
    if len(msg) > MAX_TTS_CHARS:
        msg = msg[:MAX_TTS_CHARS].rsplit(" ", 1)[0] + "..."
    print("Jarvis:", msg)
    tts.say(msg)
    tts.runAndWait()

def speak_rand(options):
    speak(random.choice(options))

# =============== CAPTURA DE AUDIO ===============
def listen_until_silence(max_sec=UTTERANCE_MAX_SEC, silence_ms=SILENCE_AFTER_SPEECH_MS):
    collected = []
    total_ms = 0
    voiced_any = False
    silent_ms = 0
    while total_ms < max_sec * 1000:
        frame = read_frame()
        if frame is None:
            break
        frame16 = resample_to_16k(frame, current_stream_sr)
        speech = is_speech(frame16)
        collected.append(frame16)
        total_ms += FRAME_MS
        if speech:
            voiced_any = True
            silent_ms = 0
        else:
            silent_ms += FRAME_MS
            if voiced_any and silent_ms >= silence_ms:
                break
    return b"".join(collected)

# =============== VERIFICACIÓN DE PERMISOS ===============
def check_microphone_permissions():
    """Verifica si tenemos acceso al micrófono."""
    print("[Audio] Verificando permisos de micrófono...")
    
    # Intentar con sounddevice primero
    try:
        import sounddevice as sd
        test_recording = sd.rec(frames=512, samplerate=16000, channels=1, dtype='int16')
        sd.wait()
        print("[Audio] Permisos de micrófono verificados con sounddevice ✅")
        return True
    except Exception as e:
        print(f"[Audio] sounddevice falló: {e}")
    
    # Intentar con PyAudio como respaldo
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=512)
        data = stream.read(512)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("[Audio] Permisos de micrófono verificados con PyAudio ✅")
        return True
    except Exception as e:
        print(f"[Audio] PyAudio también falló: {e}")
    
    print("[Audio] No se pudo acceder al micrófono")
    print("[Audio] Posibles soluciones:")
    print("  1. Verificar que el micrófono esté conectado")
    print("  2. Comprobar configuración de privacidad de Windows")
    print("  3. Cerrar otras aplicaciones que usen el micrófono")
    print("  4. Reiniciar la aplicación como administrador")
    print("  5. Instalar PyAudio: pip install pyaudio")
    return False

# =============== DETECCIÓN DE MICRÓFONO ===============
def find_microphone():
    """Encuentra automáticamente un micrófono disponible, priorizando APIs estables."""
    import sounddevice as sd
    
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    
    print("[Audio] APIs de audio disponibles:")
    for i, api in enumerate(hostapis):
        print(f"  {i}: {api['name']} ({'por defecto' if i == sd.default.hostapi else ''})")
    
    print("\n[Audio] Dispositivos disponibles:")
    
    # Buscar dispositivos de entrada, priorizando APIs más compatibles
    input_devices = []
    preferred_apis = ['MME', 'Windows DirectSound', 'Windows WASAPI']  # MME primero
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            hostapi = hostapis[device['hostapi']]['name']
            print(f"  {i}: {device['name']} ({hostapi}) - {device['max_input_channels']} canales")
            input_devices.append((i, hostapi, device))
    
    if not input_devices:
        raise RuntimeError("No se encontraron dispositivos de entrada de audio")
    
    # Priorizar dispositivos con APIs más estables
    for api_name in preferred_apis:
        for device_id, hostapi, device_info in input_devices:
            if api_name in hostapi:
                print(f"[Audio] Seleccionando dispositivo {device_id} con API estable: {hostapi}")
                return device_id
    
    # Si no encontramos una API preferida, usar el primero disponible
    chosen = input_devices[0][0]
    print(f"[Audio] Usando primer dispositivo disponible: {chosen}")
    return chosen

# =============== STREAM DE AUDIO ===============
def open_stream():
    """Abre el stream de entrada con reintentos y ajusta buffers."""
    global current_stream_sr, current_frame_bytes, pending_bytes
    import sounddevice as sd

    # Configurar sounddevice para Windows
    sd.default.latency = "high"
    sd.default.dtype = ("int16", "int16")
    
    # Intentar usar MME primero (más compatible en Windows)
    try:
        mme_hostapi = None
        for i, api in enumerate(sd.query_hostapis()):
            if 'MME' in api['name']:
                mme_hostapi = i
                break
        if mme_hostapi is not None:
            sd.default.hostapi = mme_hostapi
            print(f"[Audio] Usando MME (API {mme_hostapi}) para máxima compatibilidad")
    except Exception as e:
        print(f"[Audio] No se pudo configurar MME: {e}")

    # Detectar micrófono automáticamente si no está especificado
    chosen_device = INPUT_DEVICE_INDEX
    if chosen_device is None:
        chosen_device = find_microphone()
    
    devices = sd.query_devices()
    if chosen_device >= len(devices):
        print(f"[Audio] Dispositivo {chosen_device} no existe, detectando automáticamente...")
        chosen_device = find_microphone()
    
    device_info = devices[chosen_device]
    hostapi_name = sd.query_hostapis()[device_info["hostapi"]]["name"]
    print(f"[Audio] Dispositivo {chosen_device}: {device_info['name']} ({hostapi_name})")

    # Probar diferentes frecuencias de muestreo
    sample_rates = [48000, 44100, 16000, 22050, STREAM_SR]  # 48kHz primero (más común en Windows)
    sr_to_use = None
    
    for sr in sample_rates:
        try:
            sd.check_input_settings(device=chosen_device, samplerate=sr,
                                    channels=1, dtype="int16")
            sr_to_use = sr
            print(f"[Audio] Frecuencia de muestreo soportada: {sr} Hz")
            break
        except Exception as e:
            print(f"[Audio] {sr} Hz no soportado: {e}")
    
    if sr_to_use is None:
        raise RuntimeError("No se pudo encontrar una frecuencia de muestreo compatible")

    frame_samples = int(sr_to_use * FRAME_MS / 1000)
    target_frame_bytes = frame_samples * 2
    print(f"[Audio] Intentando abrir dispositivo {chosen_device} a {sr_to_use} Hz...")

    # Configuraciones más conservadoras para Windows
    attempts = [
        ("auto_large", 2048, {}),
        ("auto_medium", 1024, {}),
        ("frame_exact", frame_samples, {}),
        ("auto_default", 0, {})
    ]

    with audio_q.mutex:
        audio_q.queue.clear()
    pending_bytes.clear()

    last_error = None
    for label, blocksize, extra_args in attempts:
        try:
            kwargs = {
                "device": chosen_device,
                "samplerate": sr_to_use,
                "channels": 1,
                "dtype": "int16",
                "callback": audio_callback,
                "latency": "high"
            }
            
            if blocksize is not None:
                kwargs["blocksize"] = blocksize
            
            kwargs.update(extra_args)
            
            stream = sd.RawInputStream(**kwargs)
            
            current_stream_sr = sr_to_use
            current_frame_bytes = target_frame_bytes
            pending_bytes.clear()
            with audio_q.mutex:
                audio_q.queue.clear()
            print(f"[Audio] Stream abierto exitosamente (modo {label}, blocksize={blocksize or 'auto'}).")
            return stream
            
        except Exception as e:
            last_error = e
            print(f"[Audio] Fallo abriendo stream (modo {label}): {e}")
    
    print(f"[Audio] Error final: {last_error}")
    print("[Audio] Sugerencias:")
    print("  1. Cerrar otras aplicaciones que usen el micrófono")
    print("  2. Verificar que el micrófono no esté siendo usado por otra app")
    print("  3. Reiniciar el servicio de audio de Windows")
    print("  4. Ejecutar como administrador")
    raise last_error

# =============== PYAUDIO COMO RESPALDO ===============
def open_stream_pyaudio():
    """Abre stream usando PyAudio como alternativa a sounddevice."""
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError("PyAudio no está instalado. Instalar con: pip install pyaudio")
    
    global current_stream_sr, current_frame_bytes, pending_bytes
    
    p = pyaudio.PyAudio()
    
    print("[PyAudio] Dispositivos disponibles:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  {i}: {info['name']} - {info['maxInputChannels']} canales")
    
    # Buscar un dispositivo de entrada funcional, priorizando MME
    device_to_use = None
    device_info = None
    
    # Primero buscar dispositivos MME (más compatibles)
    print("[PyAudio] Buscando dispositivos MME (más compatibles)...")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            host_api_info = p.get_host_api_info_by_index(info['hostApi'])
            if 'MME' in host_api_info['name']:
                try:
                    # Probar si el dispositivo MME funciona
                    test_stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                                       input=True, input_device_index=i, frames_per_buffer=2048)
                    test_stream.close()
                    device_to_use = i
                    device_info = info
                    print(f"[PyAudio] Dispositivo MME funcional: {info['name']}")
                    break
                except Exception as e:
                    print(f"[PyAudio] Dispositivo MME {i} falló: {e}")
    
    # Si no encontramos MME, probar cualquier dispositivo
    if device_to_use is None:
        print("[PyAudio] No se encontró MME funcional, probando otros...")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                try:
                    # Probar con configuración muy conservadora
                    test_stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050,
                                       input=True, input_device_index=i, frames_per_buffer=4096)
                    test_stream.close()
                    device_to_use = i
                    device_info = info
                    print(f"[PyAudio] Dispositivo funcional encontrado: {info['name']}")
                    break
                except Exception as e:
                    print(f"[PyAudio] Dispositivo {i} no funciona: {e}")
    
    if device_to_use is None:
        raise RuntimeError("No se encontró ningún dispositivo de entrada funcional en PyAudio")
    
    # Probar frecuencias de muestreo con configuración conservadora
    sample_rates = [22050, 44100, 16000, 48000]  # Empezar con 22kHz (más compatible)
    sr_to_use = 22050  # Por defecto más conservador
    
    for sr in sample_rates:
        try:
            # Verificar si la frecuencia es soportada con buffer grande
            stream_test = p.open(format=pyaudio.paInt16, channels=1, rate=sr, 
                               input=True, input_device_index=device_to_use, 
                               frames_per_buffer=4096)  # Buffer más grande
            stream_test.close()
            sr_to_use = sr
            print(f"[PyAudio] Frecuencia soportada: {sr} Hz")
            break
        except Exception as e:
            print(f"[PyAudio] {sr} Hz no soportado: {e}")
    
    current_stream_sr = sr_to_use
    frame_samples = int(sr_to_use * FRAME_MS / 1000)
    current_frame_bytes = frame_samples * 2
    
    class PyAudioStream:
        def __init__(self):
            self.p = p
            self.device_id = device_to_use
            print(f"[PyAudio] Abriendo stream en dispositivo {device_to_use} a {sr_to_use} Hz...")
            
            # Usar configuración muy conservadora para evitar WinError 6
            buffer_size = max(4096, frame_samples * 8)  # Buffer muy grande
            self.stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sr_to_use,
                input=True,
                input_device_index=device_to_use,
                frames_per_buffer=buffer_size,
                stream_callback=self.callback
            )
            pending_bytes.clear()
            with audio_q.mutex:
                audio_q.queue.clear()
            print(f"[PyAudio] Stream configurado exitosamente")
        
        def callback(self, in_data, frame_count, time_info, status):
            if status:
                print(f"[PyAudio status] {status}", file=sys.stderr)
            try:
                audio_q.put(in_data)
            except Exception as e:
                print(f"[PyAudio callback error] {e}", file=sys.stderr)
            return (None, pyaudio.paContinue)
        
        def __enter__(self):
            self.stream.start_stream()
            print("[PyAudio] Stream iniciado")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("[PyAudio] Cerrando stream...")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
    
    try:
        return PyAudioStream()
    except Exception as e:
        print(f"[PyAudio] Stream con callback falló: {e}")
        print("[PyAudio] Intentando modo síncrono...")
        return PyAudioStreamSync(p, device_to_use, sr_to_use, frame_samples)

class PyAudioStreamSync:
    """Stream PyAudio síncrono como último recurso."""
    def __init__(self, p, device_id, sample_rate, frame_samples):
        self.p = p
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.buffer_size = max(2048, frame_samples * 4)
        
        print(f"[PyAudio Sync] Configurando stream síncrono...")
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=self.buffer_size
        )
        
        # Limpiar colas
        pending_bytes.clear()
        with audio_q.mutex:
            audio_q.queue.clear()
        
        # Iniciar hilo de lectura
        import threading
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        
    def _read_loop(self):
        """Bucle de lectura en hilo separado."""
        while self.running:
            try:
                data = self.stream.read(self.buffer_size, exception_on_overflow=False)
                audio_q.put(data)
            except Exception as e:
                if self.running:  # Solo reportar si no estamos cerrando
                    print(f"[PyAudio Sync] Error de lectura: {e}", file=sys.stderr)
                time.sleep(0.01)
    
    def __enter__(self):
        self.read_thread.start()
        print("[PyAudio Sync] Stream síncrono iniciado")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("[PyAudio Sync] Cerrando stream síncrono...")
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def main_loop_pyaudio():
    """Bucle principal usando PyAudio."""
    debounce = 0
    while True:
        frame = read_frame()
        if frame is None:
            continue
        frame16 = resample_to_16k(frame, current_stream_sr)
        if not is_speech(frame16):
            continue

        # ventana para detectar wake word
        window = [frame16]
        remain = (WAKE_WINDOW_MS // FRAME_MS) - 1
        for _ in range(remain):
            f = read_frame()
            if f is None:
                break
            window.append(resample_to_16k(f, current_stream_sr))
        wake_audio = b"".join(window)

        wake_text = transcribe_bytes(wake_audio)
        print("[Wake debug] Texto detectado:", wake_text)

        if contains_wake(wake_text) and debounce == 0:
            debounce = 12
            print("🔔 Activado:", wake_text)
            speak_rand(WAKE_ACKS)

            utter = listen_until_silence()
            if not utter:
                speak_rand(EMPTY_HEARD)
                continue
            text = transcribe_bytes(utter)
            if not text:
                speak_rand(DIDNT_UNDERSTAND)
                continue

            print("🗣️ Tú:", text)
            respuesta = llama3_1.chat(text)
            if not respuesta:
                respuesta = "No he generado respuesta. Inquietante."
            if random.random() < 0.4:
                respuesta += " " + random.choice(CLOSERS)

            print("🤖 Jarvis:", respuesta)
            speak(respuesta)

        if debounce > 0:
            debounce -= 1



# =============== MAIN LOOP ===============
def main():
    print("Jarvis listo. Verificando acceso al micrófono...")
    
    # Verificar permisos antes de continuar
    if not check_microphone_permissions():
        print("[Error] No se puede acceder al micrófono. Saliendo...")
        return
    
    print("Di tu palabra de activación (por ejemplo, 'oye jarvis').")
    speak("Listo. Di tu palabra de activación.")
    
    while True:
        try:
            with open_stream() as stream:
                print("[Main] Stream de audio abierto ✅")
                speak("Micrófono operativo, señor.")
                debounce = 0
                while True:
                    frame = read_frame()
                    if frame is None:
                        continue
                    frame16 = resample_to_16k(frame, current_stream_sr)
                    if not is_speech(frame16):
                        continue

                    # ventana para detectar wake word
                    window = [frame16]
                    remain = (WAKE_WINDOW_MS // FRAME_MS) - 1
                    for _ in range(remain):
                        f = read_frame()
                        if f is None:
                            break
                        window.append(resample_to_16k(f, current_stream_sr))
                    wake_audio = b"".join(window)

                    wake_text = transcribe_bytes(wake_audio)
                    print("[Wake debug] Texto detectado:", wake_text)

                    if contains_wake(wake_text) and debounce == 0:
                        debounce = 12
                        print("🔔 Activado:", wake_text)
                        speak_rand(WAKE_ACKS)

                        utter = listen_until_silence()
                        if not utter:
                            speak_rand(EMPTY_HEARD)
                            continue
                        text = transcribe_bytes(utter)
                        if not text:
                            speak_rand(DIDNT_UNDERSTAND)
                            continue

                        print("🗣️ Tú:", text)
                        respuesta = llama3_1.chat(text)
                        if not respuesta:
                            respuesta = "No he generado respuesta. Inquietante."
                        if random.random() < 0.4:
                            respuesta += " " + random.choice(CLOSERS)

                        print("🤖 Jarvis:", respuesta)
                        speak(respuesta)

                    if debounce > 0:
                        debounce -= 1

        except Exception as e:
            print(f"[Error] Stream falló: {e}")
            
            # Intentar con PyAudio como alternativa para errores comunes de Windows
            error_str = str(e).lower()
            if any(err in error_str for err in ["winerror 6", "controlador no válido", "invalid handle", "device unavailable"]):
                print("[Main] Detectado error de controlador de Windows, cambiando a PyAudio...")
                try:
                    with open_stream_pyaudio() as stream:
                        print("[Main] Stream PyAudio abierto ✅")
                        speak("Micrófono operativo con PyAudio, señor.")
                        # Usar el mismo bucle principal pero con PyAudio
                        main_loop_pyaudio()
                        break  # Salir del bucle de reintentos si PyAudio funciona
                except Exception as e2:
                    print(f"[Error] PyAudio también falló: {e2}")
            
            print("[Main] Reintentando abrir el stream en 3 s...")
            time.sleep(3)

# =============== INICIO ===============
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSaliendo…")
        sys.exit(0)
