import serial
import time
import numpy as np
import sounddevice as sd
import librosa
import random
import threading
from collections import deque
from configs.pixmob_conversion_funcs import bits_to_arduino_string
from configs.effect_definitions import base_color_effects, tail_codes, special_effects
import configs.config as cfg

# Setup for Arduino connection
arduino = serial.Serial(port=cfg.ARDUINO_SERIAL_PORT, baudrate=cfg.ARDUINO_BAUD_RATE, timeout=0.1)

# Parameters for real-time audio analysis
SAMPLE_RATE = 44100
CHUNK_DURATION = 0.2
FRAME_SIZE = 1024
HISTORY_SIZE = 15

# Buffers for audio data
audio_buffer = deque(maxlen=int(SAMPLE_RATE * CHUNK_DURATION))
beat_history = deque(maxlen=HISTORY_SIZE)

# Effect categories

color_effects = list(base_color_effects.keys()) + list(special_effects.keys())
fade_effects = ['FADE_1', 'FADE_2', 'FADE_4', 'FADE_5']
strobe_effects = ['SLOW_WHITE', 'SLOW_TURQUOISE', 'SLOW_ORANGE', 'SLOW_YELLOW']

low_freq_effects = ['RED_3', 'YELLOW_4', 'PULSE_RED']
mid_freq_effects = ['GREEN', 'SLOW_GREEN', 'YELLOW_GREEN']
high_freq_effects = ['BLUE', 'LIGHT_BLUE', 'MAGENTA_2']
# strobe_effects = ['SLOW_WHITE', 'SLOW_TURQUOISE', 'SLOW_ORANGE', 'SLOW_YELLOW']


def send_effect(main_effect, tail_code=None, brightness=255):
    if main_effect in base_color_effects:
        effect_bits = base_color_effects[main_effect] + tail_codes.get(tail_code, '')
    elif main_effect in special_effects:
        effect_bits = special_effects[main_effect]
    else:
        return

    # Adjust brightness based on intensity
    brightness = max(10, min(255, int(brightness)))
    effect_bits = [int(bit * (brightness / 255)) for bit in effect_bits]

    arduino_string = bits_to_arduino_string(effect_bits)
    arduino.write(bytes(arduino_string, 'utf-8'))
    print(f"Sent Effect: {main_effect} | Tail: {tail_code} | Brightness: {brightness}")


def advanced_audio_analysis(audio_data):
    y = audio_data.flatten()
    rms_energy = np.sqrt(np.mean(y ** 2))
    silence_threshold = np.median(np.abs(y)) * 1.5

    # Silence detection
    if rms_energy < silence_threshold:
        return None, None, 120, 10

    # Split frequency bands
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    low_energy = np.mean(stft[:512])
    mid_energy = np.mean(stft[512:1024])
    high_energy = np.mean(stft[1024:])

    # Dynamic color selection based on frequency
    if low_energy > (mid_energy + high_energy):
        effect = random.choice(low_freq_effects)
    elif mid_energy > (low_energy + high_energy):
        effect = random.choice(mid_freq_effects)
    else:
        effect = random.choice(high_freq_effects)

    # Calculate brightness based on energy
    max_energy = np.max(stft)
    brightness = (max_energy / np.max(y)) * 255

    # Beat detection
    tempo, _ = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE)
    if tempo == 0 or np.isnan(tempo):
        tempo = 120

    # Dynamic tail effect
    if high_energy > (low_energy + mid_energy):
        tail_code = random.choice(strobe_effects)
    else:
        tail_code = random.choice(fade_effects)

    return effect, tail_code, tempo, brightness


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())


def led_control_loop():
    while True:
        if len(audio_buffer) > 0:
            audio_data = np.concatenate(list(audio_buffer))
            effect, tail_code, tempo, brightness = advanced_audio_analysis(audio_data)

            if effect:
                send_effect(effect, tail_code, brightness)

            tempo = max(tempo, 1)
            time.sleep(max(60.0 / float(tempo) * 0.5, 0.1))
        else:
            time.sleep(0.05)


stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
stream.start()
threading.Thread(target=led_control_loop, daemon=True).start()

while True:
    time.sleep(1)
