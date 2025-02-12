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
time.sleep(2.5)

# Parameters for real-time audio analysis
SAMPLE_RATE = 44100  # Audio sampling rate
CHUNK_DURATION = 0.2  # Shorter analysis window for low latency
FRAME_SIZE = 1024  # Buffer size
HISTORY_SIZE = 10  # Beat detection stability buffer

# Rolling buffers
audio_buffer = deque(maxlen=int(SAMPLE_RATE * CHUNK_DURATION))
beat_history = deque(maxlen=HISTORY_SIZE)

# Effect categories
color_effects = list(base_color_effects.keys()) + list(special_effects.keys())
fade_effects = ['FADE_1', 'FADE_2', 'FADE_4', 'FADE_5']
strobe_effects = ['SLOW_WHITE', 'SLOW_TURQUOISE', 'SLOW_ORANGE', 'SLOW_YELLOW']


def send_effect(main_effect, tail_code=None):
    if main_effect in base_color_effects:
        effect_bits = base_color_effects[main_effect] + tail_codes.get(tail_code, '')
    elif main_effect in special_effects:
        effect_bits = special_effects[main_effect]
    else:
        return

    arduino_string = bits_to_arduino_string(effect_bits)
    arduino.write(bytes(arduino_string, 'utf-8'))
    arduino.flush()
    print(f"Sent Effect: {main_effect} | Tail: {tail_code}")


def analyze_audio(audio_data):
    y = audio_data.flatten()
    rms_energy = np.sqrt(np.mean(y ** 2))
    silence_threshold = np.median(np.abs(y)) * 2  # Adaptive threshold

    if rms_energy < silence_threshold:
        return None, None, 0, None

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    fft_data = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft_data), 1 / SAMPLE_RATE)
    magnitudes = np.abs(fft_data)

    bass = np.sum(magnitudes[(freqs >= 20) & (freqs <= 250)])
    mid = np.sum(magnitudes[(freqs >= 250) & (freqs <= 2000)])
    treble = np.sum(magnitudes[(freqs >= 2000) & (freqs <= 20000)])

    total_energy = bass + mid + treble
    bass_ratio = bass / total_energy if total_energy > 0 else 0
    mid_ratio = mid / total_energy if total_energy > 0 else 0
    treble_ratio = treble / total_energy if total_energy > 0 else 0

    if bass_ratio > mid_ratio and bass_ratio > treble_ratio:
        effect = random.choice(['RED_3', 'YELLOW_4', 'PULSE_RED'])
    elif mid_ratio > bass_ratio and mid_ratio > treble_ratio:
        effect = random.choice(['GREEN', 'SLOW_GREEN', 'YELLOW_GREEN'])
    elif treble_ratio > bass_ratio and treble_ratio > mid_ratio:
        effect = random.choice(['BLUE', 'LIGHT_BLUE', 'MAGENTA_2'])
    else:
        effect = random.choice(color_effects)

    tail_code = random.choice(fade_effects)

    # if len(beat_frames) > 0 and zero_crossing_rate > 0.05:
    #     effect = random.choice(strobe_effects)

    return effect, tail_code, tempo


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())


def led_control_loop():
    while True:
        if len(audio_buffer) > 0:
            audio_data = np.concatenate(list(audio_buffer))
            # effect, tail_code, tempo = analyze_audio(audio_data)
            result = analyze_audio(audio_data)
            if len(result) == 3:  # If only 3 values are returned, add an empty list
                effect, tail_code, tempo = result
                beat_times = []
            else:
                effect, tail_code, tempo, beat_times = result

            if effect:
                send_effect(effect, tail_code)

            # Convert tempo to a scalar if it's a NumPy array
            tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo
            beat_interval = 60.0 / float(tempo) if tempo > 0 else 0.5

            # beat_interval = 60.0 / tempo if tempo > 0 else 0.5
            time.sleep(beat_interval)
        else:
            time.sleep(0.05)


# Start audio stream and LED control thread
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
stream.start()
threading.Thread(target=led_control_loop, daemon=True).start()

while True:
    time.sleep(1)
