import serial
import time
import numpy as np
import sounddevice as sd
import librosa
import random
from collections import deque
from scipy.signal import find_peaks, spectrogram
from configs.pixmob_conversion_funcs import bits_to_arduino_string
from configs.effect_definitions import base_color_effects, tail_codes, special_effects
import configs.config as cfg

# Setup for Arduino connection
arduino = serial.Serial(port=cfg.ARDUINO_SERIAL_PORT, baudrate=cfg.ARDUINO_BAUD_RATE, timeout=.1)
time.sleep(2.5)

# Parameters for beat tracking
sample_rate = 44100  # Standard audio sampling rate
chunk_size = 0.2  # Reduced for lower latency
frame_size = 1024  # Buffer size
history_size = 10  # Rolling buffer for beat detection

# Effect categories for variety
base_effects = list(base_color_effects.keys())
spec_effects = list(special_effects.keys())
color_effects = base_effects + spec_effects
fade_effects = ['FADE_1', 'FADE_2', 'FADE_4', 'FADE_5']

# Rolling buffers
beat_history = deque(maxlen=history_size)
audio_buffer = deque(maxlen=int(sample_rate * chunk_size))


def send_effect(main_effect, tail_code, sleep_after_send=False):
    if main_effect in base_color_effects:
        effect_bits = base_color_effects[main_effect]
        if tail_code in tail_codes:
            effect_bits += tail_codes[tail_code]
    elif main_effect in special_effects:
        effect_bits = special_effects[main_effect]
    else:
        return

    arduino_string_ver = bits_to_arduino_string(effect_bits)
    arduino.write(bytes(arduino_string_ver, 'utf-8'))
    arduino.flush()

    if sleep_after_send:
        time.sleep(0.01 + 0.0005 * len(effect_bits))

    print(f"Sent Effect: {main_effect}, {'No Tail' if not tail_code else 'Tail: ' + tail_code}")


# Advanced Audio Analysis
def analyze_audio(audio_data):
    y = audio_data.flatten()

    # Compute RMS Energy
    rms_energy = np.sqrt(np.mean(y ** 2))
    silence_threshold = 0.003  # Adjusted for better sensitivity

    if rms_energy < silence_threshold:
        return None, None, 0, None

    # Compute beat detection using peaks
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
    peaks, _ = find_peaks(onset_env, height=np.mean(onset_env))
    beat_detected = len(peaks) > 0

    # Frequency Analysis via Spectrogram
    freqs, times, spec = spectrogram(y, sample_rate)
    bass = np.sum(spec[(freqs >= 20) & (freqs <= 250)])
    mid = np.sum(spec[(freqs >= 250) & (freqs <= 2000)])
    treble = np.sum(spec[(freqs >= 2000) & (freqs <= 20000)])

    total_energy = bass + mid + treble
    bass_ratio = bass / total_energy if total_energy > 0 else 0
    mid_ratio = mid / total_energy if total_energy > 0 else 0
    treble_ratio = treble / total_energy if total_energy > 0 else 0

    # Dynamic Effect Selection
    if bass_ratio > mid_ratio and bass_ratio > treble_ratio:
        effect = random.choice(["RED_3", "YELLOW_4", "RED_2"])
    elif mid_ratio > bass_ratio and mid_ratio > treble_ratio:
        effect = random.choice(["GREEN", "SLOW_GREEN", "YELLOWkGREEN"])
    elif treble_ratio > bass_ratio and treble_ratio > mid_ratio:
        effect = random.choice(["BLUE", "LIGHT_BLUE", "MAGENTA_2"])
    else:
        effect = random.choice(color_effects)

    tail_code = random.choice(fade_effects) if beat_detected else None
    return effect, tail_code, len(peaks), beat_detected


# Audio Callback
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())


# Start Audio Stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=frame_size)
stream.start()

# Main Loop
while True:
    if len(audio_buffer) > 0:
        audio_data = np.concatenate(list(audio_buffer))
        audio_buffer.clear()

        effect, tail_code, beat_count, beat_detected = analyze_audio(audio_data)

        print(f"Beats: {beat_count} | Effect: {effect} | Tail: {tail_code}")

        if effect:
            send_effect(effect, tail_code=tail_code)

        time.sleep(0.3 if beat_detected else 0.05)
    else:
        time.sleep(0.05)
