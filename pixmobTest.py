import serial
import time
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import random
from configs.pixmob_conversion_funcs import bits_to_arduino_string
from configs.effect_definitions import base_color_effects, tail_codes, special_effects
import configs.config as cfg

# Setup for Arduino connection
arduino = serial.Serial(port=cfg.ARDUINO_SERIAL_PORT, baudrate=cfg.ARDUINO_BAUD_RATE, timeout=.1)
time.sleep(2.5)

# Parameters for beat tracking
sample_rate = 44100  # Standard audio sampling rate
chunk_size = 2  # Audio analysis window (seconds)

# Effect categories for variety
color_effects = ["RED_3", "GREEN", "BLUE", "MAGENTA_2", "YELLOW", "CYAN", "WHITE"]
pulse_effects = ["PULSE_BLUE", "PULSE_RED", "PULSE_GREEN", "PULSE_WHITE"]
flash_effects = ["FLASH_WHITE", "FLASH_BLUE", "FLASH_RED"]
fade_effects = ["FADE_2", "FADE_4", "FADE_6"]


# Function to send effects to the wristband
def send_effect(main_effect, tail_code, sleep_after_send=False):
    if main_effect in base_color_effects:
        effect_bits = base_color_effects[main_effect]
        if tail_code and tail_code in tail_codes:
            effect_bits += tail_codes[tail_code]
    elif main_effect in special_effects:
        effect_bits = special_effects[main_effect]
    else:
        return  # Skip invalid effects

    arduino_string_ver = bits_to_arduino_string(effect_bits)
    arduino.write(bytes(arduino_string_ver, 'utf-8'))

    if sleep_after_send:
        time.sleep(0.01 + 0.0008 * len(effect_bits))

    print(f"Sent Effect: {main_effect}, {'No Tail' if not tail_code else 'Tail: ' + tail_code}")


# Function to analyze audio and detect beat & frequency bands
def analyze_audio():
    # Record short audio sample
    audio_data = sd.rec(int(sample_rate * chunk_size), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()

    # Convert to mono and process
    y = audio_data.flatten()
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

    # Perform FFT for frequency analysis
    fft_data = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft_data), 1 / sample_rate)
    magnitudes = np.abs(fft_data)

    # Identify bass, mid, and treble ranges with normalized scaling
    bass = np.sum(magnitudes[(freqs >= 20) & (freqs <= 250)])
    mid = np.sum(magnitudes[(freqs >= 250) & (freqs <= 2000)])
    treble = np.sum(magnitudes[(freqs >= 2000) & (freqs <= 20000)])

    total_energy = bass + mid + treble
    bass_ratio = bass / total_energy if total_energy > 0 else 0
    mid_ratio = mid / total_energy if total_energy > 0 else 0
    treble_ratio = treble / total_energy if total_energy > 0 else 0

    # Select effect based on dominant frequency
    if bass_ratio > mid_ratio and bass_ratio > treble_ratio:
        effect = random.choice(["RED_3", "PULSE_RED", "FLASH_RED"])
    elif mid_ratio > bass_ratio and mid_ratio > treble_ratio:
        effect = random.choice(["GREEN", "PULSE_GREEN", "CYAN"])
    elif treble_ratio > bass_ratio and treble_ratio > mid_ratio:
        effect = random.choice(["BLUE", "PULSE_BLUE", "MAGENTA_2"])
    else:
        effect = random.choice(color_effects)

    # If a beat is detected, amplify the effect with flash or pulse
    if len(beat_times) > 0:
        effect = random.choice(flash_effects) if random.random() > 0.5 else effect
        tail_code = random.choice(fade_effects)
    else:
        tail_code = random.choice(fade_effects)

    return effect, tail_code, tempo, beat_times


# Main loop to sync lights with beats and audio
while True:
    effect_to_send, tail_code, tempo, beat_times = analyze_audio()

    print(f"Detected Tempo: {tempo} BPM | Effect: {effect_to_send} | Tail: {tail_code}")

    send_effect(effect_to_send, tail_code, sleep_after_send=True)

    # Adjust timing based on detected tempo
    beat_interval = 60.0 / tempo if tempo > 0 else 0.5  # Default to 0.5 sec if tempo is unknown
    time.sleep(beat_interval)