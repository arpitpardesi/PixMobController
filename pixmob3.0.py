import serial
import time
import numpy as np
import sounddevice as sd
import aubio
import threading
from configs.pixmob_conversion_funcs import bits_to_arduino_string
from configs.effect_definitions import base_color_effects, tail_codes, special_effects
from configs import config as cfg

# Setup Arduino Connection
arduino = serial.Serial(port=cfg.ARDUINO_SERIAL_PORT, baudrate=cfg.ARDUINO_BAUD_RATE, timeout=.1)
time.sleep(2.5)

# Audio Processing Parameters
SAMPLE_RATE = 44100
FRAME_SIZE = 1024
BUFFER_DURATION = 0.3  # Rolling buffer size in seconds
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
BEAT_THRESHOLD = 0.01  # Adjust based on testing

# Initialize Beat Tracking
beat_detector = aubio.tempo("default", FRAME_SIZE, FRAME_SIZE, SAMPLE_RATE)
rolling_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# Effect Selection
fade_effects = ['FADE_1', 'FADE_2', 'FADE_4', 'FADE_5']
color_effects = list(base_color_effects.keys()) + list(special_effects.keys())


def send_effect(main_effect, tail_code=None):
    """Send the selected effect to Arduino."""
    if main_effect in base_color_effects:
        effect_bits = base_color_effects[main_effect]
        if tail_code and tail_code in tail_codes:
            effect_bits += tail_codes[tail_code]
    elif main_effect in special_effects:
        effect_bits = special_effects[main_effect]
    else:
        return  # Skip invalid effects

    arduino_string = bits_to_arduino_string(effect_bits)
    arduino.write(arduino_string.encode('utf-8'))
    arduino.flush()
    print(f"Sent Effect: {main_effect}, Tail: {tail_code if tail_code else 'None'}")


def analyze_audio():
    """Continuously analyze audio and control wristband effects."""
    global rolling_buffer

    while True:
        if np.max(np.abs(rolling_buffer)) < BEAT_THRESHOLD:
            time.sleep(0.05)
            continue  # Skip processing if silent

        # Compute Beat and Tempo
        is_beat = beat_detector(rolling_buffer)[0]
        tempo = beat_detector.get_bpm()

        # FFT for Frequency Analysis
        fft_data = np.fft.fft(rolling_buffer)
        freqs = np.fft.fftfreq(len(fft_data), 1 / SAMPLE_RATE)
        magnitudes = np.abs(fft_data)

        # Extract frequency bands
        bass = np.sum(magnitudes[(freqs >= 20) & (freqs <= 250)])
        mid = np.sum(magnitudes[(freqs >= 250) & (freqs <= 2000)])
        treble = np.sum(magnitudes[(freqs >= 2000) & (freqs <= 20000)])
        total_energy = bass + mid + treble

        if total_energy > 0:
            bass_ratio, mid_ratio, treble_ratio = bass / total_energy, mid / total_energy, treble / total_energy
        else:
            bass_ratio, mid_ratio, treble_ratio = 0, 0, 0

        # Effect Selection
        if bass_ratio > mid_ratio and bass_ratio > treble_ratio:
            effect = "RED_3"
        elif mid_ratio > bass_ratio and mid_ratio > treble_ratio:
            effect = "GREEN"
        elif treble_ratio > bass_ratio and treble_ratio > mid_ratio:
            effect = "BLUE"
        else:
            effect = np.random.choice(color_effects)

        tail_code = np.random.choice(fade_effects) if is_beat else None
        send_effect(effect, tail_code)

        time.sleep(max(60.0 / tempo, 0.3) if tempo > 0 else 0.5)


def audio_callback(indata, frames, time, status):
    """Update the rolling audio buffer in real-time."""
    global rolling_buffer
    if status:
        print(status)
    rolling_buffer = np.roll(rolling_buffer, -len(indata))
    rolling_buffer[-len(indata):] = indata.flatten()


# Start Audio Stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
stream.start()

# Run Analysis in a Separate Thread
analysis_thread = threading.Thread(target=analyze_audio, daemon=True)
analysis_thread.start()

# Keep the main thread alive
while True:
    time.sleep(1)
