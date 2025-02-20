import serial
import time
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import random
from python_tools.pixmob_conversion_funcs import bits_to_arduino_string
from python_tools.effect_definitions import base_color_effects, tail_codes, special_effects
import python_tools.config as cfg
from collections import deque

# Setup for Arduino connection
arduino = serial.Serial(port=cfg.ARDUINO_SERIAL_PORT, baudrate=cfg.ARDUINO_BAUD_RATE, timeout=.1)
# time.sleep(2.5)

# Parameters for beat tracking
sample_rate = 44100  # Standard audio sampling rate
chunk_size = 0.3  # Audio analysis window (seconds) 0.3
frame_size = 1024  # Small buffer for lower latency 1024
history_size = 10  # Store last few beat detections for stability

# Effect categories for variety
base_effects = list(base_color_effects.keys())
spec_effects = list(
    special_effects.keys())  # ['RED_3', 'GREEN', 'BLUE',  'MAGENTA_2',  'YELLOW_4', 'ORANGE', 'WHITISH', 'WHITISH_LONG']
color_effects = base_effects + spec_effects  # ['RED_3', 'GREEN', 'BLUE',  'MAGENTA_2',  'YELLOW_4', 'ORANGE', 'WHITISH', 'WHITISH_LONG']
fade_effects = ['FADE_1', 'FADE_2', 'FADE_4', 'FADE_5']

# Rolling beat detection buffer
beat_history = []
audio_buffer = deque(maxlen=int(sample_rate * chunk_size))  # Using deque to store chunks of audio


# Function to send effects to the wristband
def send_effect(main_effect, tail_code, sleep_after_send=False):
    if main_effect in base_color_effects:
        effect_bits = base_color_effects[main_effect]
        if tail_code in tail_codes:
            effect_bits += tail_codes[tail_code]
    elif main_effect in special_effects:
        effect_bits = special_effects[main_effect]
    else:
        return  # Skip invalid effects

    arduino_string_ver = bits_to_arduino_string(effect_bits)
    arduino.write(bytes(arduino_string_ver, 'utf-8'))
    arduino.flush()

    if sleep_after_send:
        time.sleep(0.01 + 0.0008 * len(effect_bits))

    print(f"Sent Effect: {main_effect}, {'No Tail' if not tail_code else 'Tail: ' + tail_code}")


# Function to analyze audio and detect beat & frequency bands
def analyze_audio(audio_data):
    # Convert to mono and process
    y = audio_data.flatten()

    # Compute audio energy (RMS)
    rms_energy = np.sqrt(np.mean(y ** 2))

    # Define a silence threshold (tune this value based on testing)
    silence_threshold = 0.005  # Adjust based on microphone sensitivity

    # If audio is silent, return None to indicate no effect should be sent
    if rms_energy < silence_threshold:
        return None, None, 0, None

    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, units='time')
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

    # Instead of simple FFT energy ratios, use MFCC-based frequency band detection:
    # mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    # bass = np.mean(mfcc[1:3])  # First few MFCCs represent bass
    # mid = np.mean(mfcc[4:7])  # Midrange energy
    # treble = np.mean(mfcc[8:])  # Treble energy

    total_energy = bass + mid + treble
    bass_ratio = bass / total_energy if total_energy > 0 else 0
    mid_ratio = mid / total_energy if total_energy > 0 else 0
    treble_ratio = treble / total_energy if total_energy > 0 else 0

    # Select effect based on dominant frequency

    # if bass_ratio > mid_ratio and bass_ratio > treble_ratio:
    #     effect = "RED_3" if bass_ratio > 0.6 else random.choice(["YELLOW_4", "RED_2"])

    if bass_ratio > mid_ratio and bass_ratio > treble_ratio:
        effect = random.choice(["RED_3", "YELLOW_4", "RED_2"])
    elif mid_ratio > bass_ratio and mid_ratio > treble_ratio:
        effect = random.choice(["GREEN", "SLOW_GREEN", "YELLOWkGREEN"])
    elif treble_ratio > bass_ratio and treble_ratio > mid_ratio:
        effect = random.choice(["BLUE", "LIGHT_BLUE", "MAGENTA_2"])
    else:
        effect = random.choice(color_effects)

    # If a beat is detected, amplify the effect with flash or pulse
    if len(beat_times) > 0:
        effect = random.choice(color_effects) if random.random() > 0.5 else effect
        tail_code = random.choice(fade_effects)
    else:
        tail_code = random.choice(fade_effects)

    return effect, tail_code, tempo, beat_times


# Audio callback for continuous streaming
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())  # Store incoming audio in the buffer


# Set up the audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=frame_size)
stream.start()

# Main loop to sync lights with beats and audio
# Main loop to sync lights with beats and audio
while True:
    if len(audio_buffer) > 0:
        # Get the most recent chunk of audio data from the buffer
        audio_data = np.concatenate(list(audio_buffer))  # Concatenate the chunks
        audio_buffer.clear()  # Clear the buffer after processing

        # Now, analyze the audio data
        effect_to_send, tail_code, tempo, beat_times = analyze_audio(audio_data)

        print(f"Detected Tempo: {tempo} BPM | Effect: {effect_to_send} | Tail: {tail_code}")

        # if effect_to_send in base_color_effects:
        #     send_effect(effect_to_send, tail_code, sleep_after_send=True)
        # else:
        #     send_effect(effect_to_send, None, sleep_after_send=True)

        send_effect(effect_to_send, tail_code=tail_code)

        # Adjust timing based on detected tempo
        # beat_interval = 60.0 / float(tempo) if tempo > 0 else 0.5

        beat_interval = 60.0 / float(tempo) if tempo and tempo > 0 else 0.5

        # Ensure tempo is a scalar
        # tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        # beat_interval = 60.0 / float(tempo) if tempo > 0 else 0.5

        time.sleep(beat_interval)

    else:
        time.sleep(0.05)  # If buffer is empty, just wait for new data
