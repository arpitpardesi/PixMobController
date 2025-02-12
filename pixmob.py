import serial
import time
import numpy as np
import aubio
from configs.effect_definitions import special_effects as effects # Import PixMob effect definitions

# Serial configuration (Update COM port for Windows)
SERIAL_PORT = "/dev/cu.usbmodemDC5475C4CA882"  # Change to your Arduino port (e.g., "COM3" on Windows)
BAUD_RATE = 115200

# Audio processing settings
BUFFER_SIZE = 512  # Match aubio.tempo input size
SAMPLERATE = 44100  # Standard audio sampling rate

# Initialize beat detection
tempo_detector = aubio.tempo("default", BUFFER_SIZE, BUFFER_SIZE, SAMPLERATE)

# Open serial connection to Arduino
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

print("Listening for beats and analyzing music...")

try:
    while True:
        # Simulate audio input (Replace with real microphone input later)
        audio_data = np.random.rand(BUFFER_SIZE).astype(np.float32)  # Fake audio buffer

        beat_detected = tempo_detector(audio_data[:BUFFER_SIZE])

        if beat_detected:
            # Select a random effect from the list
            effect = np.random.choice(list(effects.keys()))
            ir_signal = effects[effect]

            # Send effect signal to Arduino
            command = f"{effect},{ir_signal}\n"
            ser.write(command.encode())
            print(f"Sent: {command.strip()}")

        time.sleep(0.1)  # Adjust responsiveness

except KeyboardInterrupt:
    print("Stopping...")
    ser.close()
