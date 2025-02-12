from flask import Flask, render_template, request, jsonify
import serial
import time
from threading import Thread

app = Flask(__name__)

# Arduino serial connection settings (adjust as needed)
SERIAL_PORT = "/dev/cu.usbmodemDC5475C4CA882"  # Change this based on your OS
BAUD_RATE = 115200
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2.5)  # Allow Arduino to initialize

def send_effect(effect_code):
    """Sends an effect command to the Arduino over serial."""
    arduino.write(effect_code.encode())
    arduino.flush()
    return f"Effect {effect_code} sent!"

@app.route('/')
def home():
    return render_template('index.html')  # Modern Web UI (to be created)

@app.route('/set_effect', methods=['POST'])
def set_effect():
    """Receives effect commands from the UI and sends them to Arduino."""
    data = request.json
    effect_code = data.get('effect')
    if effect_code:
        response = send_effect(effect_code)
        return jsonify({"message": response})
    return jsonify({"error": "Invalid effect"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5090)
