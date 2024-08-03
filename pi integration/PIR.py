from gpiozero import MotionSensor, LED
import time
import subprocess
import threading
import os
PIR_PIN = 16  # Change to your PIR sensor's GPIO pin
DELAY_TIME = 5  # Delay between readings (seconds)
SETTLE_TIME = 1  # Settling time for the sensor after power-up (seconds)
TIMEOUT = 10  # Time in seconds after which to stop the script if no motion is detected
motion_detected = False
script_running = False  # Flag to track whether the script is running
script_process = None  # Variable to store the process of the running script
last_motion_time = time.time()
def start_script():
    global script_running, script_process
    if not script_running:
        print("Motion detected. Starting script...")
        script_running = True
        script_process = subprocess.Popen(
            ["bash", "-c", "source /home/pi/my_venv/bin/activate && python /home/pi/HIMOQ.py"]
        )
    else:
        print("Script is already running.")

def stop_script():
  global script_running, script_process
  if script_running and script_process:
    print("No motion detected for 10 seconds. Signaling HIMOQ to terminate...")
    # Write "q" to the flag file
    with open("terminate_flag.txt", "w") as f:
      f.write("q")
    script_process.wait()
    script_running = False
    script_process = None
  else:
    print("Script is not running.")

def handle_motion():
    global motion_detected, last_motion_time
    motion_detected = True
    last_motion_time = time.time()
    start_script()

def reset_sensor():
    global motion_detected
    motion_detected = False
    print("Sensor Ready (Waiting for Motion)")

def monitor_motion_timeout():
    global last_motion_time
    while True:
        if script_running and (time.time() - last_motion_time > TIMEOUT):
            stop_script()
        time.sleep(1)

pir = MotionSensor(PIR_PIN)

print(f"Sensor Settling for {SETTLE_TIME} seconds...")
time.sleep(SETTLE_TIME)  # Wait for sensor to settle
print("Ready to detect motion.")

# Start a background thread to monitor motion timeout
threading.Thread(target=monitor_motion_timeout, daemon=True).start()

try:
    while True:
        pir.when_activated = handle_motion
        pir.when_deactivated = reset_sensor

        pir.wait_for_active()

except KeyboardInterrupt:
    print("Exiting...")
    if script_running:
        stop_script()
