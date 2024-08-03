from gpiozero import Buzzer
import time

buzzer_pin = 4  # Replace with your actual buzzer pin if different
buzzer = Buzzer(buzzer_pin)

def buzz_for_seconds(seconds):
    buzzer.on()
    time.sleep(seconds)
    buzzer.off()

if __name__ == "__main__":
    try:
        print("Buzzer static cleaning for 0.5 second...")
        buzz_for_seconds(0.5)
        print("Buzzer turned off.")
    finally:
        # Cleanup GPIO pins
        buzzer.close()
