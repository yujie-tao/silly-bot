import serial
import time
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

print("Available ports:")
for port, desc, hwid in sorted(ports):
    print(f"- {port}: {desc}")

# Set up the serial connection (Change 'COM3' to your actual port)
arduino_port = '/dev/cu.usbmodem11301' 
baud_rate = 9600

print(f"Connecting to {arduino_port}...")
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Wait for the Arduino to reboot after opening the serial port
time.sleep(2) 
print("Connected! Type angles in the terminal.")

def send_angles(angle1, angle2):
    """Formats the angles as 'angle1,angle2\n' and sends them."""
    # Ensure angles are within safe servo limits (0 to 180)
    angle1 = max(0, min(180, int(angle1)))
    angle2 = max(0, min(180, int(angle2)))
    
    # Create the string format: "90,135\n"
    command = f"{angle1},{angle2}\n"
    
    # Send it encoded as bytes
    ser.write(command.encode('utf-8'))
    print(f"Sent: {command.strip()}")

try:
    while True:
        # Get input from the user
        user_input = input("Enter two angles separated by a space (e.g., '90 45') or 'q' to quit: ")
        
        if user_input.lower() == 'q':
            break
            
        try:
            # Split the input into two separate numbers
            val1, val2 = user_input.split()
            send_angles(val1, val2)
            
            # Wait a tiny bit for the Arduino to process and reply
            time.sleep(0.1)
            
            # Read any debug messages the Arduino sends back
            while ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').strip()
                print(f"Arduino says: {response}")
                
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a space.")

except serial.SerialException as e:
    print(f"Serial communication error: {e}")

finally:
    ser.close()
    print("Connection closed.")