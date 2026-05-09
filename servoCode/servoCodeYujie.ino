#include <Servo.h>

Servo servo1;  // Base/Shoulder
Servo servo2;  // Elbow

void setup() {
  Serial.begin(9600);
  servo1.attach(9);
  servo2.attach(10);
  
  // Set to a default starting position
  servo1.write(90);
  servo2.write(90);
  Serial.println("Arduino Ready.");
}

void loop() {
  // Check if data has arrived
  if (Serial.available() > 0) {
    
    // Read the first integer (stops when it hits the comma)
    int angle1 = Serial.parseInt(); 
    
    // Read the second integer (stops when it hits the newline)
    int angle2 = Serial.parseInt(); 
    
    // Look for the newline character '\n' that marks the end of the command
    if (Serial.read() == '\n') {
      
      // Command the servos to move
      servo1.write(angle1);
      servo2.write(angle2);
      
      // Send a confirmation back to Python
      Serial.print("Moved to: ");
      Serial.print(angle1);
      Serial.print(", ");
      Serial.println(angle2);
    }
  }
}