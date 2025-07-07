#include <Servo.h>

Servo SERVO_BASE;
Servo SERVO_SHOULDER;
Servo SERVO_HEIGHT;

bool faceDetected = false;

void setup() {
  Serial.begin(115200);
  Serial.println("Arduino ready");
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Attach servos on pins 3,5,9 with valid pulse range
  SERVO_SHOULDER.attach(3, 500, 2500);
  SERVO_BASE.attach(5,    500, 2500);
  SERVO_HEIGHT.attach(9,  500, 2500);

  randomSeed(analogRead(A0));

  // Initial positions
  SERVO_HEIGHT.write(180);
  SERVO_SHOULDER.write(0);
}

void loop() {
  // Read all incoming bytes and update flag
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      faceDetected = true;
      Serial.println("Face detected: ON");
    } else if (cmd == '0') {
      faceDetected = false;
      Serial.println("Face detected: OFF");
    }
  }

  // Debug LED: on when faceDetected==true
  digitalWrite(LED_BUILTIN, faceDetected ? HIGH : LOW);

  if (faceDetected) {
    // Perform random number of stabs
    int stabCount = random(1, 4);
    Serial.print("Stab count: ");
    Serial.println(stabCount);
    for (int i = 0; i < stabCount; i++) {
      Serial.print("Starting stab ");
      Serial.println(i);
      stab();
      Serial.print("Finished stab ");
      Serial.println(i);
      waitRandom(200, 800);
    }
  } else {
    // Search for new target
    searchTarget();
    waitRandom(300, 1000);
  }
}

void searchTarget() {
  int targetPos = random(30, 150);
  int speed     = random(3, 10);
  int current   = SERVO_BASE.read();

  if (targetPos > current) {
    for (int p = current; p <= targetPos; p++) {
      SERVO_BASE.write(p);
      delay(speed);
    }
  } else {
    for (int p = current; p >= targetPos; p--) {
      SERVO_BASE.write(p);
      delay(speed);
    }
  }
}

void stab() {
  // Generate random depth and speed
  int depth = random(40, 70);
  int speed = random(80, 250);
  Serial.print("Stab depth: ");
  Serial.print(depth);
  Serial.print(", speed: ");
  Serial.println(speed);

  // Move shoulder to depth and back to default 30°
  SERVO_SHOULDER.write(depth);
  delay(speed);
  SERVO_SHOULDER.write(0);
}

void waitRandom(int minMs, int maxMs) {
  delay(random(minMs, maxMs));
}
