#include <Servo.h>

Servo SERVO_BASE;
Servo SERVO_SHOULDER;

bool faceDetected = false;

void setup() {
  Serial.begin(115200);
  Serial.println("Arduino ready");
  pinMode(LED_BUILTIN, OUTPUT);
  
  SERVO_SHOULDER.attach(3, 500, 2500);
  SERVO_BASE.attach(5, 600, 2200);

  SERVO_SHOULDER.write(0);
  SERVO_BASE.write(90);
  delay(5000);
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

  digitalWrite(LED_BUILTIN, faceDetected ? HIGH : LOW);

  if (faceDetected) {
    int stabCount = random(1, 4);
    Serial.print("Stab");
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
  int targetPos = random(0, 180);
  int speed     = random(20, 30);
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
  int depth = random(40, 60);

  for (int pos = 0; pos <= depth; pos += 1) {
    SERVO_SHOULDER.write(pos);
    delay(5);
  }

  for (int pos = depth; pos >= 0; pos -= 1) {
    SERVO_SHOULDER.write(pos);
    delay(5);
  }

  delay(500);
  SERVO_SHOULDER.write(0);
}

void waitRandom(int minMs, int maxMs) {
  delay(random(minMs, maxMs));
}
