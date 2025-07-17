#include <Servo.h>

Servo SERVO_BASE;
Servo SERVO_SHOULDER;

bool faceDetected = false;
// Flag to track if we've already stabbed for current detection
bool stabDone = false;

// Maximum jitter range in degrees around current position
const int JITTER_RANGE = 60;

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
  // Read all incoming bytes and update faceDetected flag
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
    if (!stabDone) {
      // Do one stab on first detection
      stab();
      stabDone = true;
      waitRandom(200, 800);
    } else {
      // After stabbing, keep jittering so it no quede trabado
      searchTarget();
      waitRandom(200, 800);
    }
  } else {
    // Reset flag when face is lost
    stabDone = false;
    // Continue search in idle
    searchTarget();
    waitRandom(300, 1000);
  }
}

void searchTarget() {
  int current = SERVO_BASE.read();
  int jitter = random(-JITTER_RANGE, JITTER_RANGE + 1);
  int targetPos = current + jitter;
  targetPos = constrain(targetPos, 0, 180);
  int speed = random(20, 30);

  if (targetPos > current) {
    for (int p = current; p <= targetPos; p++) {
      SERVO_BASE.write(p);
      delay(speed);
    }
  } else if (targetPos < current) {
    for (int p = current; p >= targetPos; p--) {
      SERVO_BASE.write(p);
      delay(speed);
    }
  }
  // if equal, no movement
}

void stab() {
  int depth = random(40, 50);

  // Move shoulder forward
  for (int pos = 0; pos <= depth; pos++) {
    SERVO_SHOULDER.write(pos);
    delay(5);
  }
  // Retract shoulder
  for (int pos = depth; pos >= 0; pos--) {
    SERVO_SHOULDER.write(pos);
    delay(5);
  }

  delay(500);
  SERVO_SHOULDER.write(0);
}

void waitRandom(int minMs, int maxMs) {
  delay(random(minMs, maxMs));
}
