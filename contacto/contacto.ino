#include <Servo.h>

Servo SERVO_BASE;
Servo SERVO_SHOULDER;
Servo SERVO_GRIP;
Servo SERVO_HEIGHT;

void setup() {
  SERVO_SHOULDER.attach(3, 500, 2500);
  SERVO_BASE.attach(5, 500, 2500); 
  SERVO_HEIGHT.attach(9, 500, 2500);

  randomSeed(analogRead(A0));

  SERVO_HEIGHT.write(180);
  SERVO_SHOULDER.write(30);
}

void loop() {
  searchTarget();
  waitRandom(300, 1000);

  int stabCount = random(1, 4);
  for (int i = 0; i < stabCount; i++) {
    stab();
    waitRandom(200, 800);
  }

  searchTarget();
  waitRandom(500, 1500);
}

void searchTarget() {
  int target = random(30, 150); 
  int speed = random(3, 10);

  int current = SERVO_BASE.read();

  if (target > current) {
    for (int pos = current; pos <= target; pos++) {
      SERVO_BASE.write(pos);
      delay(speed);
    }
  } else {
    for (int pos = current; pos >= target; pos--) {
      SERVO_BASE.write(pos);
      delay(speed);
    }
  }
}

void stab() {
  int depth = random(40, 70);
  int speed = random(80, 250);

  SERVO_SHOULDER.write(depth);
  delay(speed);
  SERVO_SHOULDER.write(0);
}

void waitRandom(int minMs, int maxMs) {
  delay(random(minMs, maxMs));
}