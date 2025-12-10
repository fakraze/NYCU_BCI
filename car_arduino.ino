
#include <SoftwareSerial.h>
SoftwareSerial BT(11, 12); //HC06-TX Pin 10, HC06-RX to Arduino Pin 11
//int LED = 13; //Use whatever pins you want
int speedS = 10;
int speedT = 10;
char t;

void setup() {
  BT.begin(9600); //Baudrate 9600 , Choose your own baudrate (Uncomment if use manual bluetooth)
  Serial.begin(9600);
  //pinMode(LED, OUTPUT);
  pinMode(4, OUTPUT);   //
  pinMode(5, OUTPUT);   //
  pinMode(6, OUTPUT);   //
  pinMode(7, OUTPUT);   //
  pinMode(9, OUTPUT);   //PWM
  pinMode(10, OUTPUT);  //PWM
}

void loop() {
  // Use this one if you set up the bluetooth pin yourself
  
    if (BT.available()) {
      t = BT.read();
      BT.println(t);
      //Serial.println(t);
    }
  
  // Use this one if bluetooth goes to Arduino Tx Rx
  /*
  if (Serial.available()) {
    t = Serial.read();
    //Serial.println(t);
  }
  */

  int pwmT = map(speedT, 0, 10, 0, 255);    // Convert 1-10 to analog value
  int pwmS = map(speedS, 0, 10, 0, 255);    // Convert 1-10 to analog value

  if (t == '1') {          //move forward(all motors rotate in forward direction)
    analogWrite(9, pwmS);
    analogWrite(10, pwmS);
    digitalWrite(4, HIGH);
    digitalWrite(6, HIGH);
    Serial.print("Straight, Speed : ");
    Serial.println(speedS);
  }

  else if (t == '2') {    //move reverse (all motors rotate in reverse direction)
    analogWrite(9, pwmS);
    analogWrite(10, pwmS);
    digitalWrite(5, HIGH);
    digitalWrite(7, HIGH);
    Serial.print("Straight, Speed : ");
    Serial.println(speedS);
  }

  else if (t == '3') {    //turn right (left side motors rotate in forward direction, right side motors doesn't rotate)
    //analogWrite(9, pwmT);
    analogWrite(10, pwmT);
    digitalWrite(6, HIGH);
    Serial.print("Left, Speed : ");
    Serial.println(speedT);
  }

  else if (t == '4') {    //turn left (right side motors rotate in forward direction, left side motors doesn't rotate)
    analogWrite(9, pwmT);
    //analogWrite(10, pwmT);
    digitalWrite(4, HIGH);
    Serial.print("Right, Speed : ");
    Serial.println(speedT);
  }

  else if (t == '5') {
    if (speedS < 10) {
      speedS += 1;
    }
    else if (speedS == 10) {
      speedS = speedS;
    }
    Serial.println((String)"Straight Speed : "+speedS);
  }

  else if (t == '6') {
    if (speedS > 0) {
      speedS -= 1;
    }
    else if (speedS == 0) {
      speedS == speedS;
    }
    Serial.println((String)"Straight Speed : "+speedS);
  }

  else if (t == '7') {
    if (speedT < 10) {
      speedT += 1;
    }
    else if (speedT == 10) {
      speedT = speedT;
    }
    Serial.println((String)"Turn Speed : "+speedT);
  }

  else if (t == '8') {
    if (speedT > 0) {
      speedT -= 1;
    }
    else if (speedT == 0) {
      speedT == speedT;
    }
    Serial.println((String)"Turn Speed : "+speedT);
  }

  else if (t == '0') {    //STOP (all motors stop)
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
    digitalWrite(6, LOW);
    digitalWrite(7, LOW);
    Serial.println("Stop");
  }

  delay(200);
}
