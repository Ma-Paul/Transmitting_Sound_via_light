//what I need:
//Arduino Uno R4
#include "analogWave.h"
int x;
analogWave wave(DAC);
int freq = 10;
void Setup() {
	Serial.begin(115200);
	Serial.setTimeout(1);
	pinMode(3, OUTPUT);
        wave.sine(freq);
}

void loop() {
	while (!Serial.available()) {}
	x = Serial.readString().toInt();
	Serial.print(x+1);
}
