#include <Arduino.h>
#include <Adafruit_TinyUSB.h>

uint32_t getRandomNumber() {
  return TRNG->DATA.reg;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {} // Wait for the serial port to connect
  delay(2000); // Allow some time for the serial monitor to open

  // Initialize TinyUSB for USB communication
  if (!TinyUSB.begin()) {
    Serial.println("Failed to initialize USB");
    while (1) {}
  }

  // Wait for the USB connection to be configured
  while (!TinyUSB.isConfigured()) {}
}

void loop() {
  uint32_t rand_num = getRandomNumber();
  Serial.println(rand_num, BIN);
  delay(1000); // Print a new random number every second
}
