/*
  Arduino Uno + FSR 402

  接线：
  5V -> FSR 402 -> A0 -> 固定电阻 -> GND

  串口波特率：115200
  发送 z：开始标零
*/

// ===== 可调整参数 =====
const byte FSR_PIN = A0;

// 实际使用的固定电阻，单位：Ω
const float RESISTOR_OHMS = 500.0f; // 500 100

const uint16_t DISCARD_SAMPLES = 100; // 标零前丢弃的采样数
const uint16_t ZERO_SAMPLES = 1000;    // 用于标零的采样数
const uint16_t READ_SAMPLES = 50;     // 每次输出的平均采样数

const unsigned int SAMPLE_INTERVAL_MS = 2;
// ======================

float zeroADC = 0.0f;
bool zeroReady = false;

float readAverageADC(uint16_t samples) {
  uint32_t sum = 0;

  for (uint16_t i = 0; i < samples; i++) {
    sum += analogRead(FSR_PIN);
    delay(SAMPLE_INTERVAL_MS);
  }

  return (float)sum / samples;
}

void calibrateZero() {
  zeroReady = false;

  Serial.println(F("# 开始标零，请保持零载荷状态"));
  Serial.println(F("# 丢弃前100次采样..."));

  for (uint16_t i = 0; i < DISCARD_SAMPLES; i++) {
    analogRead(FSR_PIN);
    delay(SAMPLE_INTERVAL_MS);
  }

  Serial.println(F("# 使用接下来500次采样计算零点..."));

  zeroADC = readAverageADC(ZERO_SAMPLES);
  zeroReady = true;

  Serial.println(F("# 标0完成"));
  Serial.print(F("# 零点ADC = "));
  Serial.println(zeroADC, 3);
  Serial.println(F("ADC_processed"));
}

void setup() {
  Serial.begin(115200);
  pinMode(FSR_PIN, INPUT);
  analogReference(DEFAULT);

  delay(100);

  Serial.print(F("# 固定电阻（Ω）= "));
  Serial.println(RESISTOR_OHMS, 1);
  Serial.println(F("# 保持零载荷，发送 z 开始标零"));
}

void loop() {
  while (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'z' || command == 'Z') {
      calibrateZero();
    }
  }

  if (!zeroReady) {
    delay(10);
    return;
  }

  float averagedADC = readAverageADC(READ_SAMPLES);
  float processedADC = averagedADC - zeroADC;

  Serial.println(processedADC, 3);
}