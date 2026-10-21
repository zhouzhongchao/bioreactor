/*
  Arduino Uno + FSR 402 标定程序

  接线：
  5V -> FSR 402 -> A0 -> 固定电阻 -> GND

  串口波特率：115200
  发送 z 或 Z：重新标零

  输出：
  处理后ADC,力(N)

  大传感器，100欧
*/

const byte FSR_PIN = A0;

// 必须填写制作这张标定表时使用的实际电阻
const float RESISTOR_OHMS = 100.0f;

// 标零设置
const uint16_t DISCARD_SAMPLES = 100;
const uint16_t ZERO_SAMPLES = 500;
const unsigned int ZERO_SAMPLE_INTERVAL_MS = 10;

// 正常测量设置
const byte READ_SAMPLES = 31;
const byte TRIM_COUNT = 5;
const unsigned int READ_INTERVAL_MS = 2;

// ADC 标定点
const float calibrationADC[] = {
  144.0, 190.0, 208.0, 218.0,
  230.0, 240.0, 254.0, 260.0,
  273.0, 284.0, 290.0, 296.0,
  298.0, 308.0
};

// 对应的力，单位 N
const float calibrationForce[] = {
  5.0, 10.0, 16.5, 21.0,
  25.5, 30.7, 36.4, 41.2,
  45.7, 50.4, 57.3, 60.6,
  63.2, 70.0
};

const byte CALIBRATION_COUNT =
  sizeof(calibrationADC) / sizeof(calibrationADC[0]);

float zeroADC = 0.0;
bool zeroReady = false;


// 普通平均值，用于标零
float readAverageADC(uint16_t samples,
                     unsigned int intervalMs) {
  uint32_t sum = 0;

  for (uint16_t i = 0; i < samples; i++) {
    sum += analogRead(FSR_PIN);
    delay(intervalMs);
  }

  return (float)sum / samples;
}


// 截尾平均：去除最高和最低的异常数据
float readFilteredADC() {
  uint16_t values[READ_SAMPLES];

  for (byte i = 0; i < READ_SAMPLES; i++) {
    values[i] = analogRead(FSR_PIN);
    delay(READ_INTERVAL_MS);
  }

  // 插入排序
  for (byte i = 1; i < READ_SAMPLES; i++) {
    uint16_t currentValue = values[i];
    int8_t j = i - 1;

    while (j >= 0 && values[j] > currentValue) {
      values[j + 1] = values[j];
      j--;
    }

    values[j + 1] = currentValue;
  }

  uint32_t sum = 0;
  byte validSamples = 0;

  // 去除最低5个和最高5个数据
  for (byte i = TRIM_COUNT;
       i < READ_SAMPLES - TRIM_COUNT;
       i++) {
    sum += values[i];
    validSamples++;
  }

  return (float)sum / validSamples;
}


// 处理后ADC转换为力
float adcToForce(float processedADC) {
  if (processedADC <= 0.0) {
    return 0.0;
  }

  /*
    0～144 ADC 没有足够的实测标定点。
    这里只在 (0,0) 和 (144,5N) 之间作估算。
  */
  if (processedADC < calibrationADC[0]) {
    return processedADC *
           calibrationForce[0] /
           calibrationADC[0];
  }

  // 在相邻标定点之间作线性插值
  for (byte i = 0; i < CALIBRATION_COUNT - 1; i++) {
    if (processedADC <= calibrationADC[i + 1]) {
      float adcLow = calibrationADC[i];
      float adcHigh = calibrationADC[i + 1];

      float forceLow = calibrationForce[i];
      float forceHigh = calibrationForce[i + 1];

      float ratio =
        (processedADC - adcLow) /
        (adcHigh - adcLow);

      return forceLow +
             ratio * (forceHigh - forceLow);
    }
  }

  // 超过已标定量程时固定显示70 N
  return calibrationForce[CALIBRATION_COUNT - 1];
}


void calibrateZero() {
  zeroReady = false;

  Serial.println(F("# 开始标零，请保持零载荷"));
  Serial.println(F("# 丢弃前100次采样"));

  for (uint16_t i = 0; i < DISCARD_SAMPLES; i++) {
    analogRead(FSR_PIN);
    delay(ZERO_SAMPLE_INTERVAL_MS);
  }

  Serial.println(F("# 使用接下来500次采样计算零点"));

  zeroADC = readAverageADC(
    ZERO_SAMPLES,
    ZERO_SAMPLE_INTERVAL_MS
  );

  zeroReady = true;

  Serial.println(F("# 标0完成"));
  Serial.print(F("# 零点ADC="));
  Serial.println(zeroADC, 3);

  Serial.println(F("ADC_processed,Force_N"));
}


void setup() {
  Serial.begin(115200);

  pinMode(FSR_PIN, INPUT);
  analogReference(DEFAULT);

  delay(200);
  analogRead(FSR_PIN);

  Serial.println(F("# FSR 402 标定程序"));
  Serial.print(F("# 固定电阻(ohm)="));
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

  float filteredADC = readFilteredADC();
  float processedADC = filteredADC - zeroADC;
  float forceN = adcToForce(processedADC);

  Serial.print(processedADC, 3);
  Serial.print(',');
  Serial.println(forceN*1, 3);
}