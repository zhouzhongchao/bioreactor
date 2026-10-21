/*
  Arduino Uno + FSR 402 标定测力程序

  接线方式：
  5V -> FSR 402 -> A0 -> 500Ω电阻 -> GND

  串口波特率：G CONSOLE
姿态模式
动力学模式
CAD 实体
剖切
运动学：解耦 s–ψ
关闭
JOINT POSE
滑块设定
Mechanical Zero (Default)
sx Swing x

-6.08
sy Swing y

8.39
ψ Spin Angle Unlocked

-130.90
转动中心 Δx

66
工作零位 q₁ = 20°
ψ 放开

  115200

  启动后：
  1. 自动进行标零
  2. 丢弃前100次采样
  3. 使用后续500次采样计算零点
  4. 显示“标0完成”
  5. 输出处理后ADC和力值

  串口发送 z 或 Z：
  重新标零
*/


// ==================== 硬件参数 ====================

const byte FSR_PIN = A0;

// 制作标定表时使用的固定电阻
const float RESISTOR_OHMS = 500.0f;


// ==================== 标零参数 ====================

// 标零时首先丢弃的数据数量
const uint16_t DISCARD_SAMPLES = 100;

// 用于计算零点平均值的数据数量
const uint16_t ZERO_SAMPLES = 500;

// 标零采样间隔
const unsigned int ZERO_INTERVAL_MS = 10;


// ==================== 测量滤波参数 ====================

// 每次测量读取31个数据
const byte READ_SAMPLES = 31;

// 去除最低5个和最高5个数据
const byte TRIM_COUNT = 5;

// 正常测量时，每次ADC采样之间的间隔
const unsigned int READ_INTERVAL_MS = 2;

// 两次数据显示之间的额外延迟
const unsigned int OUTPUT_INTERVAL_MS = 100;


// ==================== 标定范围 ====================

// 500Ω电阻标定范围
const float MIN_CALIBRATED_ADC = 413.0f;
const float MAX_CALIBRATED_ADC = 624.0f;

const float MIN_CALIBRATED_FORCE = 5.0f;
const float MAX_CALIBRATED_FORCE = 70.6f;


// ==================== 全局变量 ====================

float zeroADC = 0.0f;
bool zeroReady = false;


// ==================================================
// 读取普通平均ADC
// 用于标零
// ==================================================

float readAverageADC(uint16_t numberOfSamples,
                     unsigned int intervalMs) {
  uint32_t sum = 0;

  for (uint16_t i = 0; i < numberOfSamples; i++) {
    sum += analogRead(FSR_PIN);
    delay(intervalMs);
  }

  return (float)sum / numberOfSamples;
}


// ==================================================
// 截尾平均滤波
//
// 读取31个数据，排序后去掉：
// 最低的5个
// 最高的5个
//
// 对剩余21个数据求平均
// ==================================================

float readFilteredADC() {
  uint16_t values[READ_SAMPLES];

  // 读取ADC
  for (byte i = 0; i < READ_SAMPLES; i++) {
    values[i] = analogRead(FSR_PIN);
    delay(READ_INTERVAL_MS);
  }

  // 插入排序，从小到大排列
  for (byte i = 1; i < READ_SAMPLES; i++) {
    uint16_t currentValue = values[i];
    int j = i - 1;

    while (j >= 0 && values[j] > currentValue) {
      values[j + 1] = values[j];
      j--;
    }

    values[j + 1] = currentValue;
  }

  uint32_t sum = 0;
  byte validSamples = 0;

  // 去除最低和最高的异常数据
  for (byte i = TRIM_COUNT;
       i < READ_SAMPLES - TRIM_COUNT;
       i++) {
    sum += values[i];
    validSamples++;
  }

  return (float)sum / validSamples;
}


// ==================================================
// 500Ω标定函数
//
// x = 处理后的ADC
// F = 力，单位N
//
// F = -12.4443 + 4458.561 / (677.6 - x)
//
// 有效范围：
// ADC：413～624
// 力：5～70.6 N
// ==================================================

float calculateForce500Ohm(float processedADC) {
  // 零点或负数
  if (processedADC <= 0.0f) {
    return 0.0f;
  }

  /*
    0～413 ADC之间没有足够的标定数据。

    为了避免ADC达到412时突然从0跳到约5N，
    在(0,0)和(413,5N)之间进行近似线性插值。

    这一范围只能作为估算。
  */
  if (processedADC < MIN_CALIBRATED_ADC) {
    return processedADC
           * MIN_CALIBRATED_FORCE
           / MIN_CALIBRATED_ADC;
  }

  // 超过标定范围后固定为最大标定力
  if (processedADC >= MAX_CALIBRATED_ADC) {
    return MAX_CALIBRATED_FORCE;
  }

  // 500Ω电阻对应的拟合函数
  float forceN =
    -12.4443f
    + 4458.561f / (677.6f - processedADC);

  // 防止计算结果出现负值
  if (forceN < 0.0f) {
    forceN = 0.0f;
  }

  return forceN;
}


// ==================================================
// 标零
// ==================================================

void calibrateZero() {
  zeroReady = false;

  Serial.println();
  Serial.println(F("# 开始标零"));
  Serial.println(F("# 请保持FSR处于零载荷状态"));
  Serial.println(F("# 丢弃前100次采样"));

  // 丢弃前100个数据
  for (uint16_t i = 0; i < DISCARD_SAMPLES; i++) {
    analogRead(FSR_PIN);
    delay(ZERO_INTERVAL_MS);
  }

  Serial.println(F("# 使用后续500次采样计算零点"));

  // 使用后续500个数据计算零点
  zeroADC = readAverageADC(
    ZERO_SAMPLES,
    ZERO_INTERVAL_MS
  );

  zeroReady = true;

  Serial.println(F("# 标0完成"));

  Serial.print(F("# 零点ADC="));
  Serial.println(zeroADC, 3);

  Serial.print(F("# 固定电阻="));
  Serial.print(RESISTOR_OHMS, 1);
  Serial.println(F(" ohm"));

  Serial.println(F("# 输出格式：处理后ADC,力N"));
  Serial.println(F("ADC_processed,Force_N"));
}


// ==================================================
// 初始化
// ==================================================

void setup() {
  Serial.begin(115200);

  pinMode(FSR_PIN, INPUT);
  analogReference(DEFAULT);

  // 等待串口和电源稳定
  delay(2000);

  // 丢弃Arduino启动后的第一次ADC转换
  analogRead(FSR_PIN);

  Serial.println(F("# FSR 402测力程序"));
  Serial.println(F("# 启动后自动标零"));
  Serial.println(F("# 标零期间请勿施加外力"));

  // 启动后自动标零
  calibrateZero();
}


// ==================================================
// 主循环
// ==================================================

void loop() {
  // 检查串口命令
  while (Serial.available() > 0) {
    char command = Serial.read();

    // 发送z或Z重新标零
    if (command == 'z' || command == 'Z') {
      calibrateZero();
    }
  }

  // 标零未完成时不输出测量值
  if (!zeroReady) {
    delay(10);
    return;
  }

  // 读取滤波后的原始ADC
  float filteredADC = readFilteredADC();

  // 扣除零点
  float processedADC = filteredADC - zeroADC;

  // 将非常小的零点波动归零
  if (processedADC > -0.5f &&
      processedADC < 0.5f) {
    processedADC = 0.0f;
  }

  // ADC转换成力
  float forceN =
    calculateForce500Ohm(processedADC);

  // 输出处理后ADC
  Serial.print(processedADC, 3);

  Serial.print(',');

  // 输出力，单位N
  Serial.println(forceN, 3);

  delay(OUTPUT_INTERVAL_MS);
}