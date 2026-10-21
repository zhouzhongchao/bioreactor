/**
 * control_test.ino — V0 比例阀最大电压阶跃测试（无 IMU、无闭环）
 *
 * 只做一件事：测试逻辑 V0{1,5}。实机换管后它由原物理通道3驱动：
 * GP8403@0x59 通道1输出比例阀电压；自锁气路未换，仍由 D2 控制锁阀。
 *   ① DAC 不在总线上（看上电扫描）  ② 阀没供气/接线  ③ 电压没过阀死区
 *
 * 硬件：
 *   GP8403 DAC @0x59 通道1 —— Wire (SDA=D20, SCL=D21)，0–10 V 量程
 *   D2 锁阀（逻辑V0）—— LOW=充气=自锁，上电保持 HIGH（打开）
 *   换算：0–10 V ↔ 0–900 kPa（0.09 kPa/mV，vc2 同款 regulator）
 *
 * 串口指令（115200，每行一条）：
 *   m       最大电压阶跃：V0 先锁 D2 100ms，再立即输出 3000mV，持续到 s。
 *   v<mV>   设电压，例 v800 → 0.8 V（≈72 kPa）。钳位 0–3000 mV。
 *           充气前若锁未上，自动先锁 D2、等 100ms、再给电压。
 *   l / u   锁上 / 打开 D2 锁阀（手动）
 *   s       停：电压清零（锁保持现状 → 零耗气保持；要落下再敲 u）
 *
 * 输出：每 500 ms 一行 [ctl] 遥测：当前 mV 和换算 kPa。
 */

#include <Arduino.h>
#include <Wire.h>

// ── 参数 ────────────────────────────────────────────────────────────────
static const int CLAMP_MV = 3000;           // 与 teach_mode 一致：比例阀最终硬上限 3 V
static const uint32_t LOCK_LEAD_MS = 100;   // 比例阀输出前，锁阀先动作 100 ms

// ── GP8403 DAC（寄存器级，协议同 DFRobot 库）───────────────────────────
//   0x01 OUTPUT_RANGE   写 0x11 = 0–10 V 量程
//   0x02 通道0 数据      12 位码左移 4 位、小端两字节（通道1 是 0x04）
static const uint8_t DAC_ADDR  = 0x59;      // 换管后的A1+5 = 原物理通道3
static const uint8_t DAC_REG   = 0x04;      // GP8403通道1
static const uint8_t LOCK0_PIN = 2;         // 自锁未换：逻辑V0仍用D2
static bool dacOK = false;

static void scanWire() {                    // 上电排查：总线上谁在 ACK
  Serial.print("scan Wire:");
  uint8_t found = 0;
  for (uint8_t a = 0x08; a < 0x78; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { Serial.print(" 0x"); Serial.print(a, HEX); found++; }
  }
  if (!found) Serial.print(" nothing -- check SDA=D20/SCL=D21 wiring & DAC power");
  Serial.println("  (want 0x59 for swapped A1+5)");
}

static bool dacBegin() {                    // 探测 + 设 10 V 量程
  Wire.beginTransmission(DAC_ADDR);
  Wire.write(0x01);
  if (Wire.endTransmission() != 0) return false;   // 无 ACK = 没接
  Wire.beginTransmission(DAC_ADDR);
  Wire.write(0x01); Wire.write(0x11);              // eOutputRange10V
  return Wire.endTransmission() == 0;
}

static int outMv = 0;
static void v0Mv(int mv) {
  outMv = constrain(mv, 0, CLAMP_MV);
  if (!dacOK) { Serial.println("[ctl] DAC MISSING, not sent"); return; }
  uint16_t code = (uint16_t)((float)outMv / 10000.0f * 4095) << 4;  // mV → 12位码，左对齐
  Wire.beginTransmission(DAC_ADDR);
  Wire.write(DAC_REG);                             // 换管后的A1+5：物理通道3
  Wire.write(code & 0xFF);
  Wire.write((code >> 8) & 0xFF);
  uint8_t err = Wire.endTransmission();
  if (err) { Serial.print("[ctl] i2c write error "); Serial.println(err); }
}
static bool locked = false;
static void lock0(bool on) { digitalWrite(LOCK0_PIN, on ? LOW : HIGH); locked = on; }

static bool maxTestRunning = false;
static uint32_t maxTestStartedMs = 0;

static void startV0(int mv, bool maximumTest) {
  mv = constrain(mv, 0, CLAMP_MV);
  if (mv > 0 && !locked) {
    v0Mv(0);                              // 预锁期间比例阀必须保持 0 mV
    lock0(true);
    Serial.print("[ctl] lock d0 ON, wait ");
    Serial.print(LOCK_LEAD_MS);
    Serial.println(" ms before air...");
    delay(LOCK_LEAD_MS);
  }
  v0Mv(mv);
  maxTestRunning = maximumTest && outMv == CLAMP_MV;
  if (maxTestRunning) maxTestStartedMs = millis();
}

// ── 串口指令解析（一行一条）─────────────────────────────────────────────
static char lineBuf[24]; static uint8_t lineLen = 0;
static void handleLine(const char *ln) {
  if (!ln[0]) return;
  char c = ln[0]; float v = atof(ln + 1);
  switch (c) {
    case 'm':
      startV0(CLAMP_MV, true);
      Serial.print("[ctl] MAX STEP started: V0="); Serial.print(outMv);
      Serial.println(" mV; send s to stop");
      break;
    case 'v': {
      int mv = (int)v;
      startV0(mv, false);
      Serial.print("[ctl] set "); Serial.print(outMv); Serial.print(" mV = ");
      Serial.print(outMv * 0.09f, 1); Serial.println(" kPa");
      break;
    }
    case 'l': lock0(true);  Serial.println("[ctl] lock d0 ON (air)"); break;
    case 'u':
      maxTestRunning = false;
      v0Mv(0);
      lock0(false);
      Serial.println("[ctl] V0=0 mV; lock d0 OPEN");
      break;
    case 's':
      maxTestRunning = false;
      v0Mv(0);
      Serial.println("[ctl] STOP: V0=0 mV; lock remains ON");
      break;
    default:  Serial.println("[ctl] cmds: m | v<mV> | l | u | s");   break;
  }
}

void setup() {
  digitalWrite(LOCK0_PIN, HIGH); pinMode(LOCK0_PIN, OUTPUT); digitalWrite(LOCK0_PIN, HIGH);
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}

  Wire.begin();
  Wire.setClock(400000);                    // 与 DFRobot 库一致；此总线只挂 DAC
  scanWire();
  dacOK = dacBegin();
  Serial.print("[ctl] DAC 0x59 ch1 (A1+5) "); Serial.println(dacOK ? "ok" : "MISSING");
  v0Mv(0);
  Serial.println("[ctl] V0 max-step test. cmds: m | v<mV> | l | u | s");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { lineBuf[lineLen] = 0; handleLine(lineBuf); lineLen = 0; }
    else if (lineLen < sizeof(lineBuf) - 1) lineBuf[lineLen++] = c;
  }

  static uint32_t lastTele = 0;
  if (millis() - lastTele >= 500) {
    lastTele = millis();
    Serial.print("[ctl] mv "); Serial.print(outMv);
    Serial.print(" ("); Serial.print(outMv * 0.09f, 1); Serial.print(" kPa)");
    Serial.print(locked ? "  lock ON" : "  lock OPEN");
    if (maxTestRunning) {
      Serial.print("  MAX t=");
      Serial.print((millis() - maxTestStartedMs) / 1000.0f, 1);
      Serial.print("s");
    }
    Serial.println();
  }
}
