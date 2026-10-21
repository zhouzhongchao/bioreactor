/**
 * seq_demo.ino — 六组肌肉逐个动作演示（每组：锁1s→充气5s→放气→开锁回原位）
 *
 * 顺序（作者定版 2026-09-03）：
 *   V0{1,5} → V1{4,8} → V2{2,3} → V3{6,7} → V4{9} → V5{10}
 * 每组时序：
 *   锁上开关阀 → 等 1s → 比例阀 2.0V 充气保持 5s → 指令清零放气 →
 *   放完开锁 → 臂回原位 → 停一拍 → 下一组。不在动作的组锁阀保持打开。
 *
 * 零依赖单文件：GP8403 寄存器级 I2C 内联（与 control_test.ino 同款），不用装库。
 *
 * 通道映射（与 vc2 固件一致）：
 *   保留此前V0{1,5}/V3{6,7}换管映射，再交换V3{6,7}/V5{10}比例阀；自锁不换。
 *   物理通道 p 的比例阀 = GP8403 @0x58+p/2 通道 p%2；逻辑 Vk 的锁阀仍为 D2+k。
 *   BNO055    = Wire2（D9=SDA2、D8=SCL2）——10Hz 同发标准帧，GUI 联动动画用；
 *               没接 IMU 演示照常跑。
 *
 * 串口指令（115200）：g=跑一轮，s=急停（全阀清零、锁全开、序列复位）。
 * GUI 的「跑 Demo / 急停」按钮发的就是 g/s。进度行以 "[demo]" 开头。
 * 安全：比例阀指令钳位 2000 mV（≈180 kPa @0.09 kPa/mV）。
 */

#include <Arduino.h>
#include <Wire.h>

// ── 时序参数（ms）与压力 ────────────────────────────────────────────────
static const int      DEMO_MV       = 1400;   // 演示压力 = 2.0 V，一步直达
static const int      CLAMP_MV      = 1400;   // 绝对钳位 2 V
static const uint32_t LOCK_LEAD_MS  = 1000;   // 锁上 → 充气
static const uint32_t INFLATE_MS    = 3000;   // 充气保持 5 s（作者要求）
static const uint32_t VENT_MS       = 2500;   // 指令清零 → 视为放完
static const uint32_t GAP_MS        = 1500;   // 开锁回原位 → 下一组

// ── GP8403 ×3（寄存器级，协议同 DFRobot 库）────────────────────────────
//   0x01 OUTPUT_RANGE 写 0x11 = 0–10V；0x02/0x04 = 通道0/1 数据（12位左移4）
static const uint8_t DAC_BASE = 0x58;                        // 0x58,0x59,0x5A
static const uint8_t LOCK_PIN[6] = { 2, 3, 4, 5, 6, 7 };     // d0..d5，LOW=锁
static const char   *CH_NAME[6]  = { "1+5", "4+8", "2+3", "6+7", "9", "10" };
static const uint8_t HW_CH[6]    = { 3, 1, 2, 5, 4, 0 };     // 比例阀逻辑→物理；锁阀不交换
static bool dacOK[3] = { false, false, false };

static bool dacBegin(uint8_t addr) {                         // 探测 + 设 10V 量程
  Wire.beginTransmission(addr);
  Wire.write(0x01);
  if (Wire.endTransmission() != 0) return false;
  Wire.beginTransmission(addr);
  Wire.write(0x01); Wire.write(0x11);
  return Wire.endTransmission() == 0;
}
static void valveMv(int ch, int mv) {
  mv = constrain(mv, 0, CLAMP_MV);
  if (ch < 0 || ch > 5) return;
  const uint8_t hw = HW_CH[ch];
  if (!dacOK[hw / 2]) return;
  uint16_t code = (uint16_t)((float)mv / 10000.0f * 4095) << 4;
  Wire.beginTransmission(DAC_BASE + hw / 2);
  Wire.write(hw % 2 ? 0x04 : 0x02);
  Wire.write(code & 0xFF);
  Wire.write((code >> 8) & 0xFF);
  Wire.endTransmission();
}
static void lockSet(int ch, bool lock) {                     // true=充气自锁
  if (ch >= 0 && ch <= 5) digitalWrite(LOCK_PIN[ch], lock ? LOW : HIGH);
}
static void allSafe() {                                      // 急停态：阀清零、锁全开
  for (int c = 0; c < 6; c++) { valveMv(c, 0); lockSet(c, false); }
}

// ── demo 状态机（非阻塞）────────────────────────────────────────────────
enum Phase { IDLE, LOCK_LEAD, INFLATE, VENT, GAP };
static Phase    ph = IDLE;
static int      seqCh = -1;
static uint32_t phT0 = 0;

static void demoSay(const char *what) {
  Serial.print("[demo] ");
  if (seqCh >= 0) { Serial.print('V'); Serial.print(seqCh); Serial.print(" {"); Serial.print(CH_NAME[seqCh]); Serial.print("} "); }
  Serial.println(what);
}
// 注意：参数用 int 不用 Phase —— Arduino IDE 会把自动生成的函数原型提升到
// 文件最顶部（enum 定义之前），签名里出现自定义类型会编译不过。
static void phaseTo(int p) { ph = (Phase)p; phT0 = millis(); }

static void groupBegin(int ch) {                             // 每组入口：先锁
  seqCh = ch; phaseTo(LOCK_LEAD);
  lockSet(seqCh, true);
  demoSay("lock ON, wait 1s");
}
static void demoStart() {
  if (ph != IDLE) { Serial.println("[demo] busy -- 's' first"); return; }
  allSafe();
  groupBegin(0);
}
static void demoAbort() {
  allSafe(); seqCh = -1; ph = IDLE;
  Serial.println("[demo] ABORT: all valves 0, all locks OPEN");
}
static void demoTick() {
  if (ph == IDLE) return;
  const uint32_t dt = millis() - phT0;
  switch (ph) {
    case LOCK_LEAD:                                          // 锁1s → 充气
      if (dt >= LOCK_LEAD_MS) {
        valveMv(seqCh, DEMO_MV);
        phaseTo(INFLATE); demoSay("inflate 2.0V, hold 5s");
      }
      break;
    case INFLATE:                                            // 充5s → 放气
      if (dt >= INFLATE_MS) {
        valveMv(seqCh, 0);
        phaseTo(VENT); demoSay("venting");
      }
      break;
    case VENT:                                               // 放完 → 开锁回原位
      if (dt >= VENT_MS) {
        lockSet(seqCh, false);
        phaseTo(GAP); demoSay("lock OPEN, back to home");
      }
      break;
    case GAP:                                                // 回原位稳定 → 下一组
      if (dt >= GAP_MS) {
        if (seqCh >= 5) { seqCh = -1; ph = IDLE; Serial.println("[demo] sequence done"); }
        else groupBegin(seqCh + 1);
      }
      break;
    default: break;
  }
}

// ── BNO055（与 bno055_test 同一套寄存器级代码，精简为流所需）────────────
#define BNO_WIRE Wire2
static uint8_t g_bno = 0;
static bool bnoW8(uint8_t r, uint8_t v){ BNO_WIRE.beginTransmission(g_bno); BNO_WIRE.write(r); BNO_WIRE.write(v); return BNO_WIRE.endTransmission()==0; }
static bool bnoRd(uint8_t r, uint8_t *b, uint8_t n){
  BNO_WIRE.beginTransmission(g_bno); BNO_WIRE.write(r);
  if (BNO_WIRE.endTransmission(false)!=0) return false;
  if (BNO_WIRE.requestFrom(g_bno, n)!=n) return false;
  for (uint8_t i=0;i<n;i++) b[i]=BNO_WIRE.read();
  return true;
}
static int bnoR8(uint8_t r){ uint8_t v; return bnoRd(r,&v,1)?v:-1; }
static int16_t le16(const uint8_t*p){ return (int16_t)(p[0]|(p[1]<<8)); }

static bool bnoInit() {
  BNO_WIRE.begin(); BNO_WIRE.setClock(100000);
  const uint8_t addrs[2] = { 0x28, 0x29 };
  for (int i = 0; i < 2 && !g_bno; i++) {
    uint32_t t0 = millis();
    while (millis() - t0 < 1000) { g_bno = addrs[i]; if (bnoR8(0x00) == 0xA0) break; g_bno = 0; delay(20); }
  }
  if (!g_bno) return false;
  bnoW8(0x3F, 0x20); delay(700);                             // 软复位，等启动
  uint32_t t0 = millis();
  while (millis() - t0 < 1000 && bnoR8(0x00) != 0xA0) delay(20);
  bnoW8(0x07, 0x00);                                         // page 0
  bnoW8(0x3D, 0x00); delay(25);                              // CONFIG
  bnoW8(0x3E, 0x00); bnoW8(0x3F, 0x00);
  bnoW8(0x3D, 0x0C); delay(25);                              // NDOF
  return true;
}
static void bnoStream() {                                    // 与 bno055_test 严格同格式
  uint8_t eul[6], qua[8], gyr[6], lia[6];
  if (!bnoRd(0x1A, eul, 6) || !bnoRd(0x20, qua, 8) || !bnoRd(0x14, gyr, 6) || !bnoRd(0x28, lia, 6)) return;
  int cal = bnoR8(0x35), temp = bnoR8(0x34);
  Serial.print(le16(eul) / 16.0, 2);     Serial.print(' ');
  Serial.print(le16(eul + 2) / 16.0, 2); Serial.print(' ');
  Serial.print(le16(eul + 4) / 16.0, 2); Serial.print(" | ");
  for (int i = 0; i < 4; i++) { Serial.print(le16(qua + 2 * i) / 16384.0, 4); Serial.print(' '); }
  Serial.print("| ");
  for (int i = 0; i < 3; i++) { Serial.print(le16(gyr + 2 * i) / 16.0, 1); Serial.print(' '); }
  Serial.print("| ");
  for (int i = 0; i < 3; i++) { Serial.print(le16(lia + 2 * i) / 100.0, 2); Serial.print(' '); }
  Serial.print("| "); Serial.print(temp);
  Serial.print(" | S"); Serial.print((cal >> 6) & 3);
  Serial.print(" G");   Serial.print((cal >> 4) & 3);
  Serial.print(" A");   Serial.print((cal >> 2) & 3);
  Serial.print(" M");   Serial.println(cal & 3);
}

// ── 主程序 ──────────────────────────────────────────────────────────────
void setup() {
  // 锁阀先置 OFF（HIGH）再设输出，防止上电瞬间低电平把六路全锁上
  for (int c = 0; c < 6; c++) { digitalWrite(LOCK_PIN[c], HIGH); pinMode(LOCK_PIN[c], OUTPUT); digitalWrite(LOCK_PIN[c], HIGH); }

  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}

  Wire.begin(); Wire.setClock(400000);
  for (int d = 0; d < 3; d++) {
    dacOK[d] = dacBegin(DAC_BASE + d);
    Serial.print("[demo] DAC 0x"); Serial.print(DAC_BASE + d, HEX);
    Serial.println(dacOK[d] ? " ok" : " MISSING");
  }
  allSafe();

  bool imu = bnoInit();
  Serial.println(imu ? "[demo] BNO055 ok, streaming" : "[demo] BNO055 not found (demo still works)");
  Serial.println("[demo] ready -- 'g' = run V0..V5 {1+5,4+8,2+3,6+7,9,10}: lock 1s > inflate 5s > vent > home. 's' = abort");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'g') demoStart();
    else if (c == 's') demoAbort();
  }
  demoTick();
  static uint32_t lastImu = 0;                               // 10 Hz IMU 流（非阻塞）
  if (g_bno && millis() - lastImu >= 100) { lastImu = millis(); bnoStream(); }
}
