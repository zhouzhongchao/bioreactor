/**
 * teach_mode.ino — 示教-回放（闭环 + 气压跟随的自锁逻辑）
 *
 * 版本：FW_VER（帧 v2）。GUI 连上后若帧里没有第 8/9 段会提示"固件旧版请重烧"。
 *
 * 锁阀规则（作者定版 2026-09-04，"有气就锁、没气就开"）：每路独立——
 *   任何有效驱动都必须先把比例阀保持为 0，同时给对应锁阀充气；锁阀提前 100ms 后，
 *   才允许比例阀驱动肌肉。驱动清零后锁继续保持，只有传感器换算压力为 0 才立即开锁。
 *   监测口在阀全零时读 23–28 kPa（实测偏移），上电采样扣除并把负值截为 0。
 *   30 kPa 只用于解锁判断和拮抗通道交棒；压力遥测保留实际值，不对 MPC/DataModel 截断。
 *   压力监测不可用时禁止开始新的肌肉驱动，也不根据指令或计时猜测已经放空。
 *   唯一例外：全机放气/急停命令会让六路比例阀清零，并立即强制放开六路锁阀。
 *   闲置底气 = 0V（作者 2026-09-04 撤回 300mV 底气：六个 regulator 同时工作太吵）
 * 压力判断用 AD7606 读 regulator 的 1–5V 监测口（板0 通道k = 阀k，vc2 接法，
 * 1–5V ↔ 0–900 kPa）。ADC 不在线时比例阀驱动被禁止，避免无法确认零压却提前开锁。
 *
 * 用法（三步）：
 *   1) 自标定 'c'：逐路 V0..V5 充 1.5V，实测每路把肩膀拉往摆动平面哪个方向。
 *   2) 示教：空闲态锁全开阀全零，人手拖臂，'m' 记路点（≤32，'x' 清空，'z' 对零）。
 *   3) 回放 'g'：逐路点闭环，20 Hz：
 *        e  = (sx*, sy*) − (sx, sy)         摆动角误差（°，swing-twist 分解）
 *        u_k = Kp · max(0, e · d̂_k)         投影为正才充气（肌肉只能拉）
 *      锁由上面的气压规则自动管理。到点 |e|<2° 稳 0.6s → 下一点；超时 10s 跳过。
 *      回放结束全部指令清零 → 放气 → 锁自动打开 → 臂回原位。
 *
 * 串口指令（115200，一行一条）：
 *   c        自标定（臂会逐路动，注意安全）
 *   z / m / x  对零 / 记路点 / 清路点
 *   g / s    回放 / 急停（GUI「跑 Demo/急停」= g/s）
 *   p<Kp>    P 增益 mV/°（默认 80）
 *   u        全机放气：六路比例阀清零，同时六路锁阀立即放开
 *   M<6×锁>;<6×mV>  MPC 原子命令：同时下发六路计划锁动作和实际比例阀电压。
 *            未回报 L 前对应 mV 必须为0；例 M1,0,0,0,0,0;0,0,0,0,0,0。
 *   W<6×mV>  PI 驱动真机：一行设六路电压（物理序 V0..V5，逗号分隔）。
 *            流式看门狗：2s 收不到下一条 W 自动全零并强制开锁。
 *   O<6×mV>  Qt 开环直连专用：协议同 W，但允许到 3000 mV。超时时只清零
 *            比例阀；有压通道保持自锁，压力回零后才自动开锁。
 *   V<ch>,<mV>  手动单路（无看门狗），例 V0,800
 *   r        拉线编码器全部重新置零（上电首帧自动置零）
 *   ?        方向表 + 路点 + ADC 状态
 *
 * 数据流帧格式 v2（GUI 逐段解析，段序不可变，只许在尾部追加）：
 *   eul | quat | gyro | linacc | T | CAL | L1..L20 | P0..P5 | lk
 *   第 7 段 = 20 路肌肉位移（mm；掉线打 x）：ID 1..10 为主动收缩传感器，
 *            ID 11..20 为对应被动拉伸传感器（主动 k 对应被动 k+10）。
 *   第 8 段 = 六路 regulator 实测压 kPa（AD7606 读 1–5V 监测口；无 ADC 打 x）——
 *            控制站直接拿它当气压状态，不再靠时间常数推算
 *   第 9 段 = 六路锁状态 o=开 w=锁合等待 L=锁定
 *
 * 硬件（与 vc2 / seq_demo 一致）：
 *   逻辑 V0..V5 = {1,5},{4,8},{2,3},{6,7},{9},{10}
 *   2026-09-25 实机换管后：V0{1,5}/V3{6,7} 只交换比例阀气路及其压力反馈；自锁气路未换。
 *   2026-09-27 在上述基础上再交换 V3{6,7}/V5{10}：67用原10的好阀，10用原67的慢阀；自锁不换。
 *   物理比例阀通道 p：GP8403 @0x58+p/2 的 p%2，压力=AD7606 p；逻辑 Vk 的锁阀仍为 D2+k。
 *   压力监测  = AD7606 @SPI1（SCK=D13,CIPO=D12；CONVST=D39,RST=D40,CS=D41,BUSY=D45）
 *   BNO055    = Wire2（D9/D8），10 Hz 标准帧同发，GUI 联动照常
 * 安全：MPC/PI/手动的日常限幅由PC端gains.json决定；固件对W/V/O统一保留3000mV最终保护；
 *       2s 看门狗（W/O 流断则放气）；心跳门控防 CDC 僵死。
 */

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>

// ── 参数 ────────────────────────────────────────────────────────────────
static const char     FW_VER[]     = "2026-09-27b";   // N原子命令仅允许V2/V3共充；保持100ms预锁
static const int      CLAMP_MV     = 3000;   // W/V最终硬保护；日常上限由PC端gains.json决定
static const int      OPEN_LOOP_CLAMP_MV = 3000; // 仅 O 开环专用流可到 3.0 V
static const int      CAL_MV       = 1500;   // 自标定充气电压
static const int      MIN_ON_MV    = 0;      // 不设门槛电压：0=无压，任何正电压都按控制器请求执行
static const float    WP_TOL_DEG   = 2.0f;   // 到点判据
static const uint32_t WP_SETTLE_MS = 600;
static const uint32_t WP_TIMEOUT_MS= 10000;
static const uint32_t CTL_MS       = 50;     // 控制节拍 20 Hz
static const uint32_t LOCK_LEAD_MS = 100;    // 电磁开关阀先动作100ms，之后允许比例阀驱动肌肉
static const float    KPA_ZERO_NOISE = 30.0f; // 解锁/拮抗交棒判据：校正后压力≤30kPa认为已基本放空
static const int      WP_MAX       = 32;

// ── GP8403 ×3（寄存器级）───────────────────────────────────────────────
static const uint8_t DAC_BASE = 0x58;
static const uint8_t LOCK_PIN[6] = { 2, 3, 4, 5, 6, 7 };
static const char   *CH_NAME[6]  = { "1+5", "4+8", "2+3", "6+7", "9", "10" };
// 逻辑通道→物理比例阀及阀内压力反馈：保留15的既有映射，再交换67与10；自锁不变。
static const uint8_t HW_CH[6] = { 3, 1, 2, 5, 4, 0 };
static bool dacOK[3] = { false, false, false };
static int  outMv[6] = { 0 };

static bool dacBegin(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x01);
  if (Wire.endTransmission() != 0) return false;
  Wire.beginTransmission(addr);
  Wire.write(0x01); Wire.write(0x11);            // 0–10 V 量程
  return Wire.endTransmission() == 0;
}
static void valveMv(int ch, int mv) {
  mv = constrain(mv, 0, OPEN_LOOP_CLAMP_MV);
  if (ch < 0 || ch > 5) return;
  const uint8_t hw = HW_CH[ch];
  if (!dacOK[hw / 2]) return;
  outMv[ch] = mv;
  uint16_t code = (uint16_t)((float)mv / 10000.0f * 4095) << 4;
  Wire.beginTransmission(DAC_BASE + hw / 2);
  Wire.write(hw % 2 ? 0x04 : 0x02);
  Wire.write(code & 0xFF);
  Wire.write((code >> 8) & 0xFF);
  Wire.endTransmission();
}
static void lockPinSet(int ch, bool lock) {      // 只动引脚；状态机在通道层
  if (ch >= 0 && ch <= 5) digitalWrite(LOCK_PIN[ch], lock ? LOW : HIGH);
}

// ── AD7606 压力监测（从 vc2 adc_ad7606.cpp 移植，单板 8 通道）──────────
static const uint8_t ADC_CONVST = 39, ADC_RST = 40, ADC_CS = 41, ADC_BUSY = 45;
static const float   ADC_RANGE_V = 10.0f;        // RANGE→3.3V 档：±10 V
static const float   ADC_V_GAIN = 0.9849f, ADC_V_OFF = 0.0054f;  // vc2 实测校准
// ADC raw→kPa 的运行时直线。默认值等价于旧公式
// (raw*10/32768*0.9849+0.0054-1)*225；Qt 可用 A<slope>,<intercept> 在线更新。
static float adcKpaSlope = (ADC_RANGE_V / 32768.0f) * ADC_V_GAIN * 225.0f;
static float adcKpaIntercept = (ADC_V_OFF - 1.0f) * 225.0f;
static bool    adcOn = false;
static int16_t adcRaw[8] = { 0 };
static uint32_t adcLastUs = 0;

static void adcAcquire() {
  digitalWrite(ADC_CONVST, HIGH); delayMicroseconds(1); digitalWrite(ADC_CONVST, LOW);
  uint32_t t0 = micros();
  while (digitalRead(ADC_BUSY)) { if ((uint32_t)(micros() - t0) > 300) return; }
  SPI1.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE2));
  digitalWrite(ADC_CS, LOW);
  for (int ch = 0; ch < 8; ch++) {
    uint8_t hi = SPI1.transfer(0x00), lo = SPI1.transfer(0x00);
    adcRaw[ch] = (int16_t)(((uint16_t)hi << 8) | lo);
  }
  digitalWrite(ADC_CS, HIGH);
  SPI1.endTransaction();
}
static void adcInit() {
  pinMode(ADC_CONVST, OUTPUT); digitalWrite(ADC_CONVST, LOW);
  pinMode(ADC_RST, OUTPUT);    digitalWrite(ADC_RST, LOW);
  pinMode(ADC_CS, OUTPUT);     digitalWrite(ADC_CS, HIGH);
  pinMode(ADC_BUSY, INPUT_PULLDOWN);             // 缺板时 BUSY 恒低，不会假超时
  SPI1.begin();                                  // 丝印 SPI 排针 = SPI1（不是 SPI！）
  digitalWrite(ADC_RST, HIGH); delayMicroseconds(5); digitalWrite(ADC_RST, LOW); delayMicroseconds(50);
  // 在位探测：CONVST 后在位的板会把 BUSY 拉高 ~4us
  digitalWrite(ADC_CONVST, HIGH); delayMicroseconds(1); digitalWrite(ADC_CONVST, LOW);
  uint32_t t0 = micros();
  while ((uint32_t)(micros() - t0) < 60) { if (digitalRead(ADC_BUSY)) { adcOn = true; break; } }
  Serial.print("[tch] ADC ");
  Serial.println(adcOn ? "present (measured zero-pressure release)" : "ABSENT (muscle drive inhibited)");
}
static void adcTick() {                          // 200 Hz 采样
  if (!adcOn) return;
  uint32_t now = micros();
  if ((uint32_t)(now - adcLastUs) < 5000) return;
  adcLastUs = now;
  adcAcquire();
}
static float kpaZero[6] = { 0 };                 // 上电零点（阀全零时监测口的读数，实测 23–28 kPa 偏移）
static float adcZeroRaw[6] = { 0 };              // 各路原始计数零点；分段表统一使用 raw-本路零点
static const uint8_t ADC_TABLE_MAX = 10;
static uint8_t adcTableN = 0;
static float adcTableDx[ADC_TABLE_MAX] = { 0 };
static float adcTableKpa[ADC_TABLE_MAX] = { 0 };
static bool  kpaZeroReady = false;               // 完成上电零压标定前禁止用压力值驱动/解锁
static float chKpaRaw(int ch) {
  return adcRaw[HW_CH[ch]] * adcKpaSlope + adcKpaIntercept;
}
static float adcTableEval(float dx) {
  if (dx <= 0.0f || adcTableN < 2) return 0.0f;
  for (uint8_t i = 1; i < adcTableN; i++) {
    if (dx <= adcTableDx[i]) {
      float den = adcTableDx[i] - adcTableDx[i-1];
      float a = den > 1e-3f ? (dx-adcTableDx[i-1])/den : 0.0f;
      return adcTableKpa[i-1] + a*(adcTableKpa[i]-adcTableKpa[i-1]);
    }
  }
  // 超出最高标定点时只按最后一段外推；控制器仍会在传感器量程处钳位。
  uint8_t i = adcTableN-1;
  float den = adcTableDx[i] - adcTableDx[i-1];
  float slope = den > 1e-3f ? (adcTableKpa[i]-adcTableKpa[i-1])/den : 0.0f;
  return adcTableKpa[i] + slope*(dx-adcTableDx[i]);
}
static float chKpa(int ch) {                     // 阀 ch 实测压力（kPa，已扣零点）；无 ADC 返回 -1
  if (!adcOn || ch < 0 || ch > 5) return -1.0f;
  if (adcTableN >= 2) return max(0.0f, adcTableEval((float)adcRaw[HW_CH[ch]]-adcZeroRaw[ch]));
  float k = chKpaRaw(ch) - kpaZero[ch];
  return k > 0.0f ? k : 0.0f;                    // 遥测/DataModel 保留实际压力，30kPa不在这里截断
}
static void adcZero() {                          // 上电（阀全零、气已放）采 16 次取均值作零点
  if (!adcOn) return;
  float acc[6] = { 0 }, rawAcc[6] = { 0 };
  for (int n = 0; n < 16; n++) {
    adcAcquire();
    for (int c = 0; c < 6; c++) { acc[c] += chKpaRaw(c); rawAcc[c] += adcRaw[HW_CH[c]]; }
    delay(20);
  }
  for (int c = 0; c < 6; c++) {
    kpaZero[c] = acc[c] / 16.0f;
    adcZeroRaw[c] = rawAcc[c] / 16.0f;
  }
  kpaZeroReady = true;
}

// ── 通道层：锁跟气压走 ─────────────────────────────────────────────────
// 状态：0=OPEN（没气锁开）1=LOCKING（锁上等100ms，阀先憋着）2=LOCKED（有气/在充）
static int      chReq[6]  = { 0 };               // 控制器要的 mV
static int      lockSt[6] = { 0 };
static uint32_t lkT0[6]   = { 0 };               // LOCKING 计时
static bool     venting[6]= { false };
static bool     forceOpenAll = false;             // 全机放气：保持六路锁阀强制打开，直到收到新驱动
static bool     explicitLockMode = false;          // M命令：锁动作由MPC显式规划，不再由非零电压暗中触发
static bool     lockReq[6] = { false };            // MPC本拍的六路计划锁动作
static bool     coContractStream = false;          // C命令：仅Debug手动压力允许对抗通道同时加压
static const uint8_t OPP_CH[6] = {1, 0, 3, 2, 5, 4}; // 1/5↔4/8，2/3↔6/7，9↔10

static void chRequest(int ch, int mv, int limitMv = CLAMP_MV) {
  if (ch < 0 || ch > 5) return;
  chReq[ch] = (mv < MIN_ON_MV) ? 0 : constrain(mv, 0, limitMv);
  if (chReq[ch] > 0) forceOpenAll = false;        // 新驱动退出全开放气态，再走预锁100ms
}
static bool     extStream = false;               // GUI 正在流式驱动（W/O/C 指令）
static bool     extOpenLoop = false;             // 最近流命令是 O/C；超时不能带压强制开锁
static uint32_t extLastMs = 0;
// 压力没有人为门槛；只确认 ADC 已在线并完成零点标定。
static bool pmonOk(int ch) { return adcOn && kpaZeroReady && ch >= 0 && ch <= 5; }
static bool pressureEmpty(int ch) {
  return pmonOk(ch) && chKpa(ch) <= KPA_ZERO_NOISE;
}
static bool phiPairStream = false;
static bool antagonistClear(int ch) {             // 对侧请求、输出、锁和实测压力必须全部释放
  if (coContractStream) return true;
  if (phiPairStream && (ch == 2 || ch == 3)) return true;
  int op = OPP_CH[ch];
  return chReq[op] == 0 && outMv[op] == 0 && lockSt[op] == 0 && pressureEmpty(op)
         && !(explicitLockMode && lockReq[op]);
}
static void requestVector(const int wanted[6], int limitMv = CLAMP_MV) { // 指令层硬互斥
  int d[6];
  for (int c = 0; c < 6; c++) d[c] = constrain(wanted[c], 0, limitMv);
  for (int left = 0; left < 6; left += 2) {
    int right = left + 1;
    if (phiPairStream && left == 2) continue;
    if (d[left] > 0 && d[right] > 0) {
      bool la = lockSt[left] != 0 || outMv[left] > 0 || !pressureEmpty(left);
      bool ra = lockSt[right] != 0 || outMv[right] > 0 || !pressureEmpty(right);
      if (la && !ra) d[right] = 0;
      else if (ra && !la) d[left] = 0;
      else if (d[left] >= d[right]) d[right] = 0;
      else d[left] = 0;
    }
  }
  for (int c = 0; c < 6; c++) chRequest(c, d[c], limitMv);
}

static void chTick() {                           // 每控制拍跑一遍，逐路独立
  if (forceOpenAll) {
    for (int c = 0; c < 6; c++) {
      chReq[c] = 0;
      if (outMv[c] != 0) valveMv(c, 0);
      lockPinSet(c, false);
      lockSt[c] = 0; venting[c] = false; lkT0[c] = 0;
    }
    return;
  }
  for (int c = 0; c < 6; c++) {
    const int req  = chReq[c];
    const bool plannedLock = explicitLockMode && lockReq[c];
    const bool autoLock = !explicitLockMode && req > 0;
    const bool wantLock = plannedLock || autoLock;
    const int appliedReq = (explicitLockMode && !plannedLock) ? 0 : req;
    // 锁规则：任何有效驱动前，比例阀必须保持 0，锁阀先动作100ms；驱动清零后，
    // 比例阀立即回 0，但锁保持到传感器确认压力≤30kPa，再打开本路锁阀。
    switch (lockSt[c]) {
      case 0:                                    // OPEN：绝不向肌肉供气
        if (outMv[c] != 0) valveMv(c, 0);
        venting[c] = false;
        if (pmonOk(c) && (chKpa(c) > KPA_ZERO_NOISE || (wantLock && antagonistClear(c)))) {
          lockPinSet(c, true);                    // 对应锁阀先充气
          lkT0[c] = millis(); lockSt[c] = 1;
        }
        break;
      case 1:                                    // LOCKING：比例阀保持 0，锁阀预充100ms
        if (outMv[c] != 0) valveMv(c, 0);
        if (!wantLock && req == 0 && pressureEmpty(c)) {
          lockPinSet(c, false); lockSt[c] = 0;
        } else if (wantLock && pmonOk(c) && antagonistClear(c) && millis() - lkT0[c] >= LOCK_LEAD_MS) {
          lockSt[c] = 2; venting[c] = false;
          valveMv(c, appliedReq);
        }
        break;
      case 2:                                    // LOCKED：有驱动就跟指令；无驱动就放气但绝不提前开锁
        if (outMv[c] != appliedReq) valveMv(c, appliedReq);
        if (explicitLockMode && plannedLock) {
          // 显式预锁动作允许在零压、零比例阀时保持锁合；下一拍收到MPC电压即可忠实执行。
          venting[c] = false;
        } else if (appliedReq == 0) {
          venting[c] = true;
          if (pressureEmpty(c)) {
            lockPinSet(c, false); lockSt[c] = 0; venting[c] = false;
          }
        } else { venting[c] = false; }
        break;
    }
  }
}
static void allSafe() {                          // 全机放气/急停：比例阀清零，六路锁阀立即强制打开
  extStream = false;
  extOpenLoop = false;
  explicitLockMode = false;
  coContractStream = false;
  phiPairStream = false;
  forceOpenAll = true;
  for (int c = 0; c < 6; c++) {
    lockReq[c] = false;
    chReq[c] = 0; valveMv(c, 0);
    lockPinSet(c, false);
    lockSt[c] = 0; venting[c] = false; lkT0[c] = 0;
  }
}

// ── 拉线编码器 ×20（RS-485 Modbus 非阻塞轮询；协议经 encoder_test 实机验证）──
// ID 1..10 = 主动收缩；ID 11..20 = 对应被动拉伸（主动 k ↔ 被动 k+10）。
// Serial2（丝印 TX1/RX1 = D18/D19），DE=D10。
// ⚠ mbed 的 flush() 不等移位寄存器清空——DE 必须按线上字节时间足额持有再放
//   （encoder_test 踩过；clone encoder_rs485.cpp 头注释同）。
// 帧流尾部追加 20 路位移（mm，不在线打 x），GUI 按主/被动配对显示。
#define ENC_UART Serial2
static const uint8_t  ENC_COUNT    = 20;
static const uint8_t  ENC_DE_PIN   = 10;
static const uint16_t ENC_REG      = 0x0380;
static const uint32_t ENC_TXGUARD  = 150;    // < 3.5 字符静默(304µs)
static const uint32_t ENC_GAP_US   = 500;
static const uint32_t ENC_RXTO_MS  = 50;
static const float    ENC_SPOOL_MM = 13.7f;
static const int8_t   ENC_DIR      = +1;
static float    encMm[ENC_COUNT + 1] = { 0 }; // 下标直接等于 Modbus ID（1..20）
static bool     encOn[ENC_COUNT + 1] = { false }, encZeroSet[ENC_COUNT + 1] = { false };
static double   encZero[ENC_COUNT + 1];
static uint8_t  encMiss[ENC_COUNT + 1] = { 0 };
static uint8_t  encSt = 0, encCur = 0;
static uint32_t encT0us = 0, encT0ms = 0;
static uint8_t  encRxB[24]; static uint8_t encRxLen = 0;
static uint16_t encSweep = 0;

static uint16_t encCrc(const uint8_t *d, int n){
  uint16_t c = 0xFFFF;
  for (int i = 0; i < n; i++) { c ^= d[i]; for (int b = 0; b < 8; b++) c = (c & 1) ? (c >> 1) ^ 0xA001 : (c >> 1); }
  return c;
}
static void encNext(){                       // 掉线（连丢3次）的 ID 每 16 轮才再探一次
  for (int hop = 0; hop < ENC_COUNT; hop++) {
    encCur = (encCur % ENC_COUNT) + 1;
    if (encCur == 1) encSweep++;
    if (encMiss[encCur] < 3 || (encSweep & 15) == 0) return;
  }
}
static void encTick(){
  switch (encSt) {
    case 0: {                                // 帧间静默 → 发下一条请求
      if ((uint32_t)(micros() - encT0us) < ENC_GAP_US) return;
      encNext();
      uint8_t rq[8] = { encCur, 0x03, (uint8_t)(ENC_REG >> 8), (uint8_t)(ENC_REG & 0xFF), 0, 4, 0, 0 };
      uint16_t crc = encCrc(rq, 6); rq[6] = crc & 0xFF; rq[7] = crc >> 8;
      while (ENC_UART.available()) ENC_UART.read();
      digitalWrite(ENC_DE_PIN, HIGH); delayMicroseconds(15);
      encT0us = micros();
      ENC_UART.write(rq, 8);
      // 复检修正：DE 释放不能等下一圈 loop——同圈的 I2C（IMU/DAC/流）会把 DE
      // 多拖 3~5ms，盖掉 304µs 后就应答的从站。发送共 ~845µs，原地等完立刻放。
      while ((uint32_t)(micros() - encT0us) < 8UL * 10UL * 1000000UL / 115200UL + ENC_TXGUARD) {}
      digitalWrite(ENC_DE_PIN, LOW);
      encRxLen = 0; encT0ms = millis(); encSt = 2;
      break;
    }
    case 2: {                                // 收应答：13 字节 = id+03+08+data8+crc2
      while (ENC_UART.available() && encRxLen < sizeof(encRxB)) encRxB[encRxLen++] = ENC_UART.read();
      bool done = false, ok = false;
      if (encRxLen >= 13) {
        uint16_t rc = encRxB[11] | ((uint16_t)encRxB[12] << 8);
        ok = (encRxB[0] == encCur && encRxB[1] == 0x03 && encRxB[2] == 0x08 && rc == encCrc(encRxB, 11));
        done = true;
      }
      if (!done && millis() - encT0ms > ENC_RXTO_MS) done = true;
      if (!done) return;
      if (ok) {
        uint32_t pos = ((uint32_t)encRxB[3] << 24) | ((uint32_t)encRxB[4] << 16) | ((uint32_t)encRxB[5] << 8) | encRxB[6];
        int32_t  tr  = (int32_t)(((uint32_t)encRxB[7] << 24) | ((uint32_t)encRxB[8] << 16) | ((uint32_t)encRxB[9] << 8) | encRxB[10]);
        double deg = ENC_DIR * ((double)tr * 360.0 + (double)(pos % (1UL << 21)) * 360.0 / (double)(1UL << 21));
        if (!encZeroSet[encCur]) { encZero[encCur] = deg; encZeroSet[encCur] = true; }   // 首帧自动置零
        encMm[encCur] = (float)((deg - encZero[encCur]) * (3.14159265358979 * ENC_SPOOL_MM) / 360.0);
        encOn[encCur] = true; encMiss[encCur] = 0;
      } else {
        if (encMiss[encCur] < 250) encMiss[encCur]++;
        if (encMiss[encCur] >= 3) encOn[encCur] = false;
      }
      encT0us = micros(); encSt = 0;
      break;
    }
  }
}

// ── BNO055（与 bno055_test 同源）───────────────────────────────────────
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
  bnoW8(0x3F, 0x20); delay(700);
  uint32_t t0 = millis();
  while (millis() - t0 < 1000 && bnoR8(0x00) != 0xA0) delay(20);
  bnoW8(0x07, 0x00); bnoW8(0x3D, 0x00); delay(25);
  bnoW8(0x3E, 0x00); bnoW8(0x3F, 0x00);
  bnoW8(0x3D, 0x0C); delay(25);
  return true;
}

// ── 姿态：四元数 → 摆动角 (sx, sy)（swing-twist，扭转绕 z 剔除）─────────
static float qNow[4] = {1,0,0,0}, qRef[4] = {1,0,0,0};

static bool bnoQuat() {
  uint8_t b[8]; float q[4];
  if (!bnoRd(0x20, b, 8)) return false;
  for (int i = 0; i < 4; i++) q[i] = le16(b + 2*i) / 16384.0f;
  float n = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]); if (n < 1e-6f) return false;
  for (int i = 0; i < 4; i++) qNow[i] = q[i] / n;
  return true;
}
static void qMul(const float *a, const float *b, float *r) {
  r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
  r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
  r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}
// 与 GUI 的 imuSwingTwist 同一分解；肌肉方向表是实测的，坐标系差一个
// 固定旋转也不影响控制（误差和方向在同一坐标系里）。
static void getSwing(float *sx, float *sy) {
  float cRef[4] = { qRef[0], -qRef[1], -qRef[2], -qRef[3] }, q[4];
  qMul(cRef, qNow, q);
  float n = sqrtf(q[0]*q[0] + q[3]*q[3]);
  float sw[4];
  if (n > 1e-6f) {
    float ctw[4] = { q[0]/n, 0, 0, -q[3]/n };
    qMul(q, ctw, sw);
  } else { for (int i=0;i<4;i++) sw[i]=q[i]; }
  float w = sw[0]; if (w > 1) w = 1; if (w < -1) w = -1;
  float th = 2.0f * acosf(w), sh = sqrtf(fmaxf(0.0f, 1.0f - w*w));
  *sx = 0; *sy = 0;
  if (sh > 1e-9f) { *sx = th * sw[1] / sh * 57.29578f; *sy = th * sw[2] / sh * 57.29578f; }
}

// ── 肌肉方向表 / 路点 ───────────────────────────────────────────────────
static float dirX[6] = {0}, dirY[6] = {0}, gainDeg[6] = {0};
static bool  calValid[6] = {false};
static bool  anyCal = false;
static float wpSx[WP_MAX], wpSy[WP_MAX];
static int   wpN = 0;

// ── 模式机（int 常量而非 enum：躲 Arduino IDE 的原型提升坑）─────────────
static const int MODE_IDLE = 0, MODE_CAL = 1, MODE_REPLAY = 2;
static int      mode = MODE_IDLE;
static int      phase = 0;
static uint32_t phT0 = 0;

static int   calCh = -1;
static float calAx = 0, calAy = 0; static int calN = 0;

static int      rpWp = -1;
static uint32_t rpInTolT0 = 0;
static bool     rpInTol = false;
static float    Kp = 80.0f;

static float curSx = 0, curSy = 0;

// 宿主在线标志（loop 每圈按心跳刷新）。⚠ 所有【自主状态机】里的打印都必须
// 经它门控——复检发现 CAL/REPLAY 的进度打印没有门控：宿主中途断开时，这些
// 打印会写死 CDC 发送端，臂在【带压+锁合】状态下冻住，只能物理复位。
static bool g_host = false;
static uint32_t lastRxMs = 0;                    // 最近一次收到宿主字节的时刻（心跳）
static void sayMode(const char *msg) { if (!g_host) return; Serial.print("[tch] "); Serial.println(msg); }
static void phaseTo(int p) { phase = p; phT0 = millis(); }

// ── 标定：逐路 请求1.5V（通道层自动预锁100ms）→稳3s→测0.5s→清零→等放完→下一路 ──
static void calBeginCh(int ch) {
  calCh = ch; calAx = calAy = 0; calN = 0;
  chRequest(calCh, CAL_MV); phaseTo(0);
  if (g_host) { Serial.print("[tch] cal V"); Serial.print(calCh); Serial.print(" {"); Serial.print(CH_NAME[calCh]); Serial.println("}"); }
}
static void calStart() {
  if (mode != MODE_IDLE) { sayMode("busy -- 's' first"); return; }
  if (!g_bno) { sayMode("no IMU, cannot calibrate"); return; }
  allSafe(); mode = MODE_CAL;
  sayMode("calibrating 6 directions, arm will move...");
  calBeginCh(0);
}
static void calFinishCh() {
  float mag = sqrtf(calAx*calAx + calAy*calAy);
  calValid[calCh] = (mag > 1.5f);
  if (calValid[calCh]) { dirX[calCh] = calAx/mag; dirY[calCh] = calAy/mag; gainDeg[calCh] = mag; anyCal = true; }
  if (g_host) {
    Serial.print("[tch] cal V"); Serial.print(calCh);
    if (calValid[calCh]) {
      Serial.print(" dir=("); Serial.print(dirX[calCh],2); Serial.print(","); Serial.print(dirY[calCh],2);
      Serial.print(") gain "); Serial.print(gainDeg[calCh],1); Serial.println(" deg @1.5V");
    } else Serial.println(" too weak (<1.5deg) -- DISABLED");
  }
}
static void calTick() {
  uint32_t dt = millis() - phT0;
  switch (phase) {
    case 0:                                      // 等通道层完成锁定并放气进去
      if (lockSt[calCh] == 2) phaseTo(1);
      else if (dt > 3000) { sayMode("cal: channel never inflated?"); calFinishCh(); chRequest(calCh, 0); phaseTo(3); }
      break;
    case 1: if (dt >= 3000) { calAx = calAy = 0; calN = 0; phaseTo(2); } break;
    case 2:                                      // 0.5s 平均
      calAx += curSx; calAy += curSy; calN++;
      if (dt >= 500) {
        calAx /= calN; calAy /= calN;            // 平均值再判向
        calFinishCh(); chRequest(calCh, 0); phaseTo(3);
      }
      break;
    case 3:                                      // 等放完+锁自动打开，歇 1s
      if (lockSt[calCh] == 0 && dt >= 1000) {
        if (calCh >= 5) { mode = MODE_IDLE; calCh = -1; sayMode("calibration done ('?' to review)"); }
        else calBeginCh(calCh + 1);
      }
      break;
  }
}

// ── 回放 ────────────────────────────────────────────────────────────────
static void rpGoWp(int i) {
  rpWp = i; rpInTol = false; phaseTo(0);
  if (g_host) {
    Serial.print("[tch] -> waypoint "); Serial.print(i + 1); Serial.print("/"); Serial.print(wpN);
    Serial.print(" ("); Serial.print(wpSx[i],1); Serial.print(","); Serial.print(wpSy[i],1); Serial.println(")");
  }
}
static void rpStart() {
  if (mode != MODE_IDLE) { sayMode("busy -- 's' first"); return; }
  if (!g_bno)  { sayMode("no IMU, cannot replay"); return; }
  if (!anyCal) { sayMode("no direction table -- run 'c' first"); return; }
  if (wpN == 0){ sayMode("no waypoints -- teach with 'm' first"); return; }
  mode = MODE_REPLAY;
  rpGoWp(0);                                     // 首个有效驱动先触发对应锁阀预充100ms
}
static void rpEnd() {
  for (int c = 0; c < 6; c++) chRequest(c, 0);   // 放气；锁由通道层等泄压后自动打开
  mode = MODE_IDLE; rpWp = -1;
  sayMode("replay done -- venting, locks auto-open when pressure gone");
}
static void rpTick() {
  float ex = wpSx[rpWp] - curSx, ey = wpSy[rpWp] - curSy;
  float en = sqrtf(ex*ex + ey*ey);
  for (int c = 0; c < 6; c++) {
    if (!calValid[c]) continue;
    float u = Kp * (ex * dirX[c] + ey * dirY[c]);
    chRequest(c, u > 0 ? (int)u : 0);
  }
  if (en < WP_TOL_DEG) {
    if (!rpInTol) { rpInTol = true; rpInTolT0 = millis(); }
    else if (millis() - rpInTolT0 >= WP_SETTLE_MS) {
      if (g_host) { Serial.print("[tch] waypoint "); Serial.print(rpWp + 1); Serial.println(" reached"); }
      if (rpWp >= wpN - 1) rpEnd(); else rpGoWp(rpWp + 1);
    }
  } else {
    rpInTol = false;
    if (millis() - phT0 >= WP_TIMEOUT_MS) {
      if (g_host) {
        Serial.print("[tch] waypoint "); Serial.print(rpWp + 1);
        Serial.print(" TIMEOUT, |e|="); Serial.print(en,1); Serial.println(" deg, skipping");
      }
      if (rpWp >= wpN - 1) rpEnd(); else rpGoWp(rpWp + 1);
    }
  }
}

static void abortAll(const char *why) {
  allSafe(); mode = MODE_IDLE; calCh = -1; rpWp = -1;
  if (!g_host) return;                             // 已进安全态；无宿主就别为一行日志冒僵死险
  Serial.print("[tch] ABORT ("); Serial.print(why); Serial.println("): valves 0, all locks forced open");
}

// ── 打印 ────────────────────────────────────────────────────────────────
static void printStatus() {
  Serial.print("[tch] fw "); Serial.print(FW_VER); Serial.println(" (lock leads drive by 100ms; full-vent forces all locks open)");
  Serial.print("[tch] ADC "); Serial.println(adcOn ? "on (measured zero-pressure release)" : "off (muscle drive inhibited)");
  Serial.println("[tch] direction table:");
  for (int c = 0; c < 6; c++) {
    Serial.print("[tch]   V"); Serial.print(c); Serial.print(" {"); Serial.print(CH_NAME[c]); Serial.print("} ");
    if (calValid[c]) {
      Serial.print("dir=("); Serial.print(dirX[c],2); Serial.print(","); Serial.print(dirY[c],2);
      Serial.print(") gain "); Serial.print(gainDeg[c],1); Serial.println(" deg");
    } else Serial.println("--");
  }
  Serial.print("[tch] waypoints ("); Serial.print(wpN); Serial.println("):");
  for (int i = 0; i < wpN; i++) {
    Serial.print("[tch]   #"); Serial.print(i + 1); Serial.print(" (");
    Serial.print(wpSx[i],1); Serial.print(","); Serial.print(wpSy[i],1); Serial.println(")");
  }
  Serial.print("[tch] Kp="); Serial.print(Kp,0); Serial.println(" mV/deg");
}

// ── IMU 标准帧（GUI 联动）───────────────────────────────────────────────
static void bnoStream() {
  uint8_t eul[6], gyr[6], lia[6];
  if (!bnoRd(0x1A, eul, 6) || !bnoRd(0x14, gyr, 6) || !bnoRd(0x28, lia, 6)) return;
  int cal = bnoR8(0x35), temp = bnoR8(0x34);
  Serial.print(le16(eul) / 16.0, 2);     Serial.print(' ');
  Serial.print(le16(eul + 2) / 16.0, 2); Serial.print(' ');
  Serial.print(le16(eul + 4) / 16.0, 2); Serial.print(" | ");
  for (int i = 0; i < 4; i++) { Serial.print(qNow[i], 4); Serial.print(' '); }
  Serial.print("| ");
  for (int i = 0; i < 3; i++) { Serial.print(le16(gyr + 2 * i) / 16.0, 1); Serial.print(' '); }
  Serial.print("| ");
  for (int i = 0; i < 3; i++) { Serial.print(le16(lia + 2 * i) / 100.0, 2); Serial.print(' '); }
  Serial.print("| "); Serial.print(temp);
  Serial.print(" | S"); Serial.print((cal >> 6) & 3);
  Serial.print(" G");   Serial.print((cal >> 4) & 3);
  Serial.print(" A");   Serial.print((cal >> 2) & 3);
  Serial.print(" M");   Serial.print(cal & 3);
  Serial.print(" | ");                           // 第 7 段：20 路主动/被动位移 mm，掉线打 x
  for (int i = 1; i <= ENC_COUNT; i++) {
    if (encOn[i]) Serial.print(encMm[i], 2); else Serial.print('x');
    Serial.print(' ');
  }
  Serial.print("| ");                            // 第 8 段：六路实测压 kPa（无 ADC 打 x）
  for (int c = 0; c < 6; c++) {
    if (adcOn) Serial.print(chKpa(c), 0); else Serial.print('x');
    Serial.print(' ');
  }
  Serial.print("| ");                            // 第 9 段：六路锁状态 o/w/L
  for (int c = 0; c < 6; c++) Serial.print(lockSt[c] == 0 ? 'o' : lockSt[c] == 1 ? 'w' : 'L');
  Serial.print(" | ");                          // 第 10 段：六路 AD7606 原始有符号计数（压力标定用）
  for (int c = 0; c < 6; c++) {
    if (adcOn) Serial.print(adcRaw[HW_CH[c]]); else Serial.print('x');
    Serial.print(' ');
  }
  Serial.println();
}

// ── 串口指令 ────────────────────────────────────────────────────────────
static char lineBuf[56]; static uint8_t lineLen = 0;   // M<6锁>;<6电压> 最长约43字符
static void handleLine(const char *ln) {
  if (!ln[0]) return;
  char c = ln[0]; float v = atof(ln + 1);
  switch (c) {
    case 'z':
      if (mode != MODE_IDLE) { sayMode("busy"); break; }
      if (bnoQuat()) { for (int i=0;i<4;i++) qRef[i]=qNow[i]; sayMode("zeroed"); }
      break;
    case 'k':                                     // 压力传感器重新对零；只允许全机已强制放气时执行
      if (mode != MODE_IDLE) { sayMode("busy"); break; }
      if (!adcOn) { sayMode("pressure zero refused: ADC absent"); break; }
      if (!forceOpenAll) { sayMode("pressure zero refused: send s/u and vent first"); break; }
      { bool driven = false;
        for (int i=0;i<6;i++) if (chReq[i] != 0 || outMv[i] != 0) driven = true;
        if (driven) { sayMode("pressure zero refused: valves not zero"); break; } }
      adcZero();
      Serial.print("[tch] pressure zeroed");
      for (int i=0;i<6;i++) { Serial.print(' '); Serial.print(kpaZero[i], 1); }
      Serial.println();
      break;
    case 'A': {                                   // ADC raw→kPa 标定：A<slope>,<intercept>
      if (mode != MODE_IDLE) { sayMode("ADC calibration refused: busy"); break; }
      const char *cm = strchr(ln + 1, ',');
      if (!cm) { sayMode("usage: A<kPa/count>,<kPa intercept>"); break; }
      float slope = atof(ln + 1), intercept = atof(cm + 1);
      bool driven = false;
      for (int i=0;i<6;i++) if (chReq[i] != 0 || outMv[i] != 0 || lockSt[i] != 0) driven = true;
      if (!adcOn || driven || slope < 0.01f || slope > 0.2f || intercept < -1000.0f || intercept > 1000.0f) {
        sayMode("ADC calibration refused: ADC/state/value invalid"); break;
      }
      adcTableN = 0;                              // 线性 A 命令显式退出分段表模式
      adcKpaSlope = slope; adcKpaIntercept = intercept;
      adcZero();                                  // 新公式下立即重取六路零压偏置
      Serial.print("[tch] ADC slope "); Serial.print(adcKpaSlope, 9);
      Serial.println(" kPa/count; six channel zero refreshed");
      break;
    }
    case 'T': {                                   // 多点分段表：T! 清空/对零；T<idx>,<deltaRaw>,<kPa>
      if (mode != MODE_IDLE || !adcOn) { sayMode("ADC table refused: busy/ADC absent"); break; }
      bool driven = false;
      for (int i=0;i<6;i++) if (chReq[i] != 0 || outMv[i] != 0 || lockSt[i] != 0) driven = true;
      if (driven) { sayMode("ADC table refused: valves/locks active"); break; }
      if (ln[1] == '!') {
        adcTableN = 0;
        adcZero();                                // 表以六路各自当前零压 raw 为原点
        sayMode("ADC piecewise table cleared; six channel raw zero refreshed");
        break;
      }
      const char *c1 = strchr(ln + 1, ',');
      const char *c2 = c1 ? strchr(c1 + 1, ',') : nullptr;
      if (!c1 || !c2) { sayMode("usage: T<idx>,<deltaRaw>,<kPa>"); break; }
      int idx = atoi(ln + 1);
      float dx = atof(c1 + 1), pk = atof(c2 + 1);
      bool valid = idx == adcTableN && idx >= 0 && idx < ADC_TABLE_MAX && dx >= 0.0f && pk >= 0.0f;
      if (idx == 0) valid = valid && dx < 5.0f && pk < 1.0f;
      if (idx > 0) valid = valid && dx > adcTableDx[idx-1] && pk > adcTableKpa[idx-1];
      if (!valid) { sayMode("ADC table point refused: order/value invalid"); break; }
      adcTableDx[idx] = dx; adcTableKpa[idx] = pk; adcTableN = idx + 1;
      Serial.print("[tch] ADC table point "); Serial.print(idx); Serial.print(" dx=");
      Serial.print(dx, 2); Serial.print(" P="); Serial.println(pk, 2);
      break;
    }
    case 'm':
      if (mode != MODE_IDLE) { sayMode("busy"); break; }
      if (wpN >= WP_MAX) { sayMode("waypoint list full"); break; }
      wpSx[wpN] = curSx; wpSy[wpN] = curSy; wpN++;
      Serial.print("[tch] waypoint #"); Serial.print(wpN); Serial.print(" marked (");
      Serial.print(curSx,1); Serial.print(","); Serial.print(curSy,1); Serial.println(")");
      break;
    case 'x': wpN = 0; sayMode("waypoints cleared"); break;
    case 'c': calStart(); break;
    case 'g': rpStart(); break;
    case 's': abortAll("user"); break;
    case 'p': if (v > 0) Kp = v; Serial.print("[tch] Kp = "); Serial.print(Kp,0); Serial.println(" mV/deg"); break;
    case 'u': allSafe(); sayMode("all channels venting; all locks forced open"); break;
    case 'r': for (int i=1;i<=ENC_COUNT;i++) encZeroSet[i]=false; sayMode("20 encoders re-zeroed (next poll)"); break;
    case 'b':                                    // 软复位：板子埋在产品里按不着 RST 用
      allSafe();                                 // 全机放气定义：比例阀清零、锁阀全部强制打开
      { bool ready = true; for (int i=0;i<6;i++) if (lockSt[i] != 0) ready = false;
        if (!ready) { sayMode("reboot refused: venting; retry 'b' after all locks show o"); break; } }
      sayMode("rebooting..."); delay(50);
      NVIC_SystemReset();
      break;
    case 'N':                                    // Default 1专用：仅V2/V3允许共充
    case 'M': {                                  // MPC显式锁动作+实际电压，单行原子更新
      if (mode != MODE_IDLE) { sayMode("busy"); break; }
      coContractStream = false;
      const char *sep = strchr(ln + 1, ';');
      if (!sep) { sayMode("usage: M<6lock>;<6mV>"); break; }
      bool planned[6] = {false};
      int wanted[6] = {0};
      bool valid = true;
      const char *p = ln + 1;
      for (int k = 0; k < 6; k++) {
        planned[k] = atoi(p) != 0;
        if (k < 5) {
          const char *cm = strchr(p, ',');
          if (!cm || cm >= sep) { valid = false; break; }
          p = cm + 1;
        }
      }
      p = sep + 1;
      for (int k = 0; valid && k < 6; k++) {
        wanted[k] = atoi(p);
        if (k < 5) {
          const char *cm = strchr(p, ',');
          if (!cm) { valid = false; break; }
          p = cm + 1;
        }
      }
      if (!valid) { sayMode("usage: M<6lock>;<6mV>"); break; }
      phiPairStream = (c == 'N');

      // 锁动作也遵守三对拮抗硬约束。切换方向时优先保留当前仍有锁/气的一侧，
      // 防止两侧锁请求互相等待；该侧释放并放空后，下一拍才允许对侧开始预锁。
      for (int left = 0; left < 6; left += 2) {
        int right = left + 1;
        if (phiPairStream && left == 2) continue;
        if (planned[left] && planned[right]) {
          bool la = lockSt[left] != 0 || outMv[left] > 0 || !pressureEmpty(left);
          bool ra = lockSt[right] != 0 || outMv[right] > 0 || !pressureEmpty(right);
          if (la && !ra) planned[right] = false;
          else if (ra && !la) planned[left] = false;
          else if (wanted[left] >= wanted[right]) planned[right] = false;
          else planned[left] = false;
        }
      }
      bool anyLock = false;
      explicitLockMode = true;
      for (int k = 0; k < 6; k++) {
        lockReq[k] = planned[k];
        anyLock = anyLock || lockReq[k];
        if (!lockReq[k]) wanted[k] = 0;            // 无锁动作绝不接受非零比例阀电压
      }
      if (anyLock) forceOpenAll = false;           // 零电压预锁也必须退出强制全开状态
      requestVector(wanted, CLAMP_MV);
      extStream = true; extOpenLoop = false; extLastMs = millis();
      break;
    }
    case 'W':                                    // PI流：沿用“非零电压自动预锁”
    case 'O':                                    // Qt普通开环流：保持对抗互斥
    case 'C': {                                  // Debug手动压力：允许对抗通道共收缩
      if (mode != MODE_IDLE) { sayMode("busy"); break; }
      explicitLockMode = false;
      phiPairStream = false;
      coContractStream = (c == 'C');
      for (int k = 0; k < 6; k++) lockReq[k] = false;
      int wanted[6] = {0};
      const char *p = ln + 1;
      for (int k = 0; k < 6; k++) {
        wanted[k] = atoi(p);
        const char *cm = strchr(p, ',');
        if (!cm) break;
        p = cm + 1;
      }
      const int limitMv = (c == 'W') ? CLAMP_MV : OPEN_LOOP_CLAMP_MV;
      if (coContractStream) {
        for (int k = 0; k < 6; k++) chRequest(k, wanted[k], limitMv);
      } else {
        requestVector(wanted, limitMv);
      }
      // W/O 全零只清空比例阀请求，不能决定开锁；每路锁阀只在其实测压力为 0 后自动打开。
      // 只有显式的 s/u 全机放气（以及 W 闭环安全看门狗）才允许
      // 调用 allSafe() 强制打开全部锁阀；O 开环看门狗不带压开锁。
      extStream = true; extOpenLoop = (c != 'W'); extLastMs = millis();
      break;
    }
    case 'V': {                                  // 手动单路（无看门狗）：V<ch>,<mV>
      if (mode != MODE_IDLE) { sayMode("busy"); break; }
      explicitLockMode = false;
  coContractStream = false;
  phiPairStream = false;
      for (int i = 0; i < 6; i++) lockReq[i] = false;
      const char *cm = strchr(ln, ',');
      int k = ln[1] - '0';
      if (k < 0 || k > 5 || !cm) { sayMode("usage: V<ch>,<mV> e.g. V0,800"); break; }
      int mv = atoi(cm + 1);
      if (mv > 0) chRequest(OPP_CH[k], 0);          // 手动换向先让对侧放气，实测放空后本侧才能动
      chRequest(k, mv);
      Serial.print("[tch] V"); Serial.print(k); Serial.print(" req "); Serial.println(chReq[k]);
      break;
    }
    case '?': printStatus(); break;
    default:  sayMode("cmds: c | z | k | A<slope>,<intercept> | T! / T<i>,<dx>,<kPa> | m | x | g | s | p<Kp> | u | M<6lock>;<6mv> | W<6mv> | O<6mv> | V<ch>,<mv> | ?"); break;
  }
}

// ── 主程序 ──────────────────────────────────────────────────────────────
void setup() {
  g_host = true; lastRxMs = millis();              // 开机 500ms 宽限：让自检横幅能打出来
  for (int c = 0; c < 6; c++) { digitalWrite(LOCK_PIN[c], HIGH); pinMode(LOCK_PIN[c], OUTPUT); digitalWrite(LOCK_PIN[c], HIGH); }

  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}

  Wire.begin(); Wire.setClock(400000);
  Serial.print("[tch] fw "); Serial.println(FW_VER);
  for (int d = 0; d < 3; d++) {
    dacOK[d] = dacBegin(DAC_BASE + d);
    Serial.print("[tch] DAC 0x"); Serial.print(DAC_BASE + d, HEX);
    Serial.println(dacOK[d] ? " ok" : " MISSING");
  }
  adcInit();
  allSafe();
  delay(300); adcZero();                         // 阀全零后测监测口零点（六路各自扣除）
  if (adcOn) {
    Serial.print("[tch] kPa zero");
    for (int c = 0; c < 6; c++) { Serial.print(' '); Serial.print(kpaZero[c], 0); }
    Serial.println();
  }

  pinMode(ENC_DE_PIN, OUTPUT); digitalWrite(ENC_DE_PIN, LOW);
  ENC_UART.begin(115200);
  Serial.println("[tch] encoders: polling active 1..10 + passive 11..20 on RS-485 ('r' = re-zero)");

  bool imu = bnoInit();
  Serial.println(imu ? "[tch] BNO055 ok" : "[tch] BNO055 not found -- teach/replay unavailable");
  if (imu) { delay(100); if (bnoQuat()) for (int i=0;i<4;i++) qRef[i]=qNow[i]; }
  sayMode("teach-replay ready: c=calibrate, move arm & m=mark, g=replay, s=stop, ?=status");
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    lastRxMs = millis();
    g_host = true;   // 刚收到字节=宿主在场；否则 >500ms 静默后的第一条指令，sayMode 回话会被吞掉
    if (ch == '\n' || ch == '\r') { lineBuf[lineLen] = 0; handleLine(lineBuf); lineLen = 0; }
    else if (lineLen < sizeof(lineBuf) - 1) lineBuf[lineLen++] = ch;
  }

  adcTick();                                     // 200 Hz 压力采样
  encTick();                                     // 编码器非阻塞轮询（每圈推一步）

  static uint32_t lastCtl = 0;
  if (millis() - lastCtl >= CTL_MS) {            // 20 Hz：姿态 + 模式机 + 通道层
    lastCtl = millis();
    if (g_bno && bnoQuat()) {
      getSwing(&curSx, &curSy);
      // 角度急停已按作者要求取消（2026-09-04）：安全由气源侧最大气压兜底，全系统口径统一
      if (mode == MODE_CAL)    calTick();
      if (mode == MODE_REPLAY) rpTick();
    }
    // W/O 流看门狗：GUI 断线/停播时不能让阀悬空充着气
    if (extStream && millis() - extLastMs > 2000) {
      bool any = false;
      for (int c = 0; c < 6; c++) if (chReq[c] > 0 || lockSt[c] != 0) any = true;
      if (extOpenLoop) {
        // 开环严格遵守“有压就锁”：超时只撤掉比例阀请求，锁状态机继续根据实测压力处理。
        // 压力仍高于零压阈值时保持锁定；确认回零后才自动开锁。
        extStream = false; extOpenLoop = false; coContractStream = false; phiPairStream = false;
        for (int c = 0; c < 6; c++) chRequest(c, 0);
        if (any) sayMode("O-stream timeout -- valves 0; pressurized channels remain locked");
      } else {
        allSafe();
        if (any) sayMode("W-stream timeout -- venting all, locks forced open");
      }
    }
    chTick();                                    // 锁跟气压走（模式机之后，吃最新请求）
  }

  // ⚠ 无宿主不打印：GIGA(mbed) 的 USB CDC 在宿主断开后继续写会把发送端卡死，
  //   loop 冻结、下次连上"3 秒无数据"，只能复位才能救（重烧其实是顺带复位）。
  //   判据用【心跳】而非 (bool)Serial —— mbed 核的 DTR 状态断开后可能永远为
  //   真（实测 2026-09-03：门控用 DTR 照样僵死）。宿主(qt_gui)每 150ms 发一个
  //   空行；500ms 收不到任何字节即视为无人，停止打印。串口助手想看流：随便发
  //   一个回车即可（每个字节续命 0.5 秒）。控制/锁/看门狗不受影响。
  g_host = ((bool)Serial) && (millis() - lastRxMs < 500);
  const bool host = g_host;

  static uint32_t lastImu = 0;                   // 20 Hz IMU+压力流（PI闭环反馈）
  if (host && g_bno && millis() - lastImu >= 50) { lastImu = millis(); bnoStream(); }

  static uint32_t lastTele = 0;                  // 500 ms 遥测
  if (!host) return;
  if (millis() - lastTele >= 500) {
    lastTele = millis();
    Serial.print("[tch] ");
    Serial.print(mode == MODE_IDLE ? "IDLE" : mode == MODE_CAL ? "CAL" : "REPLAY");
    Serial.print(" s=("); Serial.print(curSx,1); Serial.print(","); Serial.print(curSy,1); Serial.print(")");
    if (mode == MODE_REPLAY && rpWp >= 0) {
      Serial.print(" wp "); Serial.print(rpWp + 1); Serial.print("/"); Serial.print(wpN);
      Serial.print(" tgt=("); Serial.print(wpSx[rpWp],1); Serial.print(","); Serial.print(wpSy[rpWp],1); Serial.print(")");
    }
    Serial.print(" mv");
    for (int c = 0; c < 6; c++) { Serial.print(' '); Serial.print(outMv[c]); }
    if (adcOn) {
      Serial.print(" kPa");
      for (int c = 0; c < 6; c++) { Serial.print(' '); Serial.print(chKpa(c), 0); }
    }
    Serial.print(" lk ");
    for (int c = 0; c < 6; c++) Serial.print(lockSt[c] == 0 ? 'o' : lockSt[c] == 1 ? 'w' : 'L');
    Serial.println();
  }
}
