/**
 * encoder_test.ino — 10 路 GJW 拉线编码器冒烟测试（RS-485 Modbus RTU）
 *
 * 从站 ID 约定（作者定版 2026-09-03）：ID k = 肌肉 k（1..10），直接对应。
 *
 * 协议照抄 ref/sketch_position_speed.ino（在本站编码器上验证过）：
 *   功能码 0x03，起始 0x0380，读 4 字 = 单圈位置 u32 + 圈数 i32（大端），
 *   回帧 13 字节，CRC16-Modbus 校验。
 * 位移换算：mm = 展开角(°) × π·D / 360，D = 13.7 mm（ref 实测值；
 *   clone 的 encoder_rs485.h 用 14.0，两处差 2%，以后标定定夺——先用 ref 的）。
 *
 * ⚠ 接线与串口对象（丝印陷阱，ref sketch 这两处是错的/会撞车，此处已改）：
 *   MAX485 RO → D19、DI → D18 —— 丝印 TX1/RX1，但 Arduino 对象是 **Serial2**
 *     （GIGA variant：Serial1 在 D0/D1！用 Serial1 总线会死寂无报错）
 *   MAX485 DE+/RE 并联 → **D10**（ref 用 D2——那是咱们 {1,5} 锁阀的脚，不能占；
 *     README 引脚总表定的就是 D10）
 *   A/B 总线两端 120Ω 端接；MAX485 是 5V 件，RO 需 MAX3485/电平转换（3.3V GPIO）
 *
 * 用法（115200）：上电自动轮询 ID 1..10，一行打一轮：
 *   t=12345  1:+12.345  2:-0.012  3:---  ...  (mm，--- = 不在线)
 *   r  全部置零（上电首帧自动置零）    ?  状态（在线表、参数）
 *
 * 判据（README 台账）：在线的路读数稳定、手拉腱时对应 ID 的读数变化且方向合理。
 */

#include <Arduino.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── 参数 ────────────────────────────────────────────────────────────────
#define ENC_UART        Serial2            // D18=TX、D19=RX（丝印 TX1/RX1！）
static const uint8_t  PIN_RE_DE     = 10;  // DE&/RE：HIGH=发，LOW=收
static const uint8_t  ID_LO         = 1;   // 肌肉 1..10
static const uint8_t  ID_HI         = 10;
static const uint32_t BAUD_485      = 115200;
static const uint16_t REG_BASE      = 0x0380;
static const uint16_t WORDS_TO_READ = 4;
static const uint8_t  ENC_BITS      = 21;
static const double   SPOOL_DIA_MM  = 13.7;
static const int8_t   DIR_SIGN      = +1;  // 方向反了改 -1

static const uint16_t TX_SETTLE_US  = 15;
// DE 持有余量：从 write 之前计时，按字节时间足额持有后再加这么多。必须小于
// Modbus 的 3.5 字符静默（115200 下约 304µs），否则应答开始时我们还占着总线。
static const uint16_t TX_GUARD_US   = 150;
static const uint16_t RX_WINDOW_MS  = 50;  // 静默 ID 的超时代价（对齐 clone/webapp）
static const uint16_t IDLE_GAP_MS   = 2;
static const uint16_t SWEEP_GAP_MS  = 50;  // 两轮之间歇一拍

static const uint32_t COUNTS_PER_REV = (1UL << ENC_BITS);

// ── 每路状态 ────────────────────────────────────────────────────────────
static bool   online[11]  = { false };     // 下标=ID（1..10），0 空着不用
static double zeroDeg[11];
static bool   zeroSet[11] = { false };
static double dispMm[11]  = { 0 };

// ── RS-485 / Modbus（协议与 ref 完全一致）──────────────────────────────
static inline void setTX(){ digitalWrite(PIN_RE_DE, HIGH); }
static inline void setRX(){ digitalWrite(PIN_RE_DE, LOW);  }

static uint16_t crc16_modbus(const uint8_t* d, size_t n){
  uint16_t crc = 0xFFFF;
  for(size_t i=0;i<n;i++){
    crc ^= d[i];
    for(uint8_t b=0;b<8;b++) crc = (crc & 1) ? (uint16_t)((crc >> 1) ^ 0xA001) : (uint16_t)(crc >> 1);
  }
  return crc;
}

static size_t modbusRead03(uint8_t id, uint16_t reg, uint16_t count, uint8_t* rx, size_t cap){
  uint8_t req[8];
  req[0]=id; req[1]=0x03;
  req[2]=(uint8_t)(reg>>8);   req[3]=(uint8_t)(reg&0xFF);
  req[4]=(uint8_t)(count>>8); req[5]=(uint8_t)(count&0xFF);
  uint16_t crc = crc16_modbus(req,6);
  req[6]=(uint8_t)(crc & 0xFF); req[7]=(uint8_t)(crc >> 8);

  while(ENC_UART.available()) (void)ENC_UART.read();

  // ⚠ GIGA(mbed) 的 flush() 不等移位寄存器清空（AVR 才等）——ref sketch 的
  // flush()+30µs 在这块板上会提前放开 DE、截断请求帧 CRC，编码器永远不应答
  // （clone encoder_rs485.cpp 头注释里踩过并记录）。改为：从 write 之前计时，
  // 按 8 字节 × 10 位的线上时间足额持有 DE，再加 TX_GUARD_US 余量。
  setTX(); delayMicroseconds(TX_SETTLE_US);
  uint32_t t0us = micros();
  ENC_UART.write(req, sizeof(req));
  const uint32_t txUs = (uint32_t)sizeof(req) * 10UL * 1000000UL / BAUD_485 + TX_GUARD_US;
  while ((uint32_t)(micros() - t0us) < txUs) {}
  setRX();

  size_t rlen=0; uint32_t t0=millis(), last=t0;
  while(millis()-t0 < RX_WINDOW_MS){
    while(ENC_UART.available()){
      int b = ENC_UART.read();
      if(b>=0 && rlen<cap){ rx[rlen++]=(uint8_t)b; last=millis(); }
    }
    if(rlen>0 && (millis()-last) > IDLE_GAP_MS) break;
  }
  return rlen;
}

static bool parsePosTurns(uint8_t id, const uint8_t* rx, size_t n, uint32_t* pos, int32_t* turns){
  if(n < 13) return false;                       // id+03+len(8)+data8+crc2
  uint16_t rc = (uint16_t)rx[n-2] | ((uint16_t)rx[n-1] << 8);
  if(rc != crc16_modbus(rx, n-2)) return false;
  if(rx[0] != id || rx[1] != 0x03 || rx[2] != 0x08) return false;
  const uint8_t* p = rx + 3;
  *pos   = ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];
  uint32_t tu = ((uint32_t)p[4]<<24)|((uint32_t)p[5]<<16)|((uint32_t)p[6]<<8)|(uint32_t)p[7];
  *turns = (int32_t)tu;
  return true;
}

// 读一路：成功则更新 dispMm[id]，返回是否在线
static bool pollOne(uint8_t id){
  uint8_t rx[32]; uint32_t pos; int32_t turns;
  size_t n = modbusRead03(id, REG_BASE, WORDS_TO_READ, rx, sizeof(rx));
  if(!n || !parsePosTurns(id, rx, n, &pos, &turns)) return false;

  double singleDeg = (double)(pos % COUNTS_PER_REV) * 360.0 / (double)COUNTS_PER_REV;
  double deg = DIR_SIGN * ((double)turns * 360.0 + singleDeg);
  if(!zeroSet[id]){ zeroDeg[id] = deg; zeroSet[id] = true; }   // 首帧自动置零
  dispMm[id] = (deg - zeroDeg[id]) * (M_PI * SPOOL_DIA_MM) / 360.0;
  return true;
}

// ── 串口命令 ────────────────────────────────────────────────────────────
static void handleCmd(char c){
  if(c=='r'){
    for(int i=ID_LO;i<=ID_HI;i++) zeroSet[i]=false;            // 下一轮读到即置零
    Serial.println("# zero all (takes effect next sweep)");
  }else if(c=='?'){
    Serial.print("# dia="); Serial.print(SPOOL_DIA_MM,1);
    Serial.print("mm dir="); Serial.print((int)DIR_SIGN);
    Serial.print(" online:");
    for(int i=ID_LO;i<=ID_HI;i++){ if(online[i]){ Serial.print(' '); Serial.print(i); } }
    Serial.println();
  }
}

// ── 主程序 ──────────────────────────────────────────────────────────────
void setup(){
  pinMode(PIN_RE_DE, OUTPUT); setRX();
  Serial.begin(115200);
  uint32_t t0 = millis();
  while(!Serial && millis()-t0 < 3000){}
  ENC_UART.begin(BAUD_485);

  Serial.println("# encoder_test: muscles 1..10 on RS-485 (Serial2 D18/D19, DE=D10)");
  Serial.println("# cmds: r=zero all, ?=status.  '---'=offline");
}

void loop(){
  while(Serial.available()){ char c=Serial.read(); if(c=='r'||c=='?') handleCmd(c); }

  for(uint8_t id=ID_LO; id<=ID_HI; id++) online[id] = pollOne(id);

  Serial.print("t="); Serial.print(millis());
  for(uint8_t id=ID_LO; id<=ID_HI; id++){
    Serial.print("  "); Serial.print(id); Serial.print(':');
    if(online[id]){ if(dispMm[id]>=0) Serial.print('+'); Serial.print(dispMm[id],3); }
    else Serial.print("---");
  }
  Serial.println("  (mm)");

  delay(SWEEP_GAP_MS);
}
