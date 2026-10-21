/*
  Arduino UNO + FSR 402 LT
  Measuring resistor: 10 ohm

  Output:
  Force in Newton (N)
*/

const int FSR_PIN = A0;

float zeroADC = 0.0;


// ======================
// ADC average
// ======================
float readADC(int samples)
{
  long sum = 0;

  for (int i = 0; i < samples; i++)
  {
    sum += analogRead(FSR_PIN);
  }

  return (float)sum / samples;
}


// ======================
// Auto zero
// ======================
void autoZero()
{
  Serial.println("REMOVE ALL FORCE");
  Serial.println("ZEROING...");

  delay(2000);

  // Discard initial readings
  for (int i = 0; i < 300; i++)
  {
    analogRead(FSR_PIN);
  }

  // Calculate zero offset
  zeroADC = readADC(1000);

  Serial.print("ZeroADC = ");
  Serial.println(zeroADC, 2);

  Serial.println("======================");
  Serial.println("ZERO COMPLETE!");
  Serial.println("ADD WEIGHT NOW");
  Serial.println("======================");

  delay(1000);
}


// ======================
// ADC -> Force (N)
// ======================
float adcToForce(float x)
{
  if (x <= 0)
  {
    return 0.0;
  }

  // 最大标定点是48
  if (x > 48.0)
  {
    x = 48.0;
  }

  // 你的三次拟合公式
  float forceN =
      0.0007792326 * x * x * x
    - 0.0338754830 * x * x
    + 0.4849681440 * x
    - 1.4921246700;

  if (forceN < 0)
  {
    forceN = 0;
  }

  return forceN;
}


// ======================
// Setup
// ======================
void setup()
{
  Serial.begin(115200);

  delay(500);

  Serial.println("FSR 402 LT");
  Serial.println("Unit: N");

  autoZero();
}


// ======================
// Loop
// ======================
void loop()
{
  // 平均10次ADC
  float rawADC = readADC(10);

  // 去零
  float correctedADC = rawADC - zeroADC;

  if (correctedADC < 0)
  {
    correctedADC = 0;
  }

  // 转换成N
  float forceN = adcToForce(correctedADC);

  // 小信号直接归零
  if (correctedADC < 1.0)
  {
    forceN = 0;
  }

  // 只输出N
  Serial.print(forceN, 3);
  Serial.println(" N");

  delay(20);
}