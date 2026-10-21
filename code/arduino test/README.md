# Arduino 单元测试区

先各自单测、全部通过后再往 `giga-regulator-control` 整合。**每个测试 = 一个 PlatformIO 项目**
（本目录就是唯一正本；OneDrive\文档\PlatformIO\Projects 里的旧副本仅作备份，别在那边按 →）。
互相独立、零共享代码——测不通时只怀疑接线和芯片，不怀疑别的模块。

**用法**：双击 `pam-tests.code-workspace`（或 VS Code 打开工作区文件）→ PIO 状态栏切换项目 →
✓ 编译、→ 编译+烧录、🔌 串口监视器（115200）。烧录前先把占串口的（GUI 连接/监视器）断开。

## 台账

| 测试 | 对象 | 总线/引脚 | 状态 | 判据 |
|---|---|---|---|---|
| `bno055_test` | BNO055 IMU（Wire2） | SDA=D9、SCL=D8 | 待跑 | IDs=A0/FB/32/F；sys status=5；CAL 可到 S3 G3 A3 M3；静置时欧拉角噪声 <0.2° |
| `seq_demo` | 零耗气保持演示：V0..V5 逐组 锁1s→充气(≤2V)→保持→放气→1s→开锁 + BNO055 流同发 | 全部（见 giga-lock-pinmap 表） | 可跑 | GUI「跑 Demo」按钮或串口 g；s=急停 |
| `control_test` | V0{1,5} 单通道最简闭环：IMU 倾角反馈 + P 控制（t<deg> 指令），开环 v<mV> | 0x58ch0 + D2 + Wire2 | 可跑 | t10 后倾角收敛到目标 ±1° 内无振荡 |
| `teach_mode` | 示教-回放闭环 + GUI/MPC 网关固件（W 指令流、帧 v2）。fw `2026-09-04d`：任何有效肌肉驱动前，对应锁阀先充气 1s；比例阀清零后继续锁定，AD7606 实测压力回到零压容差并稳定后才开锁；压力监测不可用时禁止开始新驱动，不以指令或计时猜测放空。帧尾第 8 段为六路实测压 kPa，第 9 段为锁状态 o/w/L；闲置底气 0V | 全部 6 路 + Wire2 + SPI1(AD7606) + Serial2(编码器) | 待重烧验证 2026-09-04 | GUI 连上 2s 内若帧里没有第 8/9 段会提示"固件旧版"。锁状态：o=开、w=锁阀预充、L=锁定/允许肌肉驱动。 |
| `encoder_test` | GJW 拉线编码器 ×10（ID=肌肉号 1..10） | RS-485：MAX485 RO→D19、DI→D18、DE→D10（Serial2） | 通过 2026-09-03 | 在线路读数稳、拉腱对应 ID 动。⚠ GIGA(mbed) 的 flush() 不等移位寄存器——DE 须按字节时间足额持有（ref sketch 的 flush()+30µs 在 GIGA 必失败）。已并入 teach_mode（帧尾第 7 段） |
| （待建）`adc_test` | AD7606 压力采集 | SPI1：SCK=D13、CIPO=D12；CONVST=D39、RST=D40、CS=D41、BUSY=D45 | — | 6 路 1–5 V 读数稳定；断气时读数=1 V 附近 |
| （待建）`dac_itv_test` | GP8403→ITV 调压阀 | Wire：SDA=D20、SCL=D21；0x58/0x59/0x5A | — | 阶跃指令后 ADC 回读跟随；先 `c<阀>,2200,3` 扫斜率（标定第 0 项！） |
| （待建）`ratchet_test` | 6 路棘轮开关阀 | D2–D7，低电平有效 | — | 逐路吸合/释放可听见；上电瞬间不误吸合 |

## GIGA 引脚占用总表（合并测试前先查这里，别撞脚）

| 资源 | 引脚 | 使用者 |
|---|---|---|
| Wire (I2C) | D20/D21 | 3×GP8403 DAC（0x58–0x5A） |
| Wire1 | 显示排线 | GT911 触摸（0x5D） |
| **Wire2** | **D9(SDA2)/D8(SCL2)** | **BNO055**。⚠ 此组无板载上拉，靠模块自带 10k |
| Serial2 | D18/D19（丝印 TX1/RX1！） | RS-485 编码器总线，DE=D10 |
| SPI1 | D11/D12/D13（丝印 SPI 排针） | AD7606；CONVST=D39、RST=D40、CS=D41–44、BUSY=D45–48 |
| GPIO | D2–D7 | 棘轮开关阀 d0–d5（低有效） |

丝印陷阱三连（都在固件头文件里有注释）：丝印 TX1/RX1=**Serial2**、丝印 SPI 排针=**SPI1**、D9/D8=**Wire2**。

## 约定

- 波特率一律 115200；输出一行一帧、空格分隔，方便直接用串口绘图仪或 Python 采。
- 每个 sketch 头部注释写清：接线、已知陷阱、"working 长什么样"。
- 单测全绿后才动 `giga-regulator-control`（整合另开分支）。
