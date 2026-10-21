# -*- coding: utf-8 -*-
"""
gui_station.py — 本地控制台（PyQt5 + pyqtgraph），取代 HTML GUI 的控制职能

架构（2026-09-03 定版，作者要求去 HTML 化）：
  单进程单线程。串口唯一持有者就是本程序；MPC 与模型同进程直接调用
  （复用 mpc_station.py 的 Link/Plant/MPC/姿态换算，不复制代码）。
  QTimer 20ms：收串口帧（非阻塞）；QTimer 100ms：控制拍（MPC / 手动电压保活）。

功能：
  连接/断开、对零、示教记点/清空/导入导出(JSON)、MPC 播放/停止、全机放气、
  六路手动电压滑条（带看门狗保活）、s_x–s_y 平面图、时间-角度曲线（规划虚线/
  实测实线）、10 路肌肉长度、日志 CSV。
  ident/fit 仍走命令行（python mpc_station.py ident/fit），本界面读它拟好的
  params.json——没有该文件时 MPC 按钮禁用，手动/示教照常可用。

运行：python gui_station.py
"""
import json, math, os, sys, time, csv

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

import mpc_station as st                            # 复用全部控制/模型代码

DT_MS = 100                                         # 控制拍
POLL_MS = 20                                        # 串口收流

class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAM 肩关节控制台（本地）")
        self.link = None
        self.qref = None
        self.teach = []                             # [(q_model_rad(3), dwell_s)]
        self.mpc = None; self.ref = None; self.k = 0
        self.x = np.zeros(12); self.q_prev = None
        self.log = []
        self.plot_t0 = time.time()
        self.hist = {k: [] for k in ("t", "rx", "ry", "mx", "my")}
        self.params = None
        if os.path.exists(st.PARAMS_JSON):
            self.params = json.load(open(st.PARAMS_JSON))
        self._build_ui()
        self.poll_timer = QtCore.QTimer(self); self.poll_timer.timeout.connect(self._poll)
        self.ctl_timer = QtCore.QTimer(self);  self.ctl_timer.timeout.connect(self._tick)
        self.poll_timer.start(POLL_MS); self.ctl_timer.start(DT_MS)

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        w = QtWidgets.QWidget(); self.setCentralWidget(w)
        G = QtWidgets.QGridLayout(w)

        # 行0：串口
        row = QtWidgets.QHBoxLayout()
        self.cb_port = QtWidgets.QComboBox(); self._refresh_ports()
        b_rf = QtWidgets.QPushButton("刷新"); b_rf.clicked.connect(self._refresh_ports)
        self.b_conn = QtWidgets.QPushButton("连接"); self.b_conn.clicked.connect(self._toggle_conn)
        b_zero = QtWidgets.QPushButton("对零"); b_zero.clicked.connect(self._zero)
        self.cb_mount = QtWidgets.QComboBox(); self.cb_mount.addItems(["ry-", "ry+", "none"])
        self.lb_stat = QtWidgets.QLabel("未连接")
        for x in (QtWidgets.QLabel("串口"), self.cb_port, b_rf, self.b_conn, b_zero,
                  QtWidgets.QLabel("安装"), self.cb_mount, self.lb_stat):
            row.addWidget(x)
        row.addStretch()
        G.addLayout(row, 0, 0, 1, 2)

        # 行1：平面图 + 时间曲线
        pg.setConfigOptions(antialias=True)
        self.pw_plane = pg.PlotWidget(title="s_x – s_y 平面（°）")
        self.pw_plane.setAspectLocked(True); self.pw_plane.showGrid(x=True, y=True, alpha=0.3)
        self.sc_teach = pg.ScatterPlotItem(symbol="x", size=12, pen=pg.mkPen("#888", width=2))
        self.sc_ref = pg.ScatterPlotItem(symbol="o", size=12, pen=pg.mkPen("#e6a23c", width=2), brush=None)
        self.sc_cur = pg.ScatterPlotItem(symbol="o", size=10, brush=pg.mkBrush("#409eff"))
        for it in (self.sc_teach, self.sc_ref, self.sc_cur): self.pw_plane.addItem(it)
        G.addWidget(self.pw_plane, 1, 0)

        self.pw_time = pg.PlotWidget(title="时间-角度（虚线=规划，实线=实测）")
        self.pw_time.showGrid(x=True, y=True, alpha=0.3); self.pw_time.addLegend(offset=(4, 4))
        dash = QtCore.Qt.DashLine
        self.cv = {
            "rx": self.pw_time.plot(pen=pg.mkPen("#f56c6c", style=dash), name="s_x 规划"),
            "mx": self.pw_time.plot(pen=pg.mkPen("#f56c6c", width=2), name="s_x 实测"),
            "ry": self.pw_time.plot(pen=pg.mkPen("#67c23a", style=dash), name="s_y 规划"),
            "my": self.pw_time.plot(pen=pg.mkPen("#67c23a", width=2), name="s_y 实测"),
        }
        G.addWidget(self.pw_time, 1, 1)
        G.setColumnStretch(0, 1); G.setColumnStretch(1, 2); G.setRowStretch(1, 1)

        # 行2：示教
        row = QtWidgets.QHBoxLayout()
        b_rec = QtWidgets.QPushButton("记录当前点"); b_rec.clicked.connect(self._record)
        self.sp_dwell = QtWidgets.QDoubleSpinBox(); self.sp_dwell.setRange(0.5, 20); self.sp_dwell.setValue(2.0)
        b_clr = QtWidgets.QPushButton("清空"); b_clr.clicked.connect(self._clear_teach)
        b_sv = QtWidgets.QPushButton("导出"); b_sv.clicked.connect(self._save_teach)
        b_ld = QtWidgets.QPushButton("导入"); b_ld.clicked.connect(self._load_teach)
        self.lb_teach = QtWidgets.QLabel("0 个点")
        for x in (QtWidgets.QLabel("示教"), b_rec, QtWidgets.QLabel("段时长s"), self.sp_dwell,
                  b_clr, b_sv, b_ld, self.lb_teach):
            row.addWidget(x)
        row.addStretch()
        G.addLayout(row, 2, 0, 1, 2)

        # 行3：MPC 驱动
        row = QtWidgets.QHBoxLayout()
        self.b_play = QtWidgets.QPushButton("▶ MPC 播放"); self.b_play.clicked.connect(self._toggle_play)
        b_vent = QtWidgets.QPushButton("全机放气"); b_vent.clicked.connect(self._vent)
        b_vent.setStyleSheet("background:#7a2f2f;color:#fff")
        self.ck_guard = QtWidgets.QCheckBox("45° 保护"); self.ck_guard.setChecked(True)
        self.lb_mpc = QtWidgets.QLabel("MPC 就绪" if self.params else "无 params.json（先 ident+fit）——MPC 禁用")
        if not self.params: self.b_play.setEnabled(False)
        for x in (self.b_play, b_vent, self.ck_guard, self.lb_mpc): row.addWidget(x)
        row.addStretch()
        G.addLayout(row, 3, 0, 1, 2)

        # 行4：手动电压（六路滑条；MPC 播放时禁用）
        box = QtWidgets.QHBoxLayout()
        self.sliders = []; self.lb_mv = []
        names = ["V0{1,5}", "V1{4,8}", "V2{2,3}", "V3{6,7}", "V4{9}", "V5{10}"]
        for i in range(6):
            v = QtWidgets.QVBoxLayout()
            s = QtWidgets.QSlider(QtCore.Qt.Vertical); s.setRange(0, 2000); s.setFixedHeight(90)
            lb = QtWidgets.QLabel(names[i] + "\n0"); lb.setAlignment(QtCore.Qt.AlignHCenter)
            s.valueChanged.connect(lambda val, j=i: self.lb_mv[j].setText(names[j] + "\n" + str(val)))
            v.addWidget(s, alignment=QtCore.Qt.AlignHCenter); v.addWidget(lb)
            self.sliders.append(s); self.lb_mv.append(lb)
            box.addLayout(v)
        gb = QtWidgets.QGroupBox("手动电压 mV（松手即生效，播放时禁用；固件 <60 视为 0）")
        gb.setLayout(box)
        G.addWidget(gb, 4, 0)

        # 行4右：肌肉长度 + 电压回显
        v = QtWidgets.QVBoxLayout()
        self.lb_lens = QtWidgets.QLabel("肌肉 Δmm：--"); self.lb_lens.setWordWrap(True)
        self.lb_u = QtWidgets.QLabel("下发 mV：--")
        v.addWidget(self.lb_lens); v.addWidget(self.lb_u); v.addStretch()
        G.addLayout(v, 4, 1)

    def _refresh_ports(self):
        from serial.tools import list_ports
        self.cb_port.clear()
        self.cb_port.addItems([p.device for p in list_ports.comports()])

    # ── 串口 ────────────────────────────────────────────────────────────
    def _toggle_conn(self):
        if self.link:
            self._vent()
            try: self.link.ser.close()
            except Exception: pass
            self.link = None; self.b_conn.setText("连接"); self.lb_stat.setText("未连接")
            return
        try:
            self.link = st.Link(self.cb_port.currentText())
            self.link.ser.timeout = 0                # 非阻塞：UI 线程里不许卡
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "串口", f"打不开：{e}\n（别的程序占着？）")
            self.link = None; return
        self.b_conn.setText("断开"); self.lb_stat.setText("已连接，等帧…")

    def _poll(self):
        if self.link: self.link.poll()

    def _zero(self):
        if not self.link or self.link.quat is None:
            QtWidgets.QMessageBox.information(self, "对零", "还没收到 IMU 帧"); return
        self.qref = self.link.quat.copy()
        self.lb_stat.setText("已对零")

    def _meas(self):
        """当前 (q_model, sx°, sy°)；没数据返回 None。"""
        if not self.link or self.link.quat is None or self.qref is None: return None
        st.SRV["mount"] = self.cb_mount.currentText()
        q = st.q_model_from_quat(self.link.quat, self.qref, self.cb_mount.currentText())
        sx, sy = st.st_from_q_model(q)
        return q, sx, sy

    # ── 示教 ────────────────────────────────────────────────────────────
    def _record(self):
        m = self._meas()
        if m is None:
            QtWidgets.QMessageBox.information(self, "示教", "先连接并对零"); return
        q, sx, sy = m
        self.teach.append((q.copy(), float(self.sp_dwell.value())))
        self._teach_view()
        self.lb_stat.setText(f"记点 #{len(self.teach)}  s=({sx:+.1f},{sy:+.1f})°")

    def _teach_view(self):
        pts = [st.st_from_q_model(q) for q, _ in self.teach]
        self.sc_teach.setData([p[0] for p in pts], [p[1] for p in pts])
        self.lb_teach.setText(f"{len(self.teach)} 个点")

    def _clear_teach(self):
        self.teach = []; self._teach_view()

    def _save_teach(self):
        if not self.teach: return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出示教", "teach.json", "JSON (*.json)")
        if not fn: return
        json.dump({"points": [{"q_model_rad": list(map(float, q)), "t": t} for q, t in self.teach]},
                  open(fn, "w"), indent=1)

    def _load_teach(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "导入示教", "", "JSON (*.json)")
        if not fn: return
        blob = json.load(open(fn))
        self.teach = [(np.array(p["q_model_rad"]), float(p.get("t", 2))) for p in blob["points"]]
        self._teach_view()

    # ── MPC ─────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.ref is not None: self._stop("手动停止"); return
        if not self.teach:
            QtWidgets.QMessageBox.information(self, "MPC", "先记至少 1 个示教点"); return
        m = self._meas()
        if m is None:
            QtWidgets.QMessageBox.information(self, "MPC", "先连接并对零"); return
        self.qref = self.link.quat.copy()            # 以起跑瞬间静止姿态为零位
        ref, cur = [np.zeros(3)], np.zeros(3)
        for qm, dur in self.teach:
            n = max(1, int(round(dur / st.DT)))
            for j in range(1, n + 1): ref.append(cur + (qm - cur) * j / n)
            cur = qm
        ref += [cur] * int(2.0 / st.DT)
        self.ref = np.array(ref); self.k = 0
        self.mpc = st.MPC(st.Plant(self.params), N=8)
        self.x = np.zeros(12); self.q_prev = None
        self.log = []; self.hist = {k: [] for k in self.hist}
        self.plot_t0 = time.time()
        for s in self.sliders: s.setEnabled(False); s.setValue(0)
        self.b_play.setText("■ 停止")
        self.lb_mpc.setText(f"MPC 运行：{len(self.teach)} 点 / {len(self.ref)*st.DT:.0f}s")

    def _stop(self, why):
        if self.link:
            for _ in range(2): self.link.send_mv([0]*6)
        if self.log:
            os.makedirs(st.LOG_DIR, exist_ok=True)
            path = os.path.join(st.LOG_DIR, time.strftime("mpc_%Y%m%d_%H%M%S.csv"))
            with open(path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["t", "q1", "q2", "q3", "r1", "r2", "r3"] + [f"u{i}" for i in range(6)])
                wr.writerows(self.log)
            self.lb_mpc.setText(f"{why}，日志 → {os.path.basename(path)}")
        else:
            self.lb_mpc.setText(why)
        self.ref = None; self.mpc = None; self.log = []
        for s in self.sliders: s.setEnabled(True)
        self.b_play.setText("▶ MPC 播放")

    def _vent(self):
        if self.ref is not None: self._stop("全机放气")
        if self.link:
            try: self.link.ser.write(b"s\n")
            except Exception: pass
        for s in self.sliders: s.setValue(0)

    # ── 控制拍（10 Hz）───────────────────────────────────────────────────
    def _tick(self):
        if not self.link: return
        # 状态栏 + 长度显示
        if self.link.lens and any(v is not None for v in self.link.lens):
            self.lb_lens.setText("肌肉 Δmm：" + "  ".join(
                f"{i+1}:{v:+.1f}" if v is not None else f"{i+1}:--"
                for i, v in enumerate(self.link.lens)))
        m = self._meas()
        if m:
            q, sx, sy = m
            self.sc_cur.setData([sx], [sy])

        if self.ref is not None and m:               # ── MPC 拍
            q, sx, sy = m
            if self.ck_guard.isChecked() and st.tilt_deg(q) > 45:
                self._stop("!! 摆角超 45° 停止"); return
            qd = (q - self.q_prev)/st.DT if self.q_prev is not None else np.zeros(3)
            self.q_prev = q
            self.x[:3] = q; self.x[3:6] = 0.7*self.x[3:6] + 0.3*qd
            hor = [self.ref[min(self.k+j, len(self.ref)-1)] for j in range(1, 9)]
            u, xn = self.mpc.step(self.x.copy(), hor)
            self.x[6:] = xn[6:]
            self.link.send_mv(u)
            r_now = self.ref[min(self.k, len(self.ref)-1)]
            rsx, rsy = st.st_from_q_model(r_now)
            self.sc_ref.setData([rsx], [rsy])
            self.lb_u.setText("下发 mV：" + " ".join(str(int(v)) for v in u))
            t = time.time() - self.plot_t0
            for kk, vv in (("t", t), ("rx", rsx), ("ry", rsy), ("mx", sx), ("my", sy)):
                self.hist[kk].append(vv)
            for kk in ("rx", "ry", "mx", "my"):
                self.cv[kk].setData(self.hist["t"], self.hist[kk])
            self.log.append([round(self.k*st.DT, 2)] + [round(math.degrees(v), 3) for v in q]
                            + [round(math.degrees(v), 3) for v in r_now] + [int(v) for v in u])
            self.k += 1
            if self.k >= len(self.ref): self._stop("轨迹完成，已放气")
        else:                                        # ── 手动拍：滑条电压保活（喂看门狗）
            mv = [s.value() for s in self.sliders]
            if any(v > 0 for v in mv):
                self.link.send_mv(mv)
                self.lb_u.setText("下发 mV：" + " ".join(map(str, mv)))
        if self.link.quat is not None and self.qref is not None and m:
            self.lb_stat.setText(f"s=({m[1]:+6.1f},{m[2]:+6.1f})°")

    def closeEvent(self, ev):
        self._vent()
        ev.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Main(); w.resize(1150, 780); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
