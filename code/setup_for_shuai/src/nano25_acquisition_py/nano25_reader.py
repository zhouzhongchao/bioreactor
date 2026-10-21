"""Reader for the Schunk Nano25 SI-125-3 reference force/torque sensor via NI-DAQmx.

Python port of Nano25Reader.h/.cpp (same wiring, same G-matrix convention, same
lossless continuous-sampling + backdated-timestamp approach) so refdata_*.csv
output from this tool matches what the C++ force_recorder and the analysis
notebooks already expect.

Wiring:
  - 12 single-ended AI channels on one device: gauge signals on ai0:5, gauge
    reference lines on ai8:13.
  - Per-gauge value: g_k = ai_k - ai_(k+8).
  - F/T = G @ (gauge - gauge_bias), where G is the sensor's 6x6 matrix.
"""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np

import nidaqmx
from nidaqmx.constants import AcquisitionType

# The G matrix hardcoded in the lab's Python tool (ftd_live_calib.py) and
# Nano25Reader.cpp's defaultNano25GMatrix() -- fallback when no factory .cal
# / CSV is supplied.
DEFAULT_G_MATRIX = np.array([
    [-0.22646,  -0.13862,  -0.05089,  13.50811,  -0.30940, -13.44625],
    [-0.19850, -15.82667,  -0.24678,   7.80411,   0.38084,   7.88693],
    [26.05721,  -0.75365,  25.80952,  -1.11912,  25.73275,  -1.58570],
    [ 0.00345,  -0.13517,   0.25569,   0.05598,  -0.24477,   0.08085],
    [-0.29081,   0.00855,   0.14870,  -0.12171,   0.14439,   0.10817],
    [-0.00420,  -0.12660,   0.00080,  -0.12107,  -0.00180,  -0.12221],
], dtype=np.float64)

AXIS_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _parse_cal_file(path: str) -> np.ndarray:
    """Parses a Schunk/ATI calibration XML: one UserAxis/Axis element per
    output axis (Name, values=6 floats, optional scale). Row = values/scale
    (ATIDAQ ftconfig convention), matching Nano25Reader.cpp's parseCalFile()."""
    root = ET.parse(path).getroot()
    G = np.zeros((6, 6))
    found = [False] * 6

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # strip XML namespace, if any
        if tag not in ("UserAxis", "Axis"):
            continue
        name = elem.get("Name")
        if name not in AXIS_NAMES:
            continue
        row = AXIS_NAMES.index(name)

        values_str = elem.get("values")
        if values_str is None:
            raise ValueError(f"cal file axis {name} is missing a values attribute")
        vals = [float(v) for v in values_str.split()]
        if len(vals) < 6:
            raise ValueError(f"cal file axis {name} has fewer than 6 values")

        scale = float(elem.get("scale") or 1.0)
        if scale == 0.0:
            raise ValueError(f"cal file axis {name} has scale 0")

        G[row, :] = np.array(vals[:6]) / scale
        found[row] = True

    missing = [AXIS_NAMES[i] for i, ok in enumerate(found) if not ok]
    if missing:
        raise ValueError(f"cal file is missing axis/axes: {', '.join(missing)}")
    return G


def load_ref_cal_matrix(path: str) -> np.ndarray:
    """Loads a 6x6 gauge->F/T matrix. A path ending in .cal is parsed as a
    Schunk/ATI calibration XML; anything else is read as a plain 6x6 CSV
    (matches readCSVToEigenMatrix's format: comma-separated, no header)."""
    if path.lower().endswith(".cal"):
        return _parse_cal_file(path)
    G = np.loadtxt(path, delimiter=",", ndmin=2)
    if G.shape != (6, 6):
        raise ValueError(f"expected a 6x6 matrix CSV, got shape {G.shape}")
    return G


@dataclass
class Nano25Reader:
    """Continuous, lossless DAQmx acquisition of the Nano25, mirroring the
    public shape of the C++ Nano25Reader class.

    Set `on_sample` before calling initialize(). It is invoked on the reader
    thread for every acquired scan with:
        on_sample(t_steady: float,        # time.perf_counter() seconds, backdated
                   t_sys: datetime,       # wall-clock time, backdated
                   gauge: np.ndarray(6,), # volts (signal - reference line)
                   ft: np.ndarray(6,))    # Fx,Fy,Fz [N], Tx,Ty,Tz [Nm]
    Keep it fast -- heavy work (e.g. file I/O) belongs on another thread.
    """
    device: str
    rate: float = 1000.0
    voltage_range: float = 0.5
    G: np.ndarray = field(default_factory=lambda: DEFAULT_G_MATRIX.copy())

    on_sample: Optional[Callable[[float, datetime, np.ndarray, np.ndarray], None]] = None
    gauge_bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    is_connected: bool = field(default=False, init=False)

    _task: "nidaqmx.Task | None" = field(default=None, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _run_thread: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def __post_init__(self):
        self.channel_spec = f"{self.device}/ai0:5,{self.device}/ai8:13"
        # ~20 ms of scans per read: matches Nano25Reader.cpp's block size
        # (rate/50) -- enough batching to be cheap, small enough to keep
        # zeroing/status responsive.
        self._block = max(1, round(self.rate / 50.0))

    def initialize(self) -> bool:
        if not self._create_task():
            return False
        self.is_connected = True
        self._run_thread.set()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._run_thread.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._clear_task()
        self.is_connected = False

    def _create_task(self) -> bool:
        try:
            task = nidaqmx.Task()
            task.ai_channels.add_ai_voltage_chan(
                self.channel_spec, min_val=-self.voltage_range, max_val=self.voltage_range)
            task.timing.cfg_samp_clk_timing(rate=self.rate, sample_mode=AcquisitionType.CONTINUOUS)
            task.start()
        except nidaqmx.DaqError as e:
            print(f"[Nano25] task creation failed: {e}")
            return False
        self._task = task
        return True

    def _clear_task(self) -> None:
        if self._task is not None:
            try:
                self._task.close()
            except nidaqmx.DaqError:
                pass
            self._task = None

    def _read_loop(self) -> None:
        sample_period = 1.0 / self.rate

        while self._run_thread.is_set():
            try:
                data = self._task.read(number_of_samples_per_channel=self._block, timeout=1.0)
            except nidaqmx.DaqError as e:
                print(f"[Nano25] read failed: {e}")
                self.is_connected = False
                self._clear_task()
                # Retry until the device is back or we are asked to stop.
                while self._run_thread.is_set() and not self._create_task():
                    print(f"[Nano25] Reconnecting to {self.device} in 1 s...")
                    self._run_thread.wait(timeout=1.0)
                if self._run_thread.is_set():
                    self.is_connected = True
                continue

            arr = np.asarray(data)  # shape (12, nRead)
            n_read = arr.shape[1]
            if n_read == 0:
                continue
            self.is_connected = True

            # One clock read per block; individual scans are backdated by the
            # DAQ's precise sample spacing (matches Nano25Reader.cpp).
            t_block_steady = time.perf_counter()
            t_block_sys = datetime.now()

            for i in range(n_read):
                scan = arr[:, i]
                gauge = scan[0:6] - scan[6:12]  # signal (ai0:5) - reference line (ai8:13)
                ft = self.G @ (gauge - self.gauge_bias)

                backdate_s = sample_period * (n_read - 1 - i)
                t_steady = t_block_steady - backdate_s
                t_sys = t_block_sys - timedelta(seconds=backdate_s)

                if self.on_sample:
                    self.on_sample(t_steady, t_sys, gauge, ft)
