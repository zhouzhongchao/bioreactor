r"""Standalone Nano25 reference-sensor recorder.

Python port of the reference-sensor-only path in force_recorder_main.cpp:
records the Schunk Nano25 (via NI-DAQmx) to refdata_<ts>.csv, using the same
header/column format as the C++ recorder so the file is a drop-in for
analysis/data_process.ipynb's loader.

NOTE: this tool's clock (time.perf_counter()) is NOT shared with a separately
running C++ force_recorder.exe session -- it only produces a standalone Nano25
recording (the --noshell case), not a synchronized shell+reference pair.

Usage:
    python record_nano25.py [--out DIR] [--device Dev1] [--rate 1000] [--range 0.5] [--cal PATH]

Example (device Dev2, 500 Hz, factory cal file, custom output dir):
    python record_nano25.py --device Dev2 --rate 500 --cal "C:\path\to\FT58068.cal" --out C:\data\nano25_session1

Running with no arguments uses the defaults: device Dev1, 1000 Hz, +-0.5 V range,
the bundled FT58068.cal, output to .\nano25_recordings\ under the current directory.

Keys while running: s = start saving, z = re-zero (unload the sensor first), Ctrl+C = stop.
"""
from __future__ import annotations

import argparse
import json
import msvcrt
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np

from nano25_reader import DEFAULT_G_MATRIX, Nano25Reader, load_ref_cal_matrix

DEFAULT_CAL_PATH = str(Path(__file__).resolve().parent.parent / "FT58068" / "FT58068.cal")
ZERO_SECONDS = 5.0  # ~5 s of unloaded samples per zero, matches force_recorder_main.cpp
GUI_STREAM_HZ = 20.0
GUI_STREAM_PERIOD_S = 1.0 / GUI_STREAM_HZ


def format_dt(dt: datetime) -> str:
    """Matches ForceUtils.cpp's formatDatetimeWithMilliseconds()."""
    return dt.strftime("%d-%m-%Y_%H-%M-%S") + f".{dt.microsecond // 1000:03d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="nano25_recordings", help="output directory (default: ./nano25_recordings)")
    p.add_argument("--device", default="Dev1", help="NI device name, e.g. Dev1 (default: Dev1)")
    p.add_argument("--rate", type=float, default=1000.0, help="sample rate in Hz (default: 1000)")
    p.add_argument("--range", type=float, dest="voltage_range", default=0.5,
                   help="AI voltage range +- volts (default: 0.5)")
    p.add_argument("--cal", default=DEFAULT_CAL_PATH,
                   help="factory .cal XML or a plain 6x6 CSV; pass --cal \"\" to force the hardcoded fallback matrix")
    p.add_argument("--gui-stream", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.cal:
        try:
            G = load_ref_cal_matrix(args.cal)
        except (OSError, ValueError) as e:
            print(f"ERROR: could not load reference cal matrix: {e}")
            return 1
        cal_source = args.cal
    else:
        G = DEFAULT_G_MATRIX.copy()
        cal_source = "script fallback (nano25_reader.DEFAULT_G_MATRIX)"
        print("NOTE: no --cal given; using the fallback G matrix.\n"
              "      Provide the factory .cal (or a 6x6 CSV) once available.")
    print(f"Nano25 G matrix ({cal_source}):\n{G}")

    reader = Nano25Reader(device=args.device, rate=args.rate, voltage_range=args.voltage_range, G=G)

    lock = Lock()
    pending = deque()  # rows waiting to be written: (t_sys, t_steady, gauge, ft)
    zero_buf = []
    state = {
        "ref_seen": False,
        "zeroing": False,
        "recording": False,
        "latest_ft": np.zeros(6),
        "gui_ft_sum": np.zeros(6),
        "gui_ft_count": 0,
        "received_samples": 0,
        "total_samples": 0,
    }
    gui_stop = Event()
    zero_target = round(ZERO_SECONDS * args.rate)
    t0 = time.perf_counter()

    def on_sample(t_steady, t_sys, gauge, ft):
        with lock:
            state["ref_seen"] = True
            state["latest_ft"] = ft
            # Keep NI-DAQ acquisition at its native rate.  The GUI consumes a
            # non-overlapping 50 ms mean so its force/torque stream is aligned
            # with the 20 Hz BNO055 pose stream without throwing samples away.
            state["gui_ft_sum"] += ft
            state["gui_ft_count"] += 1
            state["received_samples"] += 1
            if state["zeroing"] and len(zero_buf) < zero_target:
                zero_buf.append(gauge.copy())
            if state["recording"]:
                pending.append((t_sys, t_steady, gauge.copy(), ft.copy()))
                state["total_samples"] += 1

    reader.on_sample = on_sample

    print(f"Starting NI-DAQmx acquisition on {reader.channel_spec} @ {args.rate} Hz...")
    if not reader.initialize():
        print("ERROR: could not start the Nano25 DAQ task "
              "(is the mioDAQ plugged in? device name right? check NI MAX)")
        return 1

    csv_path = None
    csv_file = None

    if args.gui_stream:
        def read_gui_commands():
            for command in sys.stdin:
                if command.strip().lower() in ("stop", "quit", "q"):
                    gui_stop.set()
                    break

        Thread(target=read_gui_commands, daemon=True).start()

    def do_zero() -> bool:
        with lock:
            zero_buf.clear()
        state["zeroing"] = True
        start = time.perf_counter()
        try:
            while True:
                with lock:
                    if len(zero_buf) >= zero_target:
                        break
                if gui_stop.is_set():
                    return False
                if time.perf_counter() - start > 5.0:
                    print("Still waiting for samples to zero (sensor connected?)...")
                time.sleep(0.02)
        finally:
            state["zeroing"] = False
        with lock:
            reader.gauge_bias = np.mean(zero_buf, axis=0)
            # Do not mix pre-zero samples into the first displayed mean.
            state["gui_ft_sum"].fill(0.0)
            state["gui_ft_count"] = 0
        return True

    def gauge_bias_csv() -> str:
        return ",".join(f"{v:.8g}" for v in reader.gauge_bias)

    def print_bias():
        print(f"Nano25 gauge bias [V] = {reader.gauge_bias}")

    def drain():
        with lock:
            rows = list(pending)
            pending.clear()
            status_ft = state["latest_ft"].copy()
        for t_sys, t_steady, gauge, ft in rows:
            run_s = t_steady - t0
            gauge_str = ",".join(f"{v:.8g}" for v in gauge)
            ft_str = ",".join(f"{v:.6g}" for v in ft)
            csv_file.write(f"{format_dt(t_sys)},{run_s:.6f},{gauge_str},{ft_str}\n")
        return status_ft

    def take_gui_mean():
        """Return and clear one non-overlapping high-rate F/T averaging window."""
        with lock:
            sample_count = int(state["gui_ft_count"])
            if sample_count:
                status_ft = state["gui_ft_sum"] / sample_count
            else:
                status_ft = state["latest_ft"].copy()
            state["gui_ft_sum"].fill(0.0)
            state["gui_ft_count"] = 0
        return np.asarray(status_ft, dtype=float), sample_count

    def start_saving() -> bool:
        nonlocal csv_path, csv_file
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        session_ts = format_dt(datetime.now())
        csv_path = out_dir / f"refdata_{session_ts}.csv"
        try:
            csv_file = open(csv_path, "w", newline="")
        except OSError as e:
            print(f"ERROR: cannot open output file {csv_path}: {e}")
            return False
        csv_file.write(
            "# nano25_acquisition_py reference recording (one row per DAQ scan)\n"
            f"# started: {format_dt(datetime.now())}\n"
            f"# channels: {reader.channel_spec} (gauge_k = ai_k - ai_(k+8))\n"
            f"# sample_rate_hz: {args.rate}\n"
            f"# voltage_range: +-{args.voltage_range}\n"
            f"# cal_matrix: {cal_source}\n"
            f"# gauge_bias:,{gauge_bias_csv()}\n"
            "# units: g* in V, fx/fy/fz in N, tx/ty/tz in Nm\n"
            "system_time,run_time_s,g0,g1,g2,g3,g4,g5,fx,fy,fz,tx,ty,tz\n"
        )
        print(f"\nSaving to:\n  {csv_path}")
        state["recording"] = True
        return True

    try:
        print("Waiting for data...")
        while not state["ref_seen"] and not gui_stop.is_set():
            time.sleep(0.05)
        if gui_stop.is_set():
            return 0

        if args.gui_stream:
            print("NANO25_EVENT " + json.dumps({
                "state": "zeroing",
                "message": "Keep the sensor unloaded; automatic zeroing is running.",
            }, separators=(",", ":")), flush=True)
        else:
            print("\nUNLOAD the sensor (no contact force), then press Enter to zero...")
            input()
        if not do_zero():
            return 0
        print("Zeroed.")
        print_bias()
        if args.gui_stream:
            print("NANO25_EVENT " + json.dumps({
                "state": "running",
                "message": "Nano25 zeroed; live force/torque streaming is active.",
            }, separators=(",", ":")), flush=True)

        print("Live view only -- NOT saving yet.\n"
              "Press 's' to start saving, 'z' to re-zero (unload sensor first), Ctrl+C to stop.\n")

        with lock:
            last_count = state["received_samples"]
        last_status = time.perf_counter()
        last_rate = last_status
        hz = 0.0
        status_period = GUI_STREAM_PERIOD_S if args.gui_stream else 0.1

        while not gui_stop.is_set():
            status_ft = drain()

            if not args.gui_stream and msvcrt.kbhit():
                c = msvcrt.getch()
                if c in (b"s", b"S"):
                    if not state["recording"] and not start_saving():
                        break
                elif c in (b"z", b"Z"):
                    was_recording = state["recording"]
                    print("\nRe-zeroing -- keep the sensor unloaded...")
                    state["recording"] = False
                    drain()
                    do_zero()
                    print("Re-zeroed.")
                    print_bias()
                    if csv_file is not None:
                        csv_file.write(f"# re-zeroed at {format_dt(datetime.now())},gauge_bias:,{gauge_bias_csv()}\n")
                    state["recording"] = was_recording

            now = time.perf_counter()
            if now - last_status >= status_period:
                with lock:
                    count = state["received_samples"]
                if now - last_rate >= 1.0:
                    hz = (count - last_count) / (now - last_rate)
                    last_count = count
                    last_rate = now
                # Advance on the fixed 20 Hz grid instead of resetting to
                # `now`; this avoids a 20 ms polling loop degrading to 16.7 Hz.
                last_status += status_period
                if now - last_status >= status_period:
                    last_status = now

                conn = "ok" if reader.is_connected else "RECONN"
                if args.gui_stream:
                    status_ft, averaged_samples = take_gui_mean()
                    print("NANO25_DATA " + json.dumps({
                        "state": "running" if reader.is_connected else "reconnecting",
                        "hz": GUI_STREAM_HZ,
                        "source_hz": round(hz, 1),
                        "averaged_samples": averaged_samples,
                        "averaged": True,
                        "force_n": [round(float(v), 6) for v in status_ft[:3]],
                        "torque_nm": [round(float(v), 6) for v in status_ft[3:]],
                    }, separators=(",", ":")), flush=True)
                else:
                    tag = "REC" if state["recording"] else "off"
                    line = (f"[{tag}] nano25 {conn:6s} @{hz:4.0f}Hz "
                            f"F=[{status_ft[0]:7.3f} {status_ft[1]:7.3f} {status_ft[2]:7.3f}] "
                            f"T=[{status_ft[3]:7.3f} {status_ft[4]:7.3f} {status_ft[5]:7.3f}]")
                    sys.stdout.write("\r" + line.ljust(100))
                    sys.stdout.flush()

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        state["recording"] = False
        reader.stop()
        drain()
        if csv_file is not None:
            csv_file.close()

        print("\n\nStopped.", end="")
        if csv_path is None:
            print(" Nothing was saved (saving was never started).")
        else:
            with lock:
                count = state["total_samples"]
            print(f"\n  {count} Nano25 samples -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
