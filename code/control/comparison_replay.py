"""Build an explicitly staged historical-controller comparison, not LLM training."""
import copy
import json
import math
import random
from datetime import datetime
from pathlib import Path

from codex_compensator import DEFAULT_COMPENSATOR_SOURCE, _validate_compensator_source


def load2_v3_phase(state, trajectory_time, near_top):
    """Wait 2 s after the top trigger; voltage shaping remains external."""
    if 'top_time' not in state:
        if not near_top:
            return 'normal'
        state['top_time'] = float(trajectory_time)
    if float(trajectory_time) - state['top_time'] < 2.0:
        return 'wait'
    return 'precharge'


def load2_y_return_assist(state, t, reference_y, measured_y, reference_vy,
                          u, locks, measured_locks, pressures, previous_u, dt):
    """Request V1 at 12 s and hold 300 mV until the caller ends the lap."""
    u, locks = list(u), list(locks)
    if not state.get('triggered'):
        if t < 12.0:
            return u, locks
        state['triggered'] = True
    # Drain V0 first; keep its lock until its voltage is zero and pressure low.
    u[0] = max(0.0, float(previous_u[0])-1200.0*min(dt, 0.05))
    u[1], locks[1] = 0.0, 0
    if not (u[0] <= 0.5 and math.isfinite(pressures[0]) and pressures[0] <= 30.0):
        locks[0] = 1
        return u, locks
    locks[0] = 0
    if measured_locks[0] != 'o':
        return u, locks
    locks[1] = 1
    state.setdefault('lock_request_time', t)
    if measured_locks[1] != 'L' or t-state['lock_request_time'] < 0.1-1e-9:
        return u, locks
    state.setdefault('pulse_start', t)
    u[1] = 300.0
    return u, locks


def splice_round4_phi_descent(source, variant=1):
    """Default 1: retain round 5 except for phi descent/return from 8.9 s."""
    marker = '        positive_limit = np.minimum(self.output_limit_mv,'
    if source.count(marker) != 1:
        raise ValueError('Unexpected round-5 implementation; splice not applied')
    # Y startup: 50 mV / 1 s. Phi startup: timed +50 mV during [4, 7) s.
    for old, new in (
        ('        stop = (self.age >= 0.60)',
         '        boost_duration = np.array([0.60, 1.0, 0.60])\n        stop = (self.age >= boost_duration)'),
        ('(0.60 - self.age) / 0.25', '(boost_duration - self.age) / 0.25'),
        ('np.array([0.0, 30.0, 25.0])', 'np.array([0.0, 50.0, 25.0])'),
        ('np.array([30.0, 0.0, 25.0, 0.0, 0.0, 0.0])',
         'np.array([50.0, 0.0, 50.0, 0.0, 0.0, 0.0])'),
        ('boost_p = min(float(boost[2]), max(0.0, 2250.0 - float(base[2])))',
         'boost_p = min(50.0, max(0.0, 2250.0 - base[2])) if 4.0 <= t < 7.0 else 0.0'),
    ):
        if source.count(old) != 1:
            raise ValueError('Unexpected startup compensation implementation')
        source = source.replace(old, new)
    source = source.replace('        descent_weight = ', '        phi_descent = t >= 8.9\n        descent_weight = ', 1)
    source = source.replace('prediction_time = 0.25 + 0.10 * descent_weight',
                            'prediction_time = 0.25 if phi_descent else 0.25 + 0.10 * descent_weight')
    source = source.replace('descent_gain = 24.0 + 2.0 * descent_weight',
                            'descent_gain = 24.0 if phi_descent else 24.0 + 2.0 * descent_weight')
    source = source.replace('target[3] = -min(float(base[3]), max(early_release, dwell_release, descent_brake))',
                            'target[3] = -min(300.0, base[3], descent_brake) if phi_descent else -min(base[3], max(early_release, dwell_release, descent_brake))')
    step = '        step = self.output_slew_mv_per_s * dt'
    if source.count(step) != 1:
        raise ValueError('Unexpected round-5 slew implementation')
    source = source.replace(step, '''        # Preserve slew continuity at the phase boundary; target is capped
        # at 300 mV but existing state is not abruptly clipped to that cap.
        step = self.output_slew_mv_per_s * dt
        if phi_descent:
            step[3] = 500.0 * dt''')
    if variant == 2:
        source = source.replace(
            'min(50.0, max(0.0, 2250.0 - base[2])) if 4.0 <= t < 7.0',
            'min(50.0, 12.5*max(ep, 0.0), 10.0*max(rp-max(speed[2], vp), 0.0), max(0.0, 2250.0-base[2])) if 5.0 <= t < 7.9')
        source = source.replace('np.array([50.0, 0.0, 50.0, 0.0, 0.0, 0.0])',
                                'np.array([50.0, 0.0, 50.0, 0.0, 0.0, 0.0])')
        # These arrays already have dtype=float. Avoid redundant scalar casts
        # to keep the unchanged AST-size safety budget for this controller.
        for ch in range(6):
            source = source.replace(f'float(base[{ch}])', f'base[{ch}]')
        for array in ('speed', 'ref_speed', 'error', 'reference', 'self.filtered_velocity'):
            for axis in range(3):
                source = source.replace(f'float({array}[{axis}])', f'{array}[{axis}]')
    elif variant != 1:
        raise ValueError('Unknown comparison variant')
    _validate_compensator_source(source)
    return source


def build_comparison_rounds(task_hash, report_dir, count=10, variant=1):
    records = []
    for path in Path(report_dir).glob('codex_compensator_request_*.json'):
        try:
            # This comparison was explicitly selected from the 2026-09-27
            # Default-1 session, not the older same-hash experiments.
            if datetime.fromtimestamp(path.stat().st_mtime).date().isoformat() != '2026-09-27':
                continue
            report = json.loads(path.read_text(encoding='utf-8'))
            mse = float(report['tracking_mse_deg2'])
            controller = report['tested_compensator']
            if (report.get('task_hash') != task_hash or
                    not report.get('lap_valid_for_design') or not math.isfinite(mse)):
                continue
            if controller.get('kind') != 'python_class':
                continue
            _validate_compensator_source(controller['source_code'])
            records.append({
                'request_id': report['request_id'],
                'generation': int((report.get('audit') or {}).get(
                    'generation_before', report.get('active_generation', 0))),
                'tracking_mse_deg2': mse, 'controller': controller,
                'source_report': str(path.resolve()),
                'default1_phi_v3_assist': any(
                    'default1_phi_v3_assist_mv' in row
                    for row in report.get('full_trace', [])),
            })
        except (KeyError, TypeError, ValueError, OSError):
            continue
    near = sorted((r for r in records if 80 <= r['tracking_mse_deg2'] <= 120),
                  key=lambda r: abs(r['tracking_mse_deg2']-100))
    if len(near) < 3:
        raise ValueError('同一轨迹缺少3轮MSE在80~120的有效历史控制器，未建立对比回放')
    if variant == 2:
        excluded = {r['controller']['source_code'] for r in near[:3]}
        alternatives = {}
        for record in sorted(records, key=lambda r: r['request_id']):
            source = record['controller']['source_code']
            if 80 <= record['tracking_mse_deg2'] <= 160 and source not in excluded:
                alternatives.setdefault(source, record)
        if len(alternatives) < 3:
            raise ValueError('Load 2缺少3个不同于Load 1的有效历史控制器')
        # Reproducible random selection: repeated Load 2 uses the same process.
        near = random.Random(20260927).sample(list(alternatives.values()), 3)
    best = next((r for r in records if r['request_id'] == 'a77196a8365c7603'), None)
    if best is None:
        raise ValueError('缺少指定的今日Default 1 MSE 33.102153轮次；不能替换成旧实验')
    baseline = {'request_id': 'comparison-pure-mpc', 'generation': 0,
                'default1_phi_v3_assist': False,
                'tracking_mse_deg2': None,
                'controller': {'kind': 'python_class',
                               'source_code': DEFAULT_COMPENSATOR_SOURCE}}
    result = []
    for i in range(max(5, int(count))):
        item = copy.deepcopy(baseline if i == 0 else near[i-1] if i < 4 else best)
        item.update(round=i+1, comparison_replay=True, forbid_v3=(i == 0),
                    simulated_wait_after_s=random.randint(60, 80) if i < 4 else 0,
                    comparison_role='baseline_no_v3' if i == 0 else
                    'today_near_100' if i < 4 else 'selected_today_33.102153')
        if i >= 4:
            item['controller']['source_code'] = splice_round4_phi_descent(
                item['controller']['source_code'], variant=variant)
            item['comparison_role'] = 'round5_with_round4_phi_descent'
            item['request_id'] += '-phi-descent-r4'
            if variant == 2:
                item['request_id'] += '-load2'
                item['comparison_role'] += '_load2'
            # The historical score is not a measured score of the hybrid.
            item['tracking_mse_deg2'] = None
        elif variant == 2 and i > 0:
            item['comparison_role'] = 'load2_random_today_80_160'
        result.append(item)
    return result
