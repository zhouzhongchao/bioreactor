# -*- coding: utf-8 -*-
"""Pressure-feasible static reachability for the V6 ten-PAM model.

This module is deliberately offline: it neither opens the serial port nor sends
valve commands.  A pose is called reachable when an antagonist-exclusive set of
six channel pressures in [0, pressure_limit] can balance the V6 static torque.
"""
from __future__ import annotations

import itertools
import base64
import json
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


ANTAGONIST_PAIRS = ((0, 1), (2, 3), (4, 5))
DEFAULT_SWING_X_MIN_DEG = -20.0
DEFAULT_SWING_X_MAX_DEG = 20.0
DEFAULT_SWING_Y_MIN_DEG = -38.0
DEFAULT_SWING_Y_MAX_DEG = 80.0
DEFAULT_TWIST_MIN_DEG = -70.0
DEFAULT_TWIST_MAX_DEG = 75.0


def _static_torque(st, plant, q, pressure_kpa):
    force = plant.forces(q, pressure_kpa)
    return (-st.spm.J_L(q).T @ force - st.spm.G_vector(q, st.ARM)
            - st.Plant.KSIL @ q + plant.tau0)


def solve_pose_pressures(st, plant, q, pressure_limit_kpa=200.0,
                         torque_tolerance_nm=0.01):
    """Return the lowest-pressure feasible antagonist-exclusive static solution."""
    best = None
    patterns = (
        np.asarray([pair[side] for pair, side in zip(ANTAGONIST_PAIRS, sides)], dtype=int)
        for sides in itertools.product((0, 1), repeat=3)
    )
    for active in patterns:
        def residual(active_pressure):
            pressure = np.zeros(6, dtype=float)
            pressure[active] = active_pressure
            return _static_torque(st, plant, q, pressure)

        solution = least_squares(
            residual, np.full(3, min(60.0, 0.3 * pressure_limit_kpa)),
            bounds=(0.0, float(pressure_limit_kpa)), max_nfev=80,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        torque_error = float(np.linalg.norm(solution.fun))
        pressure = np.zeros(6, dtype=float)
        pressure[active] = solution.x
        # First prefer equilibrium accuracy, then the solution using less peak pressure.
        score = (torque_error > torque_tolerance_nm, torque_error,
                 float(np.max(pressure)), float(np.sum(pressure)))
        if best is None or score < best[0]:
            best = (score, pressure, torque_error)
    if best is None or best[2] > torque_tolerance_nm:
        return None
    return best[1], best[2]


def _sample_directional_swing_poses(rng, count, swing_x_min_deg,
                                    swing_x_max_deg, swing_y_min_deg,
                                    swing_y_max_deg, twist_min_deg,
                                    twist_max_deg):
    """Sample the closed, directional swing envelope and a free twist angle.

    Each quadrant is a quarter ellipse whose two radii are the corresponding
    measured positive/negative X and Y limits.  Quadrants are selected in
    proportion to their area, so the complete asymmetric envelope is filled
    without inventing the unreachable corners of a rectangular angle box.
    """
    x_neg, x_pos = abs(float(swing_x_min_deg)), float(swing_x_max_deg)
    y_neg, y_pos = abs(float(swing_y_min_deg)), float(swing_y_max_deg)
    if min(x_neg, x_pos, y_neg, y_pos) <= 0.0:
        raise ValueError("directional swing limits must straddle zero")
    radii = np.asarray([
        [x_pos, y_pos], [x_neg, y_pos],
        [x_neg, y_neg], [x_pos, y_neg],
    ])
    probabilities = np.prod(radii, axis=1)
    probabilities /= np.sum(probabilities)
    quadrant = rng.choice(4, size=int(count), p=probabilities)
    local_angle = rng.uniform(0.0, 0.5*np.pi, int(count))
    radius = np.sqrt(rng.uniform(0.0, 1.0, int(count)))
    signs = np.asarray([[1.0, 1.0], [-1.0, 1.0],
                        [-1.0, -1.0], [1.0, -1.0]])
    scale = radii[quadrant]
    direction = np.column_stack((np.cos(local_angle), np.sin(local_angle)))
    swing = radius[:, None] * direction * scale * signs[quadrant]
    twist = rng.uniform(float(twist_min_deg), float(twist_max_deg), int(count))
    return np.column_stack((swing, twist))


def calculate_theoretical_map(sample_count, seed, pressure_limit_kpa=200.0,
                              swing_x_min_deg=DEFAULT_SWING_X_MIN_DEG,
                              swing_x_max_deg=DEFAULT_SWING_X_MAX_DEG,
                              swing_y_min_deg=DEFAULT_SWING_Y_MIN_DEG,
                              swing_y_max_deg=DEFAULT_SWING_Y_MAX_DEG,
                              twist_min_deg=DEFAULT_TWIST_MIN_DEG,
                              twist_max_deg=DEFAULT_TWIST_MAX_DEG,
                              torque_tolerance_nm=0.01,
                              progress: Callable[[int, int], None] | None = None):
    """Monte-Carlo inverse-static map inside the measured swing envelope."""
    import mpc_station as st

    params_path = os.path.join(os.path.dirname(__file__), "params.json")
    try:
        with open(params_path, encoding="utf-8") as stream:
            params = json.load(stream)
    except (FileNotFoundError, ValueError, TypeError):
        params = dict(st.DEF_P)
    plant = st.Plant(params)
    count = max(1, int(sample_count))
    rng = np.random.default_rng(int(seed))
    swing_x_min = float(swing_x_min_deg)
    swing_x_max = float(swing_x_max_deg)
    swing_y_min = float(swing_y_min_deg)
    swing_y_max = float(swing_y_max_deg)
    if not (swing_x_min < 0.0 < swing_x_max and
            swing_y_min < 0.0 < swing_y_max):
        raise ValueError("X and Y directional swing limits must straddle zero")
    twist_min = float(twist_min_deg)
    twist_max = float(twist_max_deg)
    if twist_min > twist_max:
        raise ValueError("twist_min_deg must not exceed twist_max_deg")
    candidates = _sample_directional_swing_poses(
        rng, count, swing_x_min, swing_x_max, swing_y_min, swing_y_max,
        twist_min, twist_max)
    results = []

    # The unloaded origin is known exactly because Plant.tau0 defines it as equilibrium.
    results.append({
        "pose_deg": [0.0, 0.0, 0.0], "pressure_kpa": [0.0] * 6,
        "max_pressure_kpa": 0.0, "torque_error_nm": 0.0,
    })
    for index, pose in enumerate(candidates):
        q = np.asarray(st.q_model_from_st(*pose), dtype=float)
        solved = solve_pose_pressures(
            st, plant, q, pressure_limit_kpa=pressure_limit_kpa,
            torque_tolerance_nm=torque_tolerance_nm)
        if solved is not None:
            pressure, torque_error = solved
            # Numerical solutions can contain tiny simultaneous values.  Make the
            # hard antagonist constraint explicit in the saved result.
            pressure[np.abs(pressure) < 1e-7] = 0.0
            results.append({
                "pose_deg": [float(v) for v in pose],
                "pressure_kpa": [float(v) for v in pressure],
                "max_pressure_kpa": float(np.max(pressure)),
                "torque_error_nm": float(torque_error),
            })
        if progress is not None and ((index + 1) % 50 == 0 or index + 1 == count):
            progress(index + 1, count)

    return {
        "schema": 1,
        "model_version": "V6 ten-PAM static model",
        "definition": "inverse-static pressure-feasible poses",
        "sample_seed": int(seed),
        "candidate_pose_count": count,
        "reachable_pose_count": len(results),
        "pressure_limit_kpa": float(pressure_limit_kpa),
        "pressure_lower_bound_kpa": 0.0,
        "unlock_threshold_used": False,
        "antagonist_hard_constraint": [list(pair) for pair in ANTAGONIST_PAIRS],
        "pose_sampling": "user-specified directional angular envelope with independent twist",
        "range_source": "explicit user angular limits; not inferred from displacement logs",
        "swing_x_range_deg": [swing_x_min, swing_x_max],
        "swing_y_range_deg": [swing_y_min, swing_y_max],
        "twist_range_deg": [twist_min, twist_max],
        "torque_tolerance_nm": float(torque_tolerance_nm),
        "parameter_source": os.path.abspath(params_path),
        "results": results,
    }


CAD_REFERENCE_SWING_Y_DEG = 0.0
# The embedded moving-part mesh is already tilted 12.6 degrees from the
# model's hanging -z_B axis in its bind pose (distal centre ~= [-48.4,0,-215.9]).
CAD_BIND_SWING_Y_DEG = 12.6


def _rotate_y_points(points, angle_deg):
    """Rotate points about the spherical-joint centre and the +y_B axis."""
    xyz = np.asarray(points, dtype=float)
    angle = np.radians(float(angle_deg))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, 0.0, sine),
                         (0.0, 1.0, 0.0),
                         (-sine, 0.0, cosine)))
    return xyz @ rotation.T


def _gui_cad_meshes(reference_swing_y_deg=None):
    """Decode the exact CAD mesh embedded in the GUI.

    Do not independently subsample triangles here.  That leaves disconnected
    facets and makes the mechanism look broken in a still image.  The embedded
    asset is small enough to render as complete surfaces offline.
    """
    html_path = Path(__file__).resolve().parent.parent / "pam_shoulder_gui.html"
    source = html_path.read_text(encoding="utf-8")
    meta_match = re.search(r"const MESH_META=(\{.*?\});", source)
    data_match = re.search(r'const MESH_B64="([A-Za-z0-9+/=]+)";', source)
    if not meta_match or not data_match:
        raise ValueError("GUI中没有找到嵌入式CAD网格")
    meta = json.loads(meta_match.group(1))
    raw = base64.b64decode(data_match.group(1))
    offset = 0
    origin = np.asarray(meta["lo"], dtype=float)
    origin[0] -= 66.0  # Same PIVOT_DX used by the GUI and V6 geometry.
    meshes = []
    for name in meta["order"]:
        group = meta["groups"][name]
        nv, nf = int(group["nv"]), int(group["nf"])
        position = np.frombuffer(raw, dtype="<u2", count=nv*3, offset=offset).reshape(-1, 3)
        offset += nv * 6
        color = np.frombuffer(raw, dtype=np.uint8, count=nv*3, offset=offset).reshape(-1, 3)
        offset += nv * 3
        faces = np.frombuffer(raw, dtype="<u4", count=nf*3, offset=offset).reshape(-1, 3)
        offset += nf * 12
        if name == "cover":
            continue  # The transparent cover obscures the mechanism in a static figure.
        xyz = position.astype(float) * float(meta["q"]) + origin
        # Show the carried parts at the initial hanging pose (Y=0 degrees).
        # The barrel and rack stay fixed.
        if reference_swing_y_deg is not None and name in ("moving", "ring"):
            xyz = _rotate_y_points(xyz, reference_swing_y_deg)
        face_color = np.mean(color[faces], axis=1) / 255.0
        meshes.append((name, xyz, faces, face_color))
    return meshes


def _tip_positions_mm(blob, arm_length_mm=216.0):
    import shoulder_pam_model_v6 as v6

    points = []
    for row in blob.get("results", []):
        try:
            pose = np.asarray(row["pose_deg"], dtype=float)
            q = v6.q_from_gui_degrees(*pose)
        except (KeyError, TypeError, ValueError):
            continue
        points.append(v6.rotation_matrix(q) @ np.array([0.0, 0.0, -arm_length_mm]))
    if not points:
        raise ValueError("没有可绘制的理论臂端点")
    return np.asarray(points, dtype=float)


def measured_extrema_calibration(measurement_blob):
    """Build the latest empirical X/Y swing polygon from valid extreme cases."""
    from scipy.spatial import ConvexHull

    poses = []
    for row in measurement_blob.get("results", []):
        if not row.get("valid"):
            continue
        try:
            pose = np.asarray(row["pose_deg"], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(pose)):
            poses.append(pose)
    if len(poses) < 3:
        raise ValueError("至少需要3个有效实机极值姿态才能建立标定包络")
    pose = np.asarray(poses, dtype=float)
    swing = np.unique(np.vstack((pose[:, :2], np.zeros((1, 2)))), axis=0)
    if len(swing) < 3 or np.linalg.matrix_rank(swing - np.mean(swing, axis=0)) < 2:
        raise ValueError("实机极值姿态不能形成二维X/Y包络")
    hull = ConvexHull(swing)
    polygon = swing[hull.vertices]
    fixed_mv = float(measurement_blob.get("extrema_fixed_command_mv", 2200.0))
    return {
        "method": (f"convex hull of valid {fixed_mv:.0f}mV single/pair measured "
                   "X/Y extrema plus origin"),
        "source_created": measurement_blob.get("created"),
        "source_seed": measurement_blob.get("seed"),
        "valid_extreme_count": int(len(pose)),
        "swing_polygon_deg": np.round(polygon, 9).tolist(),
        "swing_x_range_deg": [float(np.min(pose[:, 0])), float(np.max(pose[:, 0]))],
        "swing_y_range_deg": [float(np.min(pose[:, 1])), float(np.max(pose[:, 1]))],
        "twist_range_deg": [float(np.min(pose[:, 2])), float(np.max(pose[:, 2]))],
    }


def _sample_convex_swing_polygon(rng, polygon_deg, count, boundary_count=360):
    """Uniformly sample an ordered convex polygon by area-weighted triangulation."""
    from scipy.spatial import ConvexHull

    polygon = np.asarray(polygon_deg, dtype=float)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.all(np.isfinite(polygon))):
        raise ValueError("swing_polygon_deg必须是至少3点的有限二维多边形")
    polygon = polygon[ConvexHull(polygon).vertices]
    triangles = np.asarray([
        [polygon[0], polygon[index], polygon[index + 1]]
        for index in range(1, len(polygon) - 1)
    ])
    cross = ((triangles[:, 1, 0] - triangles[:, 0, 0]) *
             (triangles[:, 2, 1] - triangles[:, 0, 1]) -
             (triangles[:, 1, 1] - triangles[:, 0, 1]) *
             (triangles[:, 2, 0] - triangles[:, 0, 0]))
    areas = 0.5 * np.abs(cross)
    if float(np.sum(areas)) <= 0.0:
        raise ValueError("swing_polygon_deg面积为0")
    chosen = rng.choice(len(triangles), size=int(count), p=areas / np.sum(areas))
    tri = triangles[chosen]
    radius = np.sqrt(rng.uniform(0.0, 1.0, int(count)))
    blend = rng.uniform(0.0, 1.0, int(count))
    interior = ((1.0 - radius)[:, None] * tri[:, 0] +
                (radius * (1.0 - blend))[:, None] * tri[:, 1] +
                (radius * blend)[:, None] * tri[:, 2])

    edge_next = np.roll(polygon, -1, axis=0)
    edge_length = np.linalg.norm(edge_next - polygon, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_length)))
    distance = np.linspace(0.0, cumulative[-1], int(boundary_count), endpoint=False)
    edge_index = np.searchsorted(cumulative[1:], distance, side="right")
    local = ((distance - cumulative[edge_index]) /
             np.maximum(edge_length[edge_index], 1.0e-12))
    boundary = (polygon[edge_index] + local[:, None] *
                (edge_next[edge_index] - polygon[edge_index]))
    return np.vstack((interior, boundary))


def _dense_endpoint_envelope_mm(blob, point_count=4500, arm_length_mm=216.0):
    """Return a deterministic dotted cloud covering the measured angle envelope.

    The inverse-static results in ``blob['results']`` are validation samples,
    not a rasterization of the continuous workspace.  The endpoint does not
    depend on twist, so the full Cartesian envelope is generated directly from
    the measured directional swing limits.  A fixed random seed gives the
    scattered-dot appearance of the original plot while remaining reproducible.
    """
    try:
        x_min, x_max = (float(v) for v in blob["swing_x_range_deg"])
        y_min, y_max = (float(v) for v in blob["swing_y_range_deg"])
    except (KeyError, TypeError, ValueError):
        return _tip_positions_mm(blob, arm_length_mm=arm_length_mm)
    if not (x_min < 0.0 < x_max and y_min < 0.0 < y_max):
        raise ValueError("directional swing ranges must straddle zero")

    count = max(1000, int(point_count))
    seed = int(blob.get("sample_seed", 20260910)) & 0xffffffff
    polygon = blob.get("swing_polygon_deg")
    if polygon is not None:
        sampled_swing = _sample_convex_swing_polygon(
            np.random.default_rng(seed), polygon, count, boundary_count=360)
        sx_deg, sy_deg = sampled_swing[:, 0], sampled_swing[:, 1]
    else:
        sampled = _sample_directional_swing_poses(
            np.random.default_rng(seed), count,
            x_min, x_max, y_min, y_max, 0.0, 0.0)
        sx_deg, sy_deg = sampled[:, 0], sampled[:, 1]

        # Keep the outline dotted as well.  Thousands of large, overlapping points
        # made the previous version look like one solid painted surface.
        boundary_azimuth = np.linspace(0.0, 2.0*np.pi, 360, endpoint=False)
        boundary_cosine = np.cos(boundary_azimuth)
        boundary_sine = np.sin(boundary_azimuth)
        sx_deg = np.concatenate((
            sx_deg,
            boundary_cosine * np.where(boundary_cosine >= 0.0, x_max, abs(x_min)),
        ))
        sy_deg = np.concatenate((
            sy_deg,
            boundary_sine * np.where(boundary_sine >= 0.0, y_max, abs(y_min)),
        ))

    sx = np.radians(sx_deg)
    sy = np.radians(sy_deg)
    theta = np.hypot(sx, sy)
    sinc = np.sinc(theta / np.pi)
    physical_points = np.column_stack((
        -arm_length_mm * sinc * sy,
        arm_length_mm * sinc * sx,
        -arm_length_mm * np.cos(theta),
    ))
    return physical_points


def _save_workspace_plot(blob, path, measured_results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tip = _dense_endpoint_envelope_mm(blob)
    fig = plt.figure(figsize=(13.0, 7.5), facecolor="#f4f7f9")
    ax3 = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax3.set_facecolor("#f4f7f9")
    ax3.set_proj_type("ortho")
    meshes = _gui_cad_meshes(
        reference_swing_y_deg=(CAD_REFERENCE_SWING_Y_DEG - CAD_BIND_SWING_Y_DEG))
    # Draw the supporting rack first and make it only a faint spatial
    # reference.  Its many overlapping surfaces must not hide the mechanism.
    for name, xyz, faces, face_color in sorted(
            meshes, key=lambda item: item[0] not in ("frame", "column")):
        triangles = xyz[faces]
        is_rack = name in ("frame", "column")
        if is_rack:
            face_color = np.full((len(faces), 3), (0.72, 0.75, 0.78))
        collection = Poly3DCollection(
            triangles, facecolors=face_color, linewidths=0.0,
            alpha=(0.055 if is_rack else (0.72 if name == "silicone" else 0.96)),
            zorder=1 if is_rack else 4)
        collection.set_edgecolor("none")
        ax3.add_collection3d(collection)

    # A single colour encodes membership only; it deliberately carries no
    # pressure, force, or ordering information.
    reachability_points = ax3.scatter(
        tip[:, 0], tip[:, 1], tip[:, 2], s=5.0, c="#20c997",
        edgecolors="none", linewidths=0.0, alpha=0.72,
        depthshade=False, zorder=3, rasterized=True)

    measured_rows = []
    for row in measured_results or []:
        if not row.get("valid") or row.get("kind") == "origin":
            continue
        try:
            point = np.asarray(row["endpoint_mm"], dtype=float).reshape(3)
            pose = np.asarray(row["pose_deg"], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(point)) and np.all(np.isfinite(pose)):
            measured_rows.append((pose, point))
    measured_tip = []
    if measured_rows:
        pose = np.asarray([row[0] for row in measured_rows], dtype=float)
        points = np.asarray([row[1] for row in measured_rows], dtype=float)
        # Show only the four measured directional limits.  The other fourteen
        # cases define the green calibrated hull but do not need separate
        # markers in the publication-style range figure.
        selected = []
        for index in (np.argmin(pose[:, 0]), np.argmax(pose[:, 0]),
                      np.argmin(pose[:, 1]), np.argmax(pose[:, 1])):
            if int(index) not in selected:
                selected.append(int(index))
        points = points[selected]
        measured_tip.append(points)
        ax3.scatter(
            points[:, 0], points[:, 1], points[:, 2], s=62.0,
            c="#e03131", marker="o", edgecolors="white", linewidths=0.9,
            alpha=0.98, depthshade=False, zorder=7,
            label="four measured directional limits")

    all_parts = [tip, np.array([[-90.0, -112.0, -220.0],
                                [420.0, 112.0, 110.0]])]
    all_parts.extend(measured_tip)
    all_xyz = np.vstack(all_parts)
    lo, hi = np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)
    span = hi - lo
    pad = np.maximum(8.0, np.array([0.035, 0.08, 0.06]) * span)
    ax3.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax3.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax3.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax3.set_box_aspect(tuple(span + 2.0*pad))
    # GUI-like side view in the physical installation frame.  Zero hangs down;
    # Y+ raises the arm toward horizontal and Y- approaches the base.
    ax3.view_init(elev=12.0, azim=70.0)
    ax3.set_axis_off()
    ax3.set_position([0.01, 0.055, 0.98, 0.94])
    x_range = blob.get("swing_x_range_deg", [DEFAULT_SWING_X_MIN_DEG,
                                               DEFAULT_SWING_X_MAX_DEG])
    y_range = blob.get("swing_y_range_deg", [DEFAULT_SWING_Y_MIN_DEG,
                                               DEFAULT_SWING_Y_MAX_DEG])
    phi_range = blob.get("twist_range_deg", [DEFAULT_TWIST_MIN_DEG,
                                               DEFAULT_TWIST_MAX_DEG])
    green_label = str(blob.get("envelope_label", "theoretical envelope"))
    measured_caption = (f"   Green: {green_label}; red: four directional limits"
                        if measured_tip else "")
    caption = fig.text(0.5, 0.018,
             (f"Angular envelope:  X [{float(x_range[0]):.0f}°, {float(x_range[1]):.0f}°]   "
              f"Y [{float(y_range[0]):.0f}°, {float(y_range[1]):.0f}°]   "
              f"φ [{float(phi_range[0]):.0f}°, {float(phi_range[1]):.0f}°]"
              f"{measured_caption}"),
             ha="center", color="#495057", fontsize=10)
    fig.savefig(path, dpi=180)

    # Reuse this one heavy CAD scene and only move the camera.  This avoids
    # duplicating hundreds of thousands of mesh faces in four Matplotlib axes.
    envelope_text = caption.get_text()
    caption.set_visible(False)
    fig.set_size_inches(7.0, 4.2)
    fig.set_dpi(150)
    reachability_points.set_sizes(np.full(len(tip), 2.0))
    reachability_points.set_alpha(0.68)
    ax3.set_position([0.0, 0.0, 1.0, 1.0])
    views = (
        ("Oblique", 12.0, 70.0),
        ("Side (+y_B)", 0.0, 90.0),
        ("Front (+x_B)", 0.0, 0.0),
        ("Top (+z_B)", 90.0, -90.0),
    )
    images = []
    background_rgb = np.array([244, 247, 249], dtype=np.int16)
    for _label, elevation, azimuth in views:
        ax3.view_init(elev=elevation, azim=azimuth)
        fig.canvas.draw()
        rendered = np.asarray(fig.canvas.buffer_rgba()).copy()
        difference = np.max(
            np.abs(rendered[:, :, :3].astype(np.int16) - background_rgb), axis=2)
        yy, xx = np.nonzero(difference > 5)
        if len(xx):
            margin = 18
            x0, x1 = max(0, int(xx.min()) - margin), min(rendered.shape[1], int(xx.max()) + margin + 1)
            y0, y1 = max(0, int(yy.min()) - margin), min(rendered.shape[0], int(yy.max()) + margin + 1)
            rendered = rendered[y0:y1, x0:x1]
        images.append(rendered)
    plt.close(fig)

    output_path = Path(path)
    multiview_path = output_path.with_name(
        output_path.stem + "_multiview" + output_path.suffix)
    panel_fig, panels = plt.subplots(2, 2, figsize=(14.0, 8.6),
                                    facecolor="#f4f7f9")
    for panel, (label, _elevation, _azimuth), image in zip(panels.flat, views, images):
        panel.imshow(image)
        panel.set_title(label, color="#343a40", fontsize=11, pad=3)
        panel.set_axis_off()
    panel_fig.subplots_adjust(left=0.006, right=0.994, top=0.965,
                              bottom=0.065, wspace=0.018, hspace=0.055)
    panel_fig.text(0.5, 0.022, envelope_text,
                   ha="center", color="#495057", fontsize=11)
    panel_fig.savefig(multiview_path, dpi=150,
                      facecolor=panel_fig.get_facecolor())
    plt.close(panel_fig)
    return str(multiview_path)


def save_theoretical_plot(blob, path):
    """Render the theoretical angular envelope with the GUI's embedded CAD."""
    return _save_workspace_plot(blob, path)


def save_measured_extrema_plot(measurement_blob, path):
    """Overlay extrema on a dense envelope calibrated from those real measurements."""
    calibration = (measurement_blob.get("calibrated_envelope") or
                   measured_extrema_calibration(measurement_blob))
    theoretical_blob = {
        "sample_seed": int(measurement_blob.get("seed", 20260910)),
        "swing_x_range_deg": calibration["swing_x_range_deg"],
        "swing_y_range_deg": calibration["swing_y_range_deg"],
        "twist_range_deg": calibration["twist_range_deg"],
        "swing_polygon_deg": calibration["swing_polygon_deg"],
        "envelope_label": (f"latest {float(measurement_blob.get('extrema_fixed_command_mv', 2200.0)):.0f}mV "
                           "measured convex ROM"),
        "results": [{"pose_deg": [0.0, 0.0, 0.0]}],
    }
    return _save_workspace_plot(
        theoretical_blob, path, measured_results=measurement_blob.get("results", []))
