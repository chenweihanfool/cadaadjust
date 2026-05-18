"""
Cadastral boundary adjustment script.
- Tolerance formula: (0.25 + 0.07 * F^(1/4)) * F^(1/2)  (F = registered area in m2)
- Strategy: move boundary segments adjacent to 9XXX (unregistered) parcels first;
  fall back to uniform scaling from centroid when no 9XXX neighbors exist.
- Outputs updated COA and PAR files; BNP topology unchanged.
"""

import os
import re
import math
import shutil
from collections import defaultdict

TARGETS = [
    {'district': '306', 'main': 199, 'sub': 2,  'folder': 'KC03060008', 'prefix': 'KC0306'},
    {'district': '337', 'main': 541, 'sub': 9,  'folder': 'KC03370012', 'prefix': 'KC0337'},
    {'district': '356', 'main': 35,  'sub': 1,  'folder': 'KC03560008', 'prefix': 'KC0356'},
    {'district': '331', 'main': 478, 'sub': 1,  'folder': 'KC03310018', 'prefix': 'KC0331'},
    {'district': '341', 'main': 168, 'sub': 3,  'folder': 'KC03410011', 'prefix': 'KC0341'},
]

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_SUFFIX = '_adjusted'

# Aim to land at 90% of tolerance from the registered area (not zero difference).
# e.g. if tol=69.32, new_diff will be ~62.4, safely within tolerance with margin.
TARGET_FRACTION = 0.9


# ── Tolerance ────────────────────────────────────────────────────────────────
def tolerance(F):
    """(0.25 + 0.07 * F^0.25) * F^0.5  where F is registered area in m2"""
    return (0.25 + 0.07 * (F ** 0.25)) * math.sqrt(F)


# ── Parse COA ────────────────────────────────────────────────────────────────
def parse_coa(filepath):
    """
    Returns: header, points {pid: {Y,X,flag}}, raw_lines, scale
    Line format (fixed-width): NNNNN YYYYYYYYYYYYYYYXXXXXXXXXXXXXXXXF
    """
    with open(filepath, 'r', encoding='big5', errors='replace') as f:
        lines = f.readlines()
    header = lines[0]
    parts = header.split()
    scale = 500
    if len(parts) >= 3:
        try:
            scale = int(parts[2])
        except ValueError:
            pass
    points = {}
    for line in lines[1:]:
        raw = line.rstrip('\n')
        if len(raw) < 6:
            continue
        try:
            pid = int(raw[:5])
        except ValueError:
            continue
        rest = raw[6:]
        if len(rest) < 31:
            continue
        try:
            Y = float(rest[:16])
            X = float(rest[16:31])
        except ValueError:
            continue
        flag = rest[31] if len(rest) > 31 else ' '
        points[pid] = {'Y': Y, 'X': X, 'flag': flag}
    return header, points, lines, scale


# ── Parse BNP ────────────────────────────────────────────────────────────────
def parse_bnp(filepath):
    """
    Returns: header, parcel_points {(main,sub): [pid,...]}, raw_lines
    Line format: MAIN  SUB  SEG  TOTAL  P1 P2 ...
    """
    with open(filepath, 'r', encoding='big5', errors='replace') as f:
        lines = f.readlines()
    header = lines[0]
    parcel_points = defaultdict(list)
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            main = int(parts[0])
            sub  = int(parts[1])
            pts  = [int(p) for p in parts[4:]]
        except ValueError:
            continue
        parcel_points[(main, sub)].extend(pts)
    return header, parcel_points, lines


# ── Parse PAR ────────────────────────────────────────────────────────────────
def parse_par(filepath):
    """
    Returns: header, parcel_info {(main,sub): {reg,dig,status,line_idx}}, raw_lines
    """
    with open(filepath, 'r', encoding='big5', errors='replace') as f:
        lines = f.readlines()
    header = lines[0]
    parcel_info = {}
    pat = re.compile(
        r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+\d+\s+\d+\s+([\d.]+)(\S)'
    )
    for idx, line in enumerate(lines[1:], 1):
        raw = line.rstrip('\n')
        if not raw.strip():
            continue
        m = pat.match(raw)
        if m:
            main   = int(m.group(1))
            sub    = int(m.group(2))
            reg    = float(m.group(4))
            dig    = float(m.group(5))
            status = m.group(6)
            parcel_info[(main, sub)] = {
                'reg': reg, 'dig': dig, 'status': status, 'line_idx': idx
            }
    return header, parcel_info, lines


# ── Build point-to-parcels index ──────────────────────────────────────────────
def build_point_index(parcel_points):
    """Returns {pid: set of (main, sub)} for all parcels in file."""
    idx = defaultdict(set)
    for key, pts in parcel_points.items():
        for pid in pts:
            idx[pid].add(key)
    return idx


# ── Area (Shoelace) ───────────────────────────────────────────────────────────
def signed_shoelace(coords):
    """Signed area using Shoelace. coords = [(Y, X), ...] Positive = CW geographic."""
    n = len(coords)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
    return a / 2.0


def shoelace_area(coords):
    return abs(signed_shoelace(coords))


def polygon_centroid(coords):
    n = len(coords)
    return sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n


# ── Outward-shift adjustment (preferred, towards 9XXX land) ───────────────────
def directed_adjustment(coords, point_ids, target_area, point_index, target_key):
    """
    For each boundary point, check if it is shared ONLY with 9XXX parcels
    (besides the target parcel itself). Those points are 'movable' — we shift
    them outward along the average unit normal of their adjacent edges.
    Binary-search the shift distance d so the new area == target_area.

    Returns: (new_coords, d_meters, max_disp_meters, mode_str)
    """
    n = len(coords)
    target_set = {target_key}

    # Determine which points are movable
    movable = []
    for pid in point_ids:
        others = point_index.get(pid, set()) - target_set
        # Movable if every other parcel sharing this point is a 9XXX parcel
        if not others or all(m >= 9000 for (m, _s) in others):
            movable.append(True)
        else:
            movable.append(False)

    n_movable = sum(movable)
    print(f'    Movable points (adj to 9XXX only): {n_movable}/{n}', end='')
    if n_movable > 0:
        mvbl_ids = [point_ids[i] for i in range(n) if movable[i]]
        print(f'  -> {mvbl_ids}')
    else:
        print()

    if n_movable == 0:
        # Fall back to uniform scaling from centroid
        new_coords, k, max_disp = scale_polygon(coords, target_area)
        return new_coords, (k - 1.0) * 10.0, max_disp, 'uniform_scale'

    # Polygon orientation: sign_A < 0 means CCW geographically
    sign_A = signed_shoelace(coords)
    orient = -1 if sign_A < 0 else 1   # +1 means CW geographically

    # Compute shift unit-vector for each movable point
    # Outward normal for CW polygon: right perpendicular = (dX/|e|, -dY/|e|) per edge
    # (Y=northing, X=easting; for CW geographic polygon the interior is to the left
    #  of traversal direction, so outward = right perp in (Y,X) space)
    # Generalized: outward_Y = orient * dX/|e|, outward_X = orient * (-dY/|e|)
    shift_vecs = []
    for i in range(n):
        if not movable[i]:
            shift_vecs.append((0.0, 0.0))
            continue
        normals = []
        for ea, eb in [((i - 1) % n, i), (i, (i + 1) % n)]:
            ya, xa = coords[ea]
            yb, xb = coords[eb]
            dy, dx = yb - ya, xb - xa
            length = math.sqrt(dy * dy + dx * dx)
            if length > 1e-10:
                ny = orient * dx / length
                nx = orient * (-dy) / length
                normals.append((ny, nx))
        if normals:
            ny = sum(v[0] for v in normals) / len(normals)
            nx = sum(v[1] for v in normals) / len(normals)
            norm = math.sqrt(ny * ny + nx * nx)
            if norm > 1e-10:
                ny, nx = ny / norm, nx / norm
            shift_vecs.append((ny, nx))
        else:
            shift_vecs.append((0.0, 0.0))

    def area_for_d(d):
        new_c = [(y + d * sv[0], x + d * sv[1])
                 for (y, x), sv in zip(coords, shift_vecs)]
        return shoelace_area(new_c)

    # Binary search for d
    current_area = shoelace_area(coords)
    if current_area < target_area:
        lo, hi = 0.0, 5.0   # expand outward up to 5m
    else:
        lo, hi = -5.0, 0.0  # contract inward up to 5m

    for _ in range(60):
        mid = (lo + hi) / 2.0
        a = area_for_d(mid)
        if (current_area < target_area and a < target_area) or \
           (current_area > target_area and a > target_area):
            lo = mid
        else:
            hi = mid

    d = (lo + hi) / 2.0
    new_coords = [(y + d * sv[0], x + d * sv[1])
                  for (y, x), sv in zip(coords, shift_vecs)]
    max_disp = abs(d)  # all movable points displaced by the same d
    return new_coords, d, max_disp, 'directed_9xxx'


# ── Uniform scale fallback ────────────────────────────────────────────────────
def scale_polygon(coords, target_area):
    """Scale all points uniformly from centroid to reach target_area."""
    current_area = shoelace_area(coords)
    if current_area == 0:
        return coords, 1.0, 0.0
    k = math.sqrt(target_area / current_area)
    cy, cx = polygon_centroid(coords)
    new_coords = []
    max_disp = 0.0
    for (y, x) in coords:
        ny = cy + k * (y - cy)
        nx = cx + k * (x - cx)
        disp = math.sqrt((ny - y) ** 2 + (nx - x) ** 2)
        max_disp = max(max_disp, disp)
        new_coords.append((ny, nx))
    return new_coords, k, max_disp


# ── Format helpers ────────────────────────────────────────────────────────────
def format_coa_line(pid, Y, X, flag):
    """NNNNN YYYYYYYYYYYYYYYXXXXXXXXXXXXXXXXF"""
    return f'{pid:5d} {Y:16.8f}{X:15.8f}{flag}\n'


def update_par_line(raw_line, new_dig_area):
    """Replace digitized area field in PAR line, keep status character."""
    pat = re.compile(
        r'(^\s*\d+\s+\d+\s+\d+\s+[\d.]+\s+\d+\s+\d+\s+)([\d.]+)(\S)(.*)$'
    )
    m = pat.match(raw_line.rstrip('\n'))
    if m:
        return f'{m.group(1)}{new_dig_area:10.2f}{m.group(3)}{m.group(4)}\n'
    return raw_line


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    results = []

    for target in TARGETS:
        district = target['district']
        main_no  = target['main']
        sub_no   = target['sub']
        folder   = target['folder']
        prefix   = target['prefix']

        folder_path = os.path.join(BASE_DIR, folder)
        coa_path = os.path.join(folder_path, f'{prefix}.COA')
        bnp_path = os.path.join(folder_path, f'{prefix}.BNP')
        par_path = os.path.join(folder_path, f'{prefix}.PAR')

        print(f'\n{"="*65}')
        print(f'Parcel: {district} district  {main_no}-{sub_no}  ({prefix})')

        coa_header, coa_points, coa_lines, scale = parse_coa(coa_path)
        bnp_header, parcel_points, bnp_lines     = parse_bnp(bnp_path)
        par_header, parcel_info, par_lines        = parse_par(par_path)
        pt_index = build_point_index(parcel_points)

        key = (main_no, sub_no)
        point_ids = parcel_points.get(key)
        if not point_ids:
            print(f'  ERROR: not found in BNP'); continue

        missing = [pid for pid in point_ids if pid not in coa_points]
        if missing:
            print(f'  ERROR: COA missing points {missing}'); continue

        coords = [(coa_points[pid]['Y'], coa_points[pid]['X']) for pid in point_ids]
        current_area = shoelace_area(coords)

        par_rec = parcel_info.get(key)
        if not par_rec:
            print(f'  ERROR: not found in PAR'); continue

        reg_area  = par_rec['reg']
        dig_area  = par_rec['dig']
        diff_orig = reg_area - dig_area
        tol       = tolerance(reg_area)

        print(f'  Scale 1:{scale}')
        print(f'  Registered area    : {reg_area:.2f} m2')
        print(f'  PAR digitized area : {dig_area:.2f} m2  (diff: {diff_orig:+.2f})')
        print(f'  Shoelace area      : {current_area:.2f} m2')
        print(f'  Legal tolerance    : {tol:.2f} m2  (formula: (0.25+0.07*F^0.25)*F^0.5)')
        exceeds = abs(diff_orig) > tol
        print(f'  Exceeds tolerance  : {"YES" if exceeds else "NO — skipping"}')
        print(f'  Boundary pts ({len(point_ids):2d}) : {point_ids}')

        if not exceeds:
            print(f'  -> No adjustment needed.')
            results.append({
                'label': f'{district}-{main_no}-{sub_no}',
                'reg': reg_area, 'diff_orig': round(diff_orig, 2),
                'diff_new': round(diff_orig, 2), 'tol': round(tol, 2),
                'max_disp_cm': 0.0, 'mode': 'skipped', 'status': 'within_tolerance',
                'point_ids': parcel_points.get(key, []), 'point_shifts': [],
            })
            continue

        # Target: land at TARGET_FRACTION * tol from registered area (not zero diff)
        target_area = reg_area - math.copysign(tol * TARGET_FRACTION, diff_orig)
        print(f'  Target area        : {target_area:.2f} m2  '
              f'(target diff = {reg_area - target_area:+.2f}, tol = {tol:.2f})')

        # Adjustment
        print(f'  Adjustment:')
        new_coords, d_val, max_disp, mode = directed_adjustment(
            coords, point_ids, target_area, pt_index, key
        )
        new_area = shoelace_area(new_coords)
        diff_new = reg_area - new_area

        print(f'    Mode             : {mode}')
        print(f'    New area         : {new_area:.4f} m2  (remaining diff: {diff_new:+.4f})')
        print(f'    Max point shift  : {max_disp*100:.1f} cm')

        status_adj = 'OK' if abs(diff_new) <= tol else 'STILL_OVER'
        print(f'    Status           : {"within tolerance" if status_adj=="OK" else "STILL EXCEEDS - review needed"}')

        # Write output
        out_folder = os.path.join(folder_path, OUTPUT_SUFFIX)
        os.makedirs(out_folder, exist_ok=True)

        for ext in ('CTL', 'DIS', 'MAP', 'RCO', 'UPN'):
            src = os.path.join(folder_path, f'{prefix}.{ext}')
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_folder, f'{prefix}.{ext}'))

        # COA: update adjusted points
        pid_to_new = dict(zip(point_ids, new_coords))
        new_coa_lines = list(coa_lines)
        for i, line in enumerate(new_coa_lines[1:], 1):
            raw = line.rstrip('\n')
            if len(raw) < 5:
                continue
            try:
                pid = int(raw[:5])
            except ValueError:
                continue
            if pid in pid_to_new:
                ny, nx = pid_to_new[pid]
                flag = coa_points[pid]['flag']
                new_coa_lines[i] = format_coa_line(pid, ny, nx, flag)
        coa_out = os.path.join(out_folder, f'{prefix}.COA')
        with open(coa_out, 'w', encoding='big5', errors='replace') as f:
            f.writelines(new_coa_lines)
        print(f'  -> COA: {coa_out}')

        # BNP: unchanged (topology not modified)
        bnp_out = os.path.join(out_folder, f'{prefix}.BNP')
        shutil.copy2(bnp_path, bnp_out)
        print(f'  -> BNP: {bnp_out}  (copied, topology unchanged)')

        # PAR: update digitized area on target parcel line
        new_par_lines = list(par_lines)
        new_par_lines[par_rec['line_idx']] = update_par_line(
            par_lines[par_rec['line_idx']], new_area
        )
        par_out = os.path.join(out_folder, f'{prefix}.PAR')
        with open(par_out, 'w', encoding='big5', errors='replace') as f:
            f.writelines(new_par_lines)
        print(f'  -> PAR: {par_out}')

        # Record results
        # Per-point shift details
        point_shifts = []
        for i, (pid, (oy, ox), (ny, nx)) in enumerate(
            zip(point_ids, coords, new_coords)
        ):
            dy = ny - oy
            dx = nx - ox
            disp = math.sqrt(dy*dy + dx*dx)
            point_shifts.append({
                'pid': pid,
                'dY_cm': round(dy*100, 1),
                'dX_cm': round(dx*100, 1),
                'dist_cm': round(disp*100, 1),
                'movable': mode != 'uniform_scale' and
                           (point_ids[i] in [point_ids[j] for j in range(len(point_ids))
                                             if True])  # placeholder; computed above
            })

        results.append({
            'label': f'{district}-{main_no}-{sub_no}',
            'reg': reg_area,
            'diff_orig': round(diff_orig, 2),
            'diff_new': round(diff_new, 4),
            'tol': round(tol, 2),
            'max_disp_cm': round(max_disp * 100, 1),
            'mode': mode,
            'status': status_adj,
            'point_ids': point_ids,
            'point_shifts': point_shifts,
        })

    # Summary
    print(f'\n{"="*65}')
    print('SUMMARY')
    hdr = f'{"Parcel":<20} {"Reg":>8} {"OldDiff":>9} {"NewDiff":>10} {"Tol":>8} {"MaxShift":>10}  Mode/Status'
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        print(
            f'{r["label"]:<20} {r["reg"]:>8.2f} '
            f'{r["diff_orig"]:>+9.2f} {r["diff_new"]:>+10.4f} '
            f'{r["tol"]:>8.2f} {r["max_disp_cm"]:>8.1f} cm  '
            f'{r["mode"]} / {r["status"]}'
        )

    # Per-point shift detail
    print(f'\nPER-POINT SHIFTS:')
    for r in results:
        print(f'  {r["label"]}:')
        for ps in r['point_shifts']:
            print(f'    pt {ps["pid"]:5d}: dY={ps["dY_cm"]:+7.1f}cm  dX={ps["dX_cm"]:+7.1f}cm  |shift|={ps["dist_cm"]:6.1f}cm')

    print('\nDone. Adjusted files in _adjusted/ subfolder of each district folder.')


if __name__ == '__main__':
    main()
