"""Convert the three original public pigeon releases to 26-second reference windows."""

import io
import re
from collections import defaultdict
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np


def coordinate_window(position):
    velocity = np.gradient(position, 0.2, axis=0)
    acceleration = np.gradient(velocity, 0.2, axis=0)
    return np.concatenate([position, velocity, acceleration], axis=-1).astype(np.float32)


def local_horizontal(longitude, latitude):
    longitude, latitude = np.deg2rad(longitude), np.deg2rad(latitude)
    lon0, lat0 = np.deg2rad([113.522873, 34.808742])
    eccentricity2 = 6.6943799901413165e-3

    def surface_xyz(lon, lat):
        radius = 6378137 / np.sqrt(1 - eccentricity2 * np.sin(lat) ** 2)
        return np.stack([radius * np.cos(lat) * np.cos(lon),
                         radius * np.cos(lat) * np.sin(lon),
                         radius * (1 - eccentricity2) * np.sin(lat)], axis=-1)

    offset = surface_xyz(longitude, latitude) - surface_xyz(lon0, lat0)
    east = np.array([-np.sin(lon0), np.cos(lon0), 0])
    north = np.array([-np.sin(lat0) * np.cos(lon0),
                      -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    return offset @ east, offset @ north


def degrees(value):
    value = float(value)
    return value // 100 + (value % 100) / 60


def gps_segment(contents):
    rows, dates = [], set()
    for line in contents.decode('utf-8', errors='replace').splitlines():
        line = line.replace('#', '').strip()
        if not line.startswith('$GNRMC,'):
            continue
        try:
            payload, end = line[1:].split('*', 1)
            checksum = 0
            for char in payload:
                checksum ^= ord(char)
            fields = payload.split(',')
            if checksum != int(end[:2], 16) or fields[2] != 'A':
                continue
            h = float(fields[1])
            time = (h // 10000) * 3600 + ((h % 10000) // 100) * 60 + h % 100
            latitude = degrees(fields[3]) * (-1 if fields[4] == 'S' else 1)
            longitude = degrees(fields[5]) * (-1 if fields[6] == 'W' else 1)
            row = [time, longitude, latitude, float(fields[7]) * 1.852 / 3.6]
            if np.isfinite(row).all():
                rows.append(row)
                dates.add(fields[9])
        except (ValueError, IndexError):
            continue
    if len(dates) != 1:
        raise ValueError('Expected valid GPS sentences from one UTC date')
    rows = np.array(rows, dtype=float)
    _, first = np.unique(np.round(rows[:, 0], 4), return_index=True)
    rows = rows[np.sort(first)]
    rows = rows[np.argsort(rows[:, 0])]
    valid = np.flatnonzero(rows[:, 3] >= 5)
    cuts = (np.diff(valid) > 1) | (np.diff(rows[valid, 0]) > 2.0001)
    segments = [s for s in np.split(valid, np.flatnonzero(cuts) + 1) if len(s) > 1]
    index = max(segments, key=lambda s: rows[s[-1], 0] - rows[s[0], 0])
    return rows[index], dates.pop()


def huang_windows(path):
    with ZipFile(path) as archive:
        for trial_no in range(1, 4):
            record = f'analysis_trial_{trial_no:02d}'
            names = sorted(n for n in archive.namelist()
                           if f'/{record}/raw_gps/' in n and n.endswith('_GPS.txt'))
            segments = [gps_segment(archive.read(n)) for n in names]
            if len({date for _, date in segments}) != 1:
                raise ValueError(f'{record}: device dates differ')
            tracks = [s for s, _ in segments]
            start = np.ceil(max(d[0, 0] for d in tracks) * 5) / 5
            end = np.floor(min(d[-1, 0] for d in tracks) * 5) / 5
            if len(tracks) < 6 or end - start < 26:
                continue
            grid = np.round(np.arange(start, end + .001, .2), 4)
            positions = []
            for track in tracks:
                x, y = local_horizontal(track[:, 1], track[:, 2])
                positions.append(np.stack([np.interp(grid, track[:, 0], x),
                                           np.interp(grid, track[:, 0], y),
                                           np.zeros(len(grid))], axis=1))
            positions = np.stack(positions, axis=1)
            positions[:, :, :2] -= np.median(positions[0, :, :2], axis=0)
            for i, offset in enumerate(range(0, len(grid) - 130, 150)):
                name = f'{record}-window{i:03d}.npz'
                yield record, name, coordinate_window(positions[offset:offset + 131])


def elte_tracks(path, source):
    records = defaultdict(list)
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if source == 'D1':
                if not re.search(r'/(ff|hf)\d+/[^/]+\.txt$', name):
                    continue
            elif not re.fullmatch(r'flight/ff[ABC]\d+/ff[ABC]\d+_p[ABC]\d+', name):
                raise ValueError(f'Unexpected D2 member: {name}')
            text = archive.read(name).decode('latin1')
            if source == 'D1':
                text = re.sub(r'(GPS signal)(\d+\t)', r'\1\n\2', text)
            data = np.loadtxt(io.StringIO(text), comments='#', ndmin=2)
            if data.shape[1] != 11:
                raise ValueError(f'Unexpected columns in {name}')
            if not np.isfinite(data[:, [0, 10]]).all():
                if source == 'D1':
                    raise ValueError(f'Nonfinite timestamps or fix flags in {name}')
                continue
            if source == 'D1':
                ticks = (20 * np.rint(data[:, 0] / 20)).astype(np.int64)
                if np.max(abs(ticks - data[:, 0])) > 1.001:
                    continue
            else:
                if not np.all(np.diff(data[:, 0]) == 10):
                    continue
                data = data[data[:, 0] % 20 == 0]
                ticks = data[:, 0].astype(np.int64)
            if not np.all(np.diff(ticks) == 20):
                continue
            record = PurePosixPath(name).parent.name
            records[record].append((ticks, data[:, 1:4], data[:, 10] > .5))
    return records


def elte_windows(path, source):
    for record, tracks in elte_tracks(path, source).items():
        starts = [ticks[0] for ticks, _, _ in tracks]
        if len({int(t) % 20 for t in starts}) != 1:
            raise ValueError(f'{record}: unsynchronized clocks')
        begin = int(min(starts))
        end = int(max(ticks[-1] for ticks, _, _ in tracks))
        eligible = []
        for start in range(begin, end - 2600 + 1, 3000):
            positions = []
            for ticks, coords, fixes in tracks:
                if start < ticks[0] or start + 2600 > ticks[-1]:
                    continue
                offset = (start - int(ticks[0])) // 20
                p, gps = coords[offset:offset + 131], fixes[offset:offset + 131]
                if len(p) != 131 or not np.isfinite(p).all():
                    continue
                speed = np.linalg.norm(np.gradient(p, .2, axis=0)[:, :2], axis=1)
                if np.median(speed) < 5:
                    continue
                # Two fixes in every five samples also excludes an interpolated
                # run longer than the registered one-second maximum.
                if np.any(np.convolve(gps.astype(float), np.ones(5), mode='valid') < 2):
                    continue
                positions.append(p)
            if len(positions) >= 6:
                eligible.append((start, positions))
        if not eligible:
            continue
        indices = np.unique(np.rint(np.linspace(0, len(eligible) - 1,
                                                min(5, len(eligible)))).astype(int))
        for idx in indices:
            start, positions = eligible[idx]
            p = np.stack(positions, axis=1)
            p -= np.median(p[0], axis=0)
            yield record, f'{source}-{record}-{start}.npz', coordinate_window(p)
