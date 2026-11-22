# Prepare the external pigeon reference windows

From the repository root, with NumPy 1.26.4 in the inference environment, run:

```bash
python rawprep/prepare.py --archives work/source --output work/windows --download
```

The first run fetches three ZIPs (approximately 266 MiB) directly from Zenodo and ELTE. No account is required. To work offline, place the three named ZIPs from [the data access guide](../docs/data-access.md) in the archive directory and omit `--download`. An intact local copy is reused. An incorrect download or changed archive fails the SHA-256 check. Output files appear at `work/windows/<path in data/frozen-registry.json>`.

**Huang.** Decode the checksum-valid GPS sentences with active status, retain each device's longest continuous segment at or above 5 m/s, project to the local WGS84 east–north plane, interpolate on the common 5 Hz grid, and take 26-second windows at 30-second intervals. The positions are relative to the first-frame group median. Longitude and latitude supply the two observed dimensions; altitude remains zero.

**Coordinated D1 and D2.** Start with the providers' already-smoothed positions, sample on a common 5 Hz time grid, and reject irregular files. Within each 26-second window, retain individuals with finite coordinates, median horizontal speed of at least 5 m/s, and at least two original GPS fixes in each five-sample neighbourhood. Require six eligible individuals. Select up to five evenly spaced eligible windows per recording by chronological index, and centre each selected window on the first-frame group median. The ZIP member order determines the individuals' coordinate order.

**Verification.** The program verifies the original archive checksums and independently selects 9 Huang, 75 D1 and 75 D2 windows. It checks each window's recording, number of individuals, dimensions and SHA-256 of its float32 position array (C order, time × individual × coordinate, little-endian). These position digests were taken from the original archived windows after verifying their NPZ checksums against the frozen registry. The model observes positions outside the gap. The generated NPZ also carries finite differences of the reference coordinates; model inputs do not use these derivative channels. Floating-point implementations or compression libraries can produce different NPZ bytes with identical input positions. The run records both NPZ digests and exact position matches in `preprocessing-report.json`. That report is written after all 159 inputs pass.
