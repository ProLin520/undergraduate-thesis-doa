import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
REPO_ROOT = PROJECT_ROOT.parent

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

for path in [REPO_ROOT, REPO_ROOT / "doatools.py-master", PROJECT_ROOT / "external" / "doatools"]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import doatools.estimation as estimation
import doatools.model as model
import doatools.performance as perf

from Graduation.data.data_create.simulation_generator import generate_dca1000_style_data, generate_ideal_data
from Graduation.utils.metrics_utils import TRADITION_DATA_DIR, first_x_reach_threshold, nearest_value, save_csv
from Graduation.utils.radar_utils import NUM_CHIRPS, NUM_FRAMES, process_radar_data


WAVELENGTH = 1.0
D0 = 0.5
NUM_RX = 8
GRID_SIZE = 1801
L_SUBARRAYS = 2
METHOD_COLUMNS = ["MUSIC", "RootMUSIC", "ESPRIT", "SSMUSIC"]


def create_estimators(num_rx=NUM_RX):
    grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=GRID_SIZE)
    ula = model.UniformLinearArray(num_rx, D0)
    ula_ss = model.UniformLinearArray(num_rx - L_SUBARRAYS + 1, D0)
    estimators = {
        "MUSIC": estimation.MUSIC(ula, WAVELENGTH, grid),
        "RootMUSIC": estimation.RootMUSIC1D(WAVELENGTH),
        "ESPRIT": estimation.Esprit1D(WAVELENGTH),
        "SSMUSIC": estimation.MUSIC(ula_ss, WAVELENGTH, grid),
    }
    return ula, estimators


def estimate_locations(estimators, r_matrix, num_sources):
    results = {}
    res, est = estimators["MUSIC"].estimate(r_matrix, num_sources)
    results["MUSIC"] = est.locations if res else None
    res, est = estimators["RootMUSIC"].estimate(r_matrix, num_sources, D0)
    results["RootMUSIC"] = est.locations if res else None
    res, est = estimators["ESPRIT"].estimate(r_matrix, num_sources, D0)
    results["ESPRIT"] = est.locations if res else None
    r_ss = estimation.spatial_smooth(r_matrix, L_SUBARRAYS, fb=True)
    res, est = estimators["SSMUSIC"].estimate(r_ss, num_sources)
    results["SSMUSIC"] = est.locations if res else None
    return results


def mean_squared_error(locations, true_angles_rad):
    if locations is None or len(locations) != len(true_angles_rad):
        return None
    return float(np.mean(np.square(np.sort(locations) - true_angles_rad)))


def append_errors(error_map, locations_map, true_angles_rad):
    for method in METHOD_COLUMNS:
        error = mean_squared_error(locations_map[method], true_angles_rad)
        if error is not None:
            error_map[method].append(error)


def average_errors(error_map):
    return {method: float(np.mean(error_map[method])) if error_map[method] else np.nan for method in METHOD_COLUMNS}


def stochastic_crb(ula, true_angles_rad, snr_db, num_snapshots):
    sources = model.FarField1DSourcePlacement(true_angles_rad)
    rs = np.eye(len(true_angles_rad))
    power_noise = 10 ** (-snr_db / 10.0)
    crb = perf.crb_sto_farfield_1d(ula, sources, WAVELENGTH, rs, power_noise, num_snapshots)
    return float(np.real(np.mean(np.diag(crb))))


def collect_snr_mse():
    true_angle = 20.0
    true_angles_rad = np.radians([true_angle])
    snr_range = np.arange(-20, 11, 2)
    num_snapshots = 200
    num_repeats = 100
    ula, estimators = create_estimators()
    records = []

    for snr in snr_range:
        error_map = {method: [] for method in METHOD_COLUMNS}
        for _ in range(num_repeats):
            x_matrix = generate_ideal_data([true_angle], snr_db=snr, num_rx=NUM_RX, num_snapshots=num_snapshots)
            r_matrix = (x_matrix @ x_matrix.conj().T) / num_snapshots
            append_errors(error_map, estimate_locations(estimators, r_matrix, 1), true_angles_rad)
        row = {"SNR": int(snr), **average_errors(error_map), "CRB": stochastic_crb(ula, true_angles_rad, snr, num_snapshots)}
        records.append(row)
    return records


def collect_snr_mse_dca():
    true_angle = 20.0
    true_angles_rad = np.radians([true_angle])
    snr_range = np.arange(-20, 11, 2)
    ula, estimators = create_estimators()
    records = []

    for snr in snr_range:
        error_map = {method: [] for method in METHOD_COLUMNS}
        sim_cube = generate_dca1000_style_data([true_angle], snr_db=snr, sim_rx=NUM_RX)
        _, r_frames = process_radar_data(sim_cube, is_simulation=True)
        for frame in range(NUM_FRAMES):
            append_errors(error_map, estimate_locations(estimators, r_frames[frame], 1), true_angles_rad)
        row = {"SNR": int(snr), **average_errors(error_map), "CRB": stochastic_crb(ula, true_angles_rad, snr, NUM_CHIRPS)}
        records.append(row)
    return records


def collect_snapshot_mse():
    true_angles = [-20.0, 0.0, 20.0]
    true_angles_rad = np.sort(np.radians(true_angles))
    snapshots_range = [10, 20, 30, 50, 80, 100, 150, 200, 300, 500]
    fixed_snr = 0
    num_repeats = 500
    ula, estimators = create_estimators()
    records = []

    for snapshots in snapshots_range:
        error_map = {method: [] for method in METHOD_COLUMNS}
        for _ in range(num_repeats):
            x_matrix = generate_ideal_data(true_angles, snr_db=fixed_snr, num_rx=NUM_RX, num_snapshots=snapshots)
            r_matrix = (x_matrix @ x_matrix.conj().T) / snapshots
            append_errors(error_map, estimate_locations(estimators, r_matrix, len(true_angles)), true_angles_rad)
        row = {"Snapshots": int(snapshots), **average_errors(error_map), "CRB": stochastic_crb(ula, true_angles_rad, fixed_snr, snapshots)}
        records.append(row)
    return records


def collect_snapshot_mse_dca():
    true_angles = [-20.0, 0.0, 20.0]
    true_angles_rad = np.sort(np.radians(true_angles))
    snapshots_range = [5, 10, 15, 20, 30, 40, 50, NUM_CHIRPS]
    fixed_snr = 0
    ula, estimators = create_estimators()
    sim_cube = generate_dca1000_style_data(true_angles, snr_db=fixed_snr, sim_rx=NUM_RX)
    x_frames, _ = process_radar_data(sim_cube, is_simulation=True)
    records = []

    for snapshots in snapshots_range:
        error_map = {method: [] for method in METHOD_COLUMNS}
        for frame in range(NUM_FRAMES):
            x_frame = x_frames[frame]
            if x_frame.shape[0] != NUM_RX:
                x_frame = x_frame.T
            x_snap = x_frame[:, :snapshots]
            r_matrix = (x_snap @ x_snap.conj().T) / snapshots
            append_errors(error_map, estimate_locations(estimators, r_matrix, len(true_angles)), true_angles_rad)
        row = {"Snapshots": int(snapshots), **average_errors(error_map), "CRB": stochastic_crb(ula, true_angles_rad, fixed_snr, snapshots)}
        records.append(row)
    return records


def is_resolved_2(locations, delta_rad):
    if locations is None or len(locations) < 2:
        return False
    locs = np.sort(locations)
    if locs[0] >= 0 or locs[0] <= -delta_rad:
        return False
    if locs[1] <= 0 or locs[1] >= delta_rad:
        return False
    return True


def is_resolved_3(locations, delta_rad):
    if locations is None or len(locations) < 3:
        return False
    locs = np.sort(locations)
    true_locs = np.array([-delta_rad, 0.0, delta_rad])
    threshold = delta_rad / 2.0
    for idx in range(3):
        if np.abs(locs[idx] - true_locs[idx]) >= threshold:
            return False
    return True


def collect_resolution_success_for_source_num(source_num):
    delta_range = np.arange(0, 21, 2)
    snr = 10.0
    num_snapshots = 200
    num_repeats = 100
    _, estimators = create_estimators()
    records = []

    for delta in delta_range:
        counts = {method: 0 for method in METHOD_COLUMNS}
        delta_rad = np.radians(delta)
        true_angles = [-delta / 2, delta / 2] if source_num == 2 else [-delta, 0.0, delta]
        for _ in range(num_repeats):
            x_matrix = generate_ideal_data(true_angles, snr_db=snr, num_rx=NUM_RX, num_snapshots=num_snapshots)
            r_matrix = (x_matrix @ x_matrix.conj().T) / num_snapshots
            locations_map = estimate_locations(estimators, r_matrix, source_num)
            for method in METHOD_COLUMNS:
                checker = is_resolved_2 if source_num == 2 else is_resolved_3
                if checker(locations_map[method], delta_rad):
                    counts[method] += 1
        row = {"DeltaTheta": int(delta), "SourceNum": int(source_num), **{method: counts[method] / num_repeats for method in METHOD_COLUMNS}}
        records.append(row)
    return records


def collect_resolution_success():
    return collect_resolution_success_for_source_num(2) + collect_resolution_success_for_source_num(3)


def make_snr_key_points(ideal_records, dca_records):
    records = []
    for data_type, source_records in [("ideal", ideal_records), ("dca", dca_records)]:
        snr_values = [row["SNR"] for row in source_records]
        for target in [-10, 0, 10]:
            snr = nearest_value(snr_values, target)
            row = next(item for item in source_records if item["SNR"] == snr)
            records.append({"DataType": data_type, **row})
    return records


def first_x_below_threshold(x_list, y_list, threshold):
    for x_value, y_value in zip(x_list, y_list):
        if y_value < threshold:
            return x_value
    return None


def make_snapshot_thresholds(ideal_records, dca_records, threshold=1e-3):
    records = []
    for data_type, source_records in [("ideal", ideal_records), ("dca", dca_records)]:
        snapshots = [row["Snapshots"] for row in source_records]
        row = {"DataType": data_type, "Threshold": threshold}
        for method in METHOD_COLUMNS + ["CRB"]:
            row[method] = first_x_below_threshold(snapshots, [record[method] for record in source_records], threshold)
        records.append(row)
    return records


def make_resolution_thresholds(resolution_records, threshold=0.9):
    records = []
    for source_num in [2, 3]:
        source_records = [row for row in resolution_records if row["SourceNum"] == source_num]
        delta_values = [row["DeltaTheta"] for row in source_records]
        row = {"SourceNum": source_num, "Threshold": threshold}
        for method in METHOD_COLUMNS:
            row[method] = first_x_reach_threshold(delta_values, [record[method] for record in source_records], threshold)
        records.append(row)
    return records


def main():
    snr_records = collect_snr_mse()
    snr_dca_records = collect_snr_mse_dca()
    snapshot_records = collect_snapshot_mse()
    snapshot_dca_records = collect_snapshot_mse_dca()
    resolution_records = collect_resolution_success()

    save_csv(snr_records, TRADITION_DATA_DIR / "tradition_snr_mse.csv")
    save_csv(snr_dca_records, TRADITION_DATA_DIR / "tradition_snr_mse_dca.csv")
    save_csv(snapshot_records, TRADITION_DATA_DIR / "tradition_snapshot_mse.csv")
    save_csv(snapshot_dca_records, TRADITION_DATA_DIR / "tradition_snapshot_mse_dca.csv")
    save_csv(resolution_records, TRADITION_DATA_DIR / "tradition_resolution_success.csv")
    save_csv(make_snr_key_points(snr_records, snr_dca_records), TRADITION_DATA_DIR / "tradition_snr_key_points.csv")
    save_csv(make_snapshot_thresholds(snapshot_records, snapshot_dca_records), TRADITION_DATA_DIR / "tradition_snapshot_thresholds.csv")
    save_csv(make_resolution_thresholds(resolution_records), TRADITION_DATA_DIR / "tradition_resolution_thresholds.csv")


if __name__ == "__main__":
    main()
