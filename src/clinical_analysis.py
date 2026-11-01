"""Reusable SNA, SNB, signed ANB, paired agreement, and category analysis.

Coordinates must use verified source-image (x, y) pixel geometry. Axis spacing
is either dimensionless relative spacing or verified physical units per pixel.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

CANONICAL_LANDMARKS = (
    "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
    "Pogonion", "Menton", "Gnathion", "Gonion", "Lower incisal incision",
    "Upper incisal incision", "Upper lip", "Lower lip", "Subnasale",
    "Soft tissue pogonion", "Posterior nasal spine", "Anterior nasal spine", "Articulare",
)
LANDMARKS = ("Sella", "Nasion", "Subspinale", "Supramentale")
MEASURES = ("SNA", "SNB", "ANB")
CATEGORY_ORDER = ("Class III", "Class I", "Class II")
CLINICAL_LANDMARK_INDICES = {"Sella": 0, "Nasion": 1, "Subspinale": 4, "Supramentale": 5}
ALIASES = {name.lower(): name for name in CANONICAL_LANDMARKS}
ALIASES.update({
    "s": "Sella", "n": "Nasion", "a point": "Subspinale", "a-point": "Subspinale", "point a": "Subspinale",
    "b point": "Supramentale", "b-point": "Supramentale", "point b": "Supramentale",
})
GEOMETRY_COLUMNS = ("coordinate_system", "source_width", "source_height", "x_spacing", "y_spacing", "origin", "orientation")
SUPPORTED_COORDINATE_SYSTEM = "source_image_pixels_xy"


def _point(value, label):
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must be a finite (x, y) pair")
    return point


def interior_angle(vertex, first, second, spacing=(1.0, 1.0), minimum_norm=1e-12):
    """Return a stable [0, 180] interior angle in degrees."""
    v = _point(vertex, "vertex")
    scale = np.asarray(spacing, dtype=np.float64)
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("spacing must contain two positive finite axis spacings")
    a = (_point(first, "first") - v) * scale
    b = (_point(second, "second") - v) * scale
    if np.linalg.norm(a) <= minimum_norm or np.linalg.norm(b) <= minimum_norm:
        raise ValueError("degenerate geometry: a ray has near-zero length")
    cross = abs(float(a[0] * b[1] - a[1] * b[0]))
    return float(np.degrees(np.arctan2(cross, float(np.dot(a, b)))))


def angles_from_landmarks(points: Mapping[str, Sequence[float]], spacing=(1.0, 1.0)):
    """Compute SNA and SNB at Nasion and retain signed ANB = SNA - SNB."""
    missing = [name for name in LANDMARKS if name not in points]
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")
    s, n, a, b = (_point(points[name], name) for name in LANDMARKS)
    sna = interior_angle(n, s, a, spacing)
    snb = interior_angle(n, s, b, spacing)
    return {"SNA": sna, "SNB": snb, "ANB": sna - snb}


def angles_from_ordered_array(points, indices=None, spacing=(1.0, 1.0)):
    """Compute angles using an explicit zero-based landmark index mapping."""
    indices = CLINICAL_LANDMARK_INDICES if indices is None else indices
    required = set(LANDMARKS)
    if set(indices) != required:
        raise ValueError(f"Landmark index map must define exactly {sorted(required)}")
    values = list(indices.values())
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in values):
        raise ValueError("Landmark indices must be non-negative integers")
    if len(values) != len(set(values)):
        raise ValueError("Landmark indices must be unique")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or max(values) >= len(array):
        raise ValueError(f"Expected an ordered (N, 2) array containing index {max(values)}")
    return angles_from_landmarks({name: array[index] for name, index in indices.items()}, spacing)


def classify_anb(anb, class_iii_upper=0.0, class_i_upper=4.0):
    """Apply the approved thresholds without rounding: <0, [0,4], >4 degrees."""
    value = float(anb)
    if not math.isfinite(value):
        raise ValueError("ANB must be finite")
    if not class_iii_upper < class_i_upper:
        raise ValueError("category thresholds must be strictly increasing")
    if value < class_iii_upper:
        return "Class III"
    return "Class I" if value <= class_i_upper else "Class II"


def canonical_landmark_name(value):
    key = " ".join(str(value).strip().lower().replace("_", " ").split())
    try:
        return ALIASES[key]
    except KeyError as error:
        raise ValueError(f"Unrecognised landmark identity {value!r}") from error


def validate_coordinate_table(rows, expected_ids=None, required_landmarks=LANDMARKS,
                              require_source_geometry=True, coordinate_system=SUPPORTED_COORDINATE_SYSTEM):
    """Validate all input rows; retain full coordinate data, calculate angles from four points only."""
    rows = list(rows)
    required_landmarks = tuple(canonical_landmark_name(n) for n in required_landmarks)
    if not rows:
        raise ValueError("Coordinate input contains no rows")
    required_columns = {"image_id", "landmark_name", "reference_x", "reference_y", "predicted_x", "predicted_y"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(f"Coordinate input missing columns: {sorted(missing_columns)}")
    grouped, audit, seen = {}, [], set()
    for row_number, row in enumerate(rows, start=2):
        image_id = str(row.get("image_id", "")).strip()
        record = grouped.setdefault(image_id, {"metadata": None, "points": {}, "bad": [], "rows": [], "raw_rows": []})
        record["raw_rows"].append(dict(row))
        try:
            if not image_id:
                raise ValueError("image_id must not be empty")
            landmark = canonical_landmark_name(row["landmark_name"])
            key = (image_id, landmark)
            if key in seen:
                raise ValueError("duplicate image-landmark pair")
            seen.add(key)
            ref = np.asarray([float(row["reference_x"]), float(row["reference_y"])])
            pred = np.asarray([float(row["predicted_x"]), float(row["predicted_y"])])
            if not np.isfinite(ref).all() or not np.isfinite(pred).all():
                raise ValueError("reference/prediction coordinates must be finite")
            if require_source_geometry:
                metadata = {}
                for col in GEOMETRY_COLUMNS:
                    if col not in row or row[col] in (None, ""):
                        if col in ("x_spacing", "y_spacing") and row.get("spacing_units") == "dimensionless_relative":
                            metadata[col] = 1.0
                            continue
                        raise ValueError(f"missing required geometry metadata: {col}")
                    metadata[col] = row[col]
                if not metadata["origin"] or not metadata["orientation"]:
                    raise ValueError("origin and orientation metadata must not be empty")
                if metadata["coordinate_system"] != SUPPORTED_COORDINATE_SYSTEM:
                    raise ValueError("unsupported coordinate system; source_image_pixels_xy is required")
                if coordinate_system and metadata["coordinate_system"] != coordinate_system:
                    raise ValueError("inconsistent coordinate system")
                width, height = float(metadata["source_width"]), float(metadata["source_height"])
                sx, sy = float(metadata["x_spacing"]), float(metadata["y_spacing"])
                if not all(math.isfinite(v) and v > 0 for v in (width, height, sx, sy)):
                    raise ValueError("source dimensions and axis spacing must be positive finite values")
                units = str(row.get("spacing_units", "physical"))
                if units not in ("physical", "dimensionless_relative"):
                    raise ValueError("spacing_units must be physical or dimensionless_relative")
                unit_name = row.get("spacing_unit_name")
                if units == "physical" and not unit_name:
                    raise ValueError("verified physical spacing requires explicit spacing_unit_name")
                if units == "dimensionless_relative" and not (math.isclose(sx, 1.0) and math.isclose(sy, 1.0)):
                    raise ValueError("dimensionless relative spacing must be (1, 1)")
                signature = tuple(str(metadata[col]) for col in GEOMETRY_COLUMNS) + (units, str(unit_name or ""))
                if record["metadata"] is not None and record["metadata"] != signature:
                    raise ValueError("inconsistent geometry metadata within image")
                record["metadata"] = signature
            record["rows"].append(dict(row, landmark_name=landmark))
            if landmark in required_landmarks:
                record["points"][landmark] = (ref, pred)
            audit.append({"row_number": row_number, "image_id": image_id, "landmark_name": landmark,
                          "status": "accepted", "reason": ""})
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            record["bad"].append(str(error))
            audit.append({"row_number": row_number, "image_id": image_id,
                          "landmark_name": str(row.get("landmark_name", "")),
                          "status": "invalid", "reason": str(error)})
    if expected_ids is not None:
        expected = {str(value) for value in expected_ids}
        actual = set(grouped)
        missing, unexpected = sorted(expected - actual), sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(f"Cohort mismatch; missing IDs={missing}, unexpected IDs={unexpected}")
    patients = {}
    for image_id, record in grouped.items():
        if require_source_geometry and record["metadata"] is None:
            record["bad"].append("no valid source geometry metadata")
        missing = sorted(set(required_landmarks) - set(record["points"]))
        if record["bad"] or missing:
            audit.append({"row_number": None, "image_id": image_id, "landmark_name": "", "status": "invalid_patient",
                          "reason": "; ".join(record["bad"] + ([f"missing landmarks: {missing}"] if missing else []))})
            continue
        if require_source_geometry:
            values = record["metadata"]
            metadata = dict(zip(GEOMETRY_COLUMNS, values[:len(GEOMETRY_COLUMNS)]))
            metadata["spacing_units"] = values[-2]
            metadata["spacing_unit_name"] = values[-1] or None
            spacing = (float(metadata["x_spacing"]), float(metadata["y_spacing"]))
            if values[-2] == "dimensionless_relative":
                metadata.update(physical_spacing_x=None, physical_spacing_y=None, physical_spacing_units=None)
            else:
                metadata.update(physical_spacing_x=spacing[0], physical_spacing_y=spacing[1],
                                physical_spacing_units=f"{metadata['spacing_unit_name']}_per_pixel")
        else:
            metadata, spacing = {}, (1.0, 1.0)
        patients[image_id] = {"reference": {n: record["points"][n][0] for n in required_landmarks},
                              "predicted": {n: record["points"][n][1] for n in required_landmarks},
                              "metadata": metadata, "spacing": spacing, "coordinate_rows": record["rows"], "raw_coordinate_rows": record["raw_rows"]}
    return patients, audit


def read_coordinate_export(path, **kwargs):
    """Load and validate a long-form CSV or JSON coordinate export."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON must be a row list or an object with a 'rows' list")
    else:
        raise ValueError("Coordinate export must be CSV or JSON")
    return validate_coordinate_table(rows, **kwargs)


def compute_patient_angles(patients):
    """Return sorted paired angles, signed prediction errors, and categories."""
    rows = []
    for image_id in sorted(patients):
        record = patients[image_id]
        reference = angles_from_landmarks(record["reference"], record.get("spacing", (1.0, 1.0)))
        predicted = angles_from_landmarks(record["predicted"], record.get("spacing", (1.0, 1.0)))
        rows.append({"image_id": image_id,
                     **{f"reference_{k.lower()}_deg": v for k, v in reference.items()},
                     **{f"predicted_{k.lower()}_deg": v for k, v in predicted.items()},
                     **{f"{k.lower()}_signed_error_deg": predicted[k] - reference[k] for k in MEASURES},
                     "reference_category": classify_anb(reference["ANB"]),
                     "predicted_category": classify_anb(predicted["ANB"]), "metadata": record.get("metadata", {})})
    return rows


def _summary_stats(reference, predicted):
    ref, pred = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    diff, n = pred - ref, len(ref)
    if n == 0:
        return {"n": 0, "reference_mean_deg": None, "reference_sd_deg": None, "predicted_mean_deg": None,
                "predicted_sd_deg": None, "mae_deg": None, "rmse_deg": None, "bias_deg": None,
                "difference_sd_deg": None, "loa_lower_deg": None, "loa_upper_deg": None}
    bias = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if n > 1 else None
    return {"n": n, "reference_mean_deg": float(np.mean(ref)),
            "reference_sd_deg": float(np.std(ref, ddof=1)) if n > 1 else None,
            "predicted_mean_deg": float(np.mean(pred)), "predicted_sd_deg": float(np.std(pred, ddof=1)) if n > 1 else None,
            "mae_deg": float(np.mean(np.abs(diff))), "rmse_deg": float(np.sqrt(np.mean(diff ** 2))),
            "bias_deg": bias, "difference_sd_deg": sd,
            "loa_lower_deg": bias - 1.96 * sd if sd is not None else None,
            "loa_upper_deg": bias + 1.96 * sd if sd is not None else None}


def _bootstrap_ci(reference, predicted, rng, repetitions):
    ref, pred = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    names = ("mae_deg", "rmse_deg", "bias_deg")
    if len(ref) < 2 or repetitions <= 0:
        return {n: {"lower": None, "upper": None, "valid_replicates": 0, "undefined_replicates": 0} for n in names}
    samples = {n: [] for n in names}
    for _ in range(repetitions):
        indices = rng.integers(0, len(ref), len(ref))
        diff = pred[indices] - ref[indices]
        vals = {"mae_deg": np.mean(np.abs(diff)), "rmse_deg": np.sqrt(np.mean(diff ** 2)), "bias_deg": np.mean(diff)}
        for name, value in vals.items():
            if np.isfinite(value): samples[name].append(float(value))
    return {name: {"lower": float(np.percentile(vals, 2.5)) if vals else None,
                   "upper": float(np.percentile(vals, 97.5)) if vals else None,
                   "valid_replicates": len(vals), "undefined_replicates": repetitions-len(vals)}
            for name, vals in samples.items()}


def agreement_analysis(rows, seed=42, bootstrap_repetitions=5000):
    """Angular agreement metrics and paired patient-level percentile bootstrap CIs."""
    rng = np.random.default_rng(seed); output = {}
    for measure in MEASURES:
        ref = [row[f"reference_{measure.lower()}_deg"] for row in rows]
        pred = [row[f"predicted_{measure.lower()}_deg"] for row in rows]
        output[measure] = {"summary": _summary_stats(ref, pred),
                           "bootstrap_95_ci": _bootstrap_ci(ref, pred, rng, bootstrap_repetitions)}
    return output


def _cohen_kappa(confusion):
    matrix = np.asarray(confusion, dtype=float); total = matrix.sum()
    if total == 0: return None
    expected = float(np.dot(matrix.sum(axis=1), matrix.sum(axis=0)) / total**2)
    return None if math.isclose(expected, 1.0) else float((np.trace(matrix) / total - expected) / (1 - expected))


def _category_summary(rows):
    matrix = np.zeros((3, 3), dtype=int); index = {name: i for i, name in enumerate(CATEGORY_ORDER)}
    for row in rows: matrix[index[row["reference_category"]], index[row["predicted_category"]]] += 1
    support = matrix.sum(axis=1)
    rates = np.divide(matrix, support[:, None], out=np.full((3, 3), np.nan), where=support[:, None] != 0)
    per_class = {}
    for i, label in enumerate(CATEGORY_ORDER):
        tp = int(matrix[i, i]); fp = int(matrix[:, i].sum() - tp); fn = int(support[i] - tp)
        precision = tp / (tp + fp) if tp + fp else None; recall = tp / support[i] if support[i] else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if precision == 0 or recall == 0 else None)
        per_class[label] = {"support": int(support[i]), "recall": recall, "precision": precision, "f1": f1}
    n = int(matrix.sum())
    return {"labels": list(CATEGORY_ORDER), "confusion_counts": matrix.tolist(),
            "reference_row_rates": [[float(v) if np.isfinite(v) else None for v in row] for row in rates],
            "n": n, "overall_agreement": float(np.trace(matrix) / n) if n else None,
            "kappa_unweighted": _cohen_kappa(matrix), "per_class": per_class}


def category_analysis(rows, seed=42, bootstrap_repetitions=5000):
    """Fixed-order category agreement and patient-level bootstrap CIs."""
    rows = list(rows); summary = _category_summary(rows); rng = np.random.default_rng(seed)
    reps = bootstrap_repetitions if len(rows) >= 2 else 0
    agreement, kappas, undefined_kappa = [], [], 0
    for _ in range(reps):
        sampled = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        result = _category_summary(sampled); agreement.append(result["overall_agreement"])
        if result["kappa_unweighted"] is None: undefined_kappa += 1
        else: kappas.append(result["kappa_unweighted"])
    summary["bootstrap_95_ci"] = {
        "overall_agreement": {"lower": float(np.percentile(agreement, 2.5)) if agreement else None,
                              "upper": float(np.percentile(agreement, 97.5)) if agreement else None,
                              "valid_replicates": len(agreement), "undefined_replicates": reps-len(agreement)},
        "kappa": {"lower": float(np.percentile(kappas, 2.5)) if kappas else None,
                  "upper": float(np.percentile(kappas, 97.5)) if kappas else None,
                  "valid_replicates": len(kappas),
                  "undefined_replicates": undefined_kappa + max(0, bootstrap_repetitions-reps)}}
    return summary


def discordant_cases(rows):
    """Export category-discordant cases with all angles and boundary distances."""
    output = []
    for row in rows:
        if row["reference_category"] == row["predicted_category"]: continue
        ref, pred = row["reference_anb_deg"], row["predicted_anb_deg"]
        output.append({"image_id": row["image_id"], **{k: v for k, v in row.items() if k.endswith("_deg")},
                       "reference_category": row["reference_category"], "predicted_category": row["predicted_category"],
                       "anb_signed_error_deg": pred-ref, "reference_distance_to_0_deg": abs(ref),
                       "predicted_distance_to_0_deg": abs(pred), "reference_distance_to_4_deg": abs(ref-4),
                       "predicted_distance_to_4_deg": abs(pred-4)})
    return output


def json_safe(value):
    """Convert NumPy values and non-finite floats to JSON-compatible values."""
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


__all__ = ["ALIASES", "CANONICAL_LANDMARKS", "CATEGORY_ORDER", "CLINICAL_LANDMARK_INDICES", "LANDMARKS", "MEASURES",
           "agreement_analysis", "angles_from_landmarks", "angles_from_ordered_array", "canonical_landmark_name",
           "category_analysis", "classify_anb", "compute_patient_angles", "discordant_cases", "interior_angle",
           "json_safe", "read_coordinate_export", "validate_coordinate_table", "write_json"]
