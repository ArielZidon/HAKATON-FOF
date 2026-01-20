# Data Team Guide: Coordinate Classification Dataset

This reference explains exactly how to supply training/validation CSVs for the coordinate-classification model. Please share this document with anyone preparing data extracts.

## 1. File Format
- **File type**: CSV (UTF-8, comma-separated, header row required).
- **Row granularity**: Each row captures a single tactical snapshot containing:
  1. A list of historical friend positions.
  2. A list of historical foe positions.
  3. The target coordinate that must be labeled.
  4. The ground-truth label (friend/foe). Omit or leave blank only for live inference batches.
- **Delimiter escaping**: Because coordinate lists are stored as JSON strings, wrap each JSON array in double quotes inside the CSV.

## 2. Column Specification
| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `friend_pos_list` | JSON string | Yes | JSON array of `[longitude, latitude]` float pairs (WGS84 degrees). At least one entry required. |
| `foe_pos_list` | JSON string | Yes | Same format as `friend_pos_list`, representing opponent positions. At least one entry required. |
| `target_coord` | JSON string | Yes | Single `[longitude, latitude]` float pair (longitude first). |
| `label` | string | Yes for training | Literal `friend` or `foe`. Case-sensitive, lowercase recommended. |

## 3. Coordinate Rules
- Longitude range: **-180 ≤ lon ≤ 180**.
- Latitude range: **-90 ≤ lat ≤ 90**.
- Precision: Provide at least 5 decimal places when available to preserve spatial fidelity.
- Ordering: Maintain chronological or observation order in the lists; the model can leverage spatial-temporal structure later.
- Missing values: Do not emit `null`/`None`. If a scenario lacks friend or foe coordinates, discard the row or consult the ML team before submission.

## 4. Example Rows
```
friend_pos_list,foe_pos_list,target_coord,label
"[[-73.98,40.75],[-73.97,40.76]]","[[-74.02,40.71],[-74.03,40.70]]","[-73.99,40.74]",friend
"[[34.80,31.25],[34.82,31.27]]","[[34.78,31.20],[34.79,31.18]]","[34.81,31.26]",friend
"[[139.70,35.68],[139.72,35.69]]","[[139.75,35.66],[139.77,35.65],[139.73,35.64]]","[139.74,35.67]",foe
```
A downloadable copy lives at `model_coordinate/data/sample_training_input.csv`.

## 5. Validation Checklist (run before delivery)
- [ ] Headers exactly match: `friend_pos_list, foe_pos_list, target_coord, label`.
- [ ] Every JSON string parses without errors.
- [ ] No empty coordinate lists.
- [ ] All coordinates respect longitude/latitude bounds.
- [ ] Labels only `friend` or `foe` (lowercase).
- [ ] File encoded in UTF-8, no BOM.

## 6. Delivery Expectations
- Provide separate CSVs for **train**, **validation**, and **test** if possible, or one master CSV with a `split` column we can honor.
- Include a short README describing data provenance, collection time range, coordinate reference system (CRS), and any known anomalies.
- Transfer files via our secure bucket or the agreed Git LFS path; avoid email attachments for large datasets.

Questions? Contact the ML team (#coord-classifier Slack channel) with example rows if anything is ambiguous. Thank you for keeping the inputs clean and consistent!
