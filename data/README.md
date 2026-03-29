# Data

Raw MRI data is **not committed** to this repository. Download from the source below.

## ABIDE-I Dataset

- **Source:** [Autism Brain Imaging Data Exchange (ABIDE-I)](https://fcon_1000.projects.nitrc.org/indi/abide/)
- **Access:** Via COINS (Collaborative Informatics and Neuroimaging Suite) — free, requires registration
- **Format:** NIfTI (.nii.gz), 3D volumetric scans
- **Scale:** 1,112 subjects from 17 international research centres

## Preprocessing Steps (done in notebooks)

1. **NIfTI → DICOM conversion** — using `nifti2dicom` ([reference](https://pycad.co/nifti2dicom/))
2. **Data cleaning** — remove subjects with missing/unclear sMRI (19 removed) and "Unknown" label (26 removed) → **1,067 subjects**
3. **DICOM → PNG** — extract individual 2D axial slices
4. **Slice filtering** — remove empty or partially formed brain slices → **100,510 usable PNG slices**
5. **CSV label files** — slice paths and binary labels (ASD=1, TC=0) stored in CSV files:
   - `extracted_random_labels_train.csv` — 2,400 slices
   - `extracted_random_labels_validation.csv` — 300 slices
   - `extracted_random_labels_test.csv` — 300 slices
   - `half_train.csv` — 50% subset used in early experiments

## Kaggle Setup

All notebooks were developed on Kaggle. The processed PNG slices and CSV files were uploaded as Kaggle datasets:
- Image dataset: `/kaggle/input/autism/`
- CSV files: `/kaggle/input/autism-csv/`

## Local / Colab Adaptation

To run notebooks locally or on Colab, update the path replacement in `process_image()`:

```python
# In each notebook, find this line and update the path:
image_path = image_path.replace(
    "E:\\TARUN\\Projects\\Autism Detection\\Data\\data_png",
    "/your/local/path/to/png_slices"
)
```
