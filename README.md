# SudokuScan (COMP 482, Project 1)

Photograph of a printed sudoku -> 9x9 digital grid. The four outer corners are
found with the Harris detector, the puzzle is rectified with an affine (A3, A4)
and a projective (H4) model, the 81 cells are cut out and the digits are read
with Tesseract.

## Install

```
pip install -r requirements.txt
```

The Tesseract engine is a separate program:

* Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (default
  path `C:\Program Files\Tesseract-OCR\tesseract.exe` is found automatically).
  Elsewhere: set the environment variable `TESSERACT_CMD` to the executable.
* Ubuntu/Debian: `sudo apt install tesseract-ocr`
* macOS: `brew install tesseract`

## Run

```
python sudoku_scan.py --input photo.jpg --outdir out/
python sudoku_scan.py --batch images_dir/ --outdir results/          # grader mode
python sudoku_scan.py --batch input/ --outdir output/ --debug        # + debug folders
```

For every image `<name>` the output directory gets `<name>_grid.txt`,
`<name>_result.json`, `<name>_harris.png`, `<name>_rect_A3.png`,
`<name>_rect_A4.png`, `<name>_rect_H4.png` and `<name>_debug.png`. If no puzzle
is found, `<name>_result.json` contains `"verdict": "not_found"` (a failing
image never stops a batch). `--debug` also writes
`<outdir>/<name>/{preprocessing,lines,corners,recognition}/`.

The parameters are the constants at the top of `sudoku_scan.py` (edit and
rerun for the parameter study). The result JSON contains
`n_harris_candidates` and `n_region_candidates` for that purpose.

## Layout

| File | Content |
|---|---|
| `sudoku_scan.py` | main script: numbered steps from photo to result |
| `helpers.py` | drawing, puzzle region, sub-pixel refinement, saving the output files |
| `corner_detector.py` | Harris candidate extraction (threshold + NMS) and outer-corner selection |
| `rectification.py` | A3, A4, H4 estimation, residuals, warping |
| `recognition.py` | cell extraction, empty-cell criterion, OCR preparation, Tesseract |
| `validation.py` | row / column / box check |

## Notes

* All pixel parameters are tuned for a working size of 1000 px on the longer
  side; larger photos are downscaled, corners are reported in original pixels.
* Matrices in the JSON map original-image pixels to the 450x450 target.
