# SudokuScan

SudokuScan is a computer vision project developed for COMP482. The goal is to detect a printed Sudoku puzzle from a photograph, correct its perspective, recognize its digits, and check the validity of the resulting Sudoku grid.

## Project Goals

- Preprocess Sudoku photographs.
- Detect the four outer corners using Harris corner detection.
- Rectify the Sudoku using affine transformations and a homography.
- Recognize digits using OCR.
- Represent the puzzle as a 9×9 grid, using `0` for empty cells.
- Check Sudoku row, column, and 3×3 box validity.

## Technologies

- Python 3.10+
- OpenCV
- NumPy
- Tesseract OCR
- pytesseract

## Requirements

Python dependencies are listed in `requirements.txt`.

Tesseract OCR must also be installed separately on your computer and accessible from the command line.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd sudokuScan
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install the Tesseract OCR application separately and make sure the `tesseract` command is available in your terminal.

Verify the installation:

```bash
tesseract --version
```

## Project Status

The development environment and initial OCR setup have been prepared. The image-processing pipeline, corner detection, perspective correction, digit recognition, and validation stages will be implemented incrementally.

## Usage

The command-line interface and output formats will be documented here as they are implemented.

## Course

COMP482 — Computer Vision

Project: SudokuScan