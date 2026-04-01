# PDF Parser Arena

Benchmarking framework for comparing three PDF-to-Markdown parsers: **Docling**, **MinerU**, and **PaddleOCR**. Each parser runs in its own conda environment, producing rich intermediate outputs and detailed logs for quality analysis.

## Prerequisites

Three conda environments, each with its respective parser installed:

| Environment | Package | Install |
|---|---|---|
| `docling` | IBM Docling | `pip install docling` |
| `mineru` | OpenDataLab MinerU | `pip install mineru[core]` |
| `paddleocr` | Baidu PaddleOCR | `pip install paddleocr` |

GPU (CUDA) recommended for reasonable performance. Each parser will fall back to CPU if no GPU is available.

## Quick Start

### Run all parsers on all PDFs

```bash
python pdf_parser_arena/run_all.py --pdf-dir pdfs
```

This scans `pdfs/` for all PDF files and runs each parser sequentially, with automatic conda environment switching and GPU cleanup between runs.

### Run a single parser on one PDF

```bash
# Activate the corresponding environment first
conda activate docling
python pdf_parser_arena/run_docling.py pdfs/papers/Corvid.pdf

conda activate mineru
python pdf_parser_arena/run_mineru.py pdfs/papers/Corvid.pdf

conda activate paddleocr
python pdf_parser_arena/run_paddleocr.py pdfs/papers/Corvid.pdf
```

## Scripts

| Script | Purpose |
|---|---|
| `run_all.py` | Master orchestrator - runs all parsers on all PDFs |
| `run_docling.py` | Standalone Docling runner |
| `run_mineru.py` | Standalone MinerU runner |
| `run_paddleocr.py` | Standalone PaddleOCR runner |
| `utils.py` | Shared utilities (logging, output dirs, GPU cleanup, timing) |

## Command-Line Options

### `run_all.py`

```
python run_all.py [--pdf-dir DIR] [--output-dir DIR] [--parsers PARSER [PARSER ...]]

Options:
  --pdf-dir       Directory containing PDFs (default: ../pdfs)
  --output-dir    Base output directory (default: results/)
  --parsers       Which parsers to run: docling mineru paddleocr (default: all)
```

### Individual runners

```
python run_<parser>.py <pdf_path> [--output-dir DIR]

Arguments:
  pdf_path        Path to the input PDF file
  --output-dir    Base output directory (default: results/)
```

## Output Structure

Each run creates a timestamped directory:

```
results/
└── Corvid_20260401_103000/
    ├── docling/
    │   ├── output.md               # Final markdown
    │   ├── output.json             # Full document structure (JSON)
    │   ├── output.html             # HTML rendering
    │   ├── pages/                  # Page images (PNG per page)
    │   ├── tables/                 # Table images, CSV, HTML exports
    │   ├── figures/                # Extracted figure images
    │   ├── debug/                  # Layout/OCR/table visualizations
    │   ├── profiling.json          # Per-stage timing from Docling
    │   ├── arena_timings.json      # Wall-clock timing per phase
    │   └── run.log                 # Full debug log
    │
    ├── mineru/
    │   ├── mineru_raw/             # MinerU's native output
    │   │   └── <pdf_stem>/
    │   │       └── auto/
    │   │           ├── <name>.md           # Final markdown
    │   │           ├── <name>_middle.json  # Intermediate representation
    │   │           ├── <name>_model.json   # Model predictions
    │   │           ├── <name>_content_list.json
    │   │           ├── <name>_layout.pdf   # Layout bounding box visualization
    │   │           ├── <name>_span.pdf     # Span bounding box visualization
    │   │           ├── <name>_origin.pdf   # Original PDF copy
    │   │           └── images/             # Extracted images
    │   ├── produced_files.json     # Inventory of all produced files
    │   ├── arena_timings.json      # Wall-clock timing
    │   └── run.log                 # Full debug log (includes loguru output)
    │
    └── paddleocr/
        ├── output.md               # Concatenated markdown (all pages)
        ├── per_page/               # Per-page results
        │   ├── page_001.md         # Page markdown
        │   ├── page_001/           # Page images
        │   └── page_001_meta.json  # Page result metadata
        ├── layout/                 # Layout detection JSON per page
        ├── ocr/                    # OCR results JSON per page
        ├── tables/                 # Table HTML per page
        ├── arena_timings.json      # Wall-clock timing
        └── run.log                 # Full debug log
```

## Clone PDF Parser 

### git implement
Pull the repo
```bash
# 一步到位
git clone --recursive https://github.com/clairetsai1222/pdf_parser_arena.git
cd pdf_parser_arena/<comparing method>/

# 先clone主文件
git clone https://github.com/clairetsai1222/pdf_parser_arena.git
cd pdf_parser_arena/
git submodule init
git submodule update
```

Add new pdf parser
```bash
# git submodule add <远程仓库地址> <本地存放路径>
git submodule add https://github.com/author/pdf_parser_lib_a.git ./pdf_parser_lib_a

git add .
git commit -m "new pdf parser: pdf_parser_lib_a"
git push origin main

```

Pull New Method Version
```bash
cd pdf_parser_arena/<comparing method>/
git pull origin main  # 获取该子模块的最新代码
cd ..                 # 回到主项目根目录
git add ./pdf_parser_arena/<comparing method>/ # 将更新后的子模块指针添加到暂存区
git commit -m "Update pdf_parser_lib_a to latest version"
```


## Intermediate Files Explained

### Docling
- **`debug/`**: Visualization overlays showing layout clusters, OCR regions, table cells, and PDF cells on top of page images. Generated by Docling's built-in debug mode.
- **`profiling.json`**: Per-pipeline-stage timing (layout detection, OCR, table structure, etc.) with count, total, and average.
- **`tables/*.csv`**: Each detected table exported as CSV for easy comparison.

### MinerU
- **`*_middle.json`**: Core intermediate representation containing per-page layout info, text blocks, images, and reading order.
- **`*_model.json`**: Raw model predictions before post-processing.
- **`*_layout.pdf`**: Original PDF annotated with layout bounding boxes (colored by type).
- **`*_span.pdf`**: Original PDF annotated with text span bounding boxes.

### PaddleOCR
- **`layout/*.json`**: Per-page layout detection results with region types, bounding boxes, and confidence scores.
- **`ocr/*.json`**: Per-page OCR results with detected text polygons, recognized text, and confidence scores.
- **`tables/*.html`**: Per-page table recognition results in HTML format.

## GPU Memory Management

The orchestrator (`run_all.py`) handles GPU cleanup between parsers:

1. Each parser runs as a **separate subprocess** - process exit releases most GPU memory
2. After each parser, a cleanup script runs in the same conda env to clear framework-specific GPU caches (`torch.cuda.empty_cache()` or `paddle.device.cuda.empty_cache()`)
3. Only the current process's memory is cleaned - **other users' GPU processes are not affected**

## Adding PDFs

Place PDF files anywhere under `pdfs/` (supports subdirectories):

```
pdfs/
├── papers/
│   ├── Corvid.pdf
│   └── another_paper.pdf
└── reports/
    └── annual_report.pdf
```

## Timeout

Each parser has a 30-minute timeout per PDF (configurable via `TIMEOUT_SECONDS` in `run_all.py`).

