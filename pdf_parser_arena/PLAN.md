# PDF Parser Arena - Implementation Plan

## Context

We need to build a comparison framework in `pdf_parser_arena/` to benchmark three PDF parsers (Docling, MinerU, PaddleOCR) for PDF-to-Markdown/DOCX conversion quality. The goal is to collect rich intermediate outputs and logs from each parser to analyze their strengths/weaknesses. **Critical constraint**: we must NOT change any code logic in `docling/`, `MinerU/`, or `PaddleOCR/` — only allowed to add/change intermediate file storage and logging verbosity in those folders.

## File Structure

```
pdf_parser_arena/
├── utils.py              # Shared utilities (logging, output dir, GPU cleanup, timing)
├── run_docling.py        # Standalone Docling runner
├── run_mineru.py         # Standalone MinerU runner
├── run_paddleocr.py      # Standalone PaddleOCR runner
├── run_all.py            # Master orchestrator (scans pdfs/, runs all 3 sequentially)
├── PLAN.md               # This plan
└── README.md             # Usage documentation
```

## Output Directory Structure

Each parser run creates: `results/{pdf_stem}_{YYYYMMDD_HHMMSS}/{parser_name}/`

```
results/
└── Corvid_20260401_103000/
    ├── docling/
    │   ├── output.md
    │   ├── output.json
    │   ├── output.html
    │   ├── pages/              # Page images (PNG per page)
    │   ├── tables/             # Table images + CSV exports
    │   ├── figures/            # Extracted figure images
    │   ├── debug/              # Layout/OCR/table visualizations
    │   ├── profiling.json      # Stage-by-stage timing
    │   └── run.log             # Full debug log
    ├── mineru/
    │   ├── auto/               # MinerU's native output structure
    │   │   ├── {name}.md
    │   │   ├── {name}_middle.json
    │   │   ├── {name}_model.json
    │   │   ├── {name}_content_list.json
    │   │   ├── {name}_layout.pdf
    │   │   ├── {name}_span.pdf
    │   │   └── images/
    │   └── run.log             # Full debug log
    └── paddleocr/
        ├── output.md           # Concatenated markdown (all pages)
        ├── per_page/           # Per-page markdown + images
        │   ├── page_001.md
        │   ├── page_001/       # Images for that page
        │   └── ...
        ├── layout/             # Layout detection results (JSON per page)
        ├── ocr/                # OCR results (JSON per page)
        ├── tables/             # Table HTML per page
        └── run.log             # Full debug log
```

## Module Designs

### 1. `utils.py` — Shared Utilities

- **`create_output_dir(pdf_path, parser_name, base_dir="results")`** — creates timestamped output directory, returns Path
- **`setup_logger(log_path)`** — creates a logger that writes to both file and stdout, returns logger + log file path
- **`cleanup_gpu(parser_name)`** — parser-specific GPU/memory cleanup:
  - For docling/mineru (PyTorch): `torch.cuda.empty_cache()`, `gc.collect()`
  - For paddleocr (Paddle): `paddle.device.cuda.empty_cache()`, `gc.collect()`
  - Only cleans current process memory, no `kill` or system-wide ops
- **`get_pdf_files(pdf_dir)`** — scans directory for all `.pdf` files
- **`TimingContext`** — context manager that records elapsed time for a named stage
- **`save_json(data, path)`** — helper to save dict as formatted JSON

### 2. `run_docling.py` — Docling Runner

**Usage**: `python run_docling.py <pdf_path> [--output-dir <dir>]`

**Logic**:
1. Configure Docling with max intermediate output (debug visualizations, page images, profiling)
2. Run conversion, capture timing
3. Save outputs: markdown, JSON, HTML
4. Extract and save page images, table images+CSV, figure images
5. Save profiling data as JSON
6. Log summary

### 3. `run_mineru.py` — MinerU Runner

**Usage**: `python run_mineru.py <pdf_path> [--output-dir <dir>]`

**Logic**:
1. Set `MINERU_LOG_LEVEL=DEBUG` before import
2. Call `do_parse()` with all dump flags enabled (middle_json, model_output, content_list, layout_bbox, span_bbox)
3. MinerU natively saves all intermediates
4. Log summary and timing

### 4. `run_paddleocr.py` — PaddleOCR Runner

**Usage**: `python run_paddleocr.py <pdf_path> [--output-dir <dir>]`

**Logic**:
1. Initialize PPStructureV3 with table+formula recognition
2. Run predict, save per-page: markdown, images, layout JSON, OCR JSON, table HTML
3. Concatenate all pages into single output.md
4. Log summary and timing

### 5. `run_all.py` — Master Orchestrator

**Usage**: `python run_all.py [--pdf-dir pdfs] [--output-dir results]`

**Logic**:
1. Scan pdfs/ recursively for all .pdf files
2. For each PDF × parser: run via `conda run -n {env_name}` in subprocess
3. After each parser: run GPU cleanup script in same conda env
4. Print summary table

**Environment names**: `docling`, `mineru`, `paddleocr`

## Verification

```bash
# Unit test each runner
conda activate docling && python pdf_parser_arena/run_docling.py pdfs/papers/Corvid.pdf
conda activate mineru && python pdf_parser_arena/run_mineru.py pdfs/papers/Corvid.pdf
conda activate paddleocr && python pdf_parser_arena/run_paddleocr.py pdfs/papers/Corvid.pdf

# Full orchestration
python pdf_parser_arena/run_all.py --pdf-dir pdfs
```
