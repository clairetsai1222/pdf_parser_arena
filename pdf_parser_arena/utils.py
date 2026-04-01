"""
Shared utilities for PDF Parser Arena.
Provides logging, output directory management, GPU cleanup, and timing helpers.
"""

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def create_output_dir(pdf_path, parser_name, base_dir=None, timestamp=None):
    """Create a timestamped output directory for a parser run.

    Structure: <base_dir>/<pdf_stem>_<timestamp>/<parser_name>/

    Args:
        pdf_path: Path to the input PDF file.
        parser_name: One of 'docling', 'mineru', 'paddleocr'.
        base_dir: Root results directory. Defaults to <arena>/results.
        timestamp: Shared timestamp string. Auto-generated if None.

    Returns:
        Path to the created parser output directory.
    """
    pdf_path = Path(pdf_path)
    if base_dir is None:
        base_dir = Path(__file__).parent / "results"
    else:
        base_dir = Path(base_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_dir / f"{pdf_path.stem}_{timestamp}" / parser_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name, log_path, level=logging.DEBUG):
    """Create a logger that writes to both a file and stdout.

    Args:
        name: Logger name (use parser name for uniqueness).
        log_path: Path to the log file.
        level: Logging level.

    Returns:
        logging.Logger instance.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class TimingContext:
    """Context manager that records wall-clock time for a named stage.

    Usage:
        timings = {}
        with TimingContext("ocr", timings):
            do_ocr()
        print(timings["ocr"])  # seconds as float
    """

    def __init__(self, stage_name, timings_dict, logger=None):
        self.stage_name = stage_name
        self.timings = timings_dict
        self.logger = logger

    def __enter__(self):
        self.start = time.perf_counter()
        if self.logger:
            self.logger.info("Stage [%s] started", self.stage_name)
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        self.timings[self.stage_name] = round(elapsed, 4)
        if self.logger:
            self.logger.info("Stage [%s] finished in %.2fs", self.stage_name, elapsed)
        return False


# ---------------------------------------------------------------------------
# GPU / memory cleanup
# ---------------------------------------------------------------------------

def cleanup_gpu(parser_name, logger=None):
    """Release GPU memory and run garbage collection.

    Only cleans the current process — does NOT kill other processes.

    Args:
        parser_name: 'docling', 'mineru', or 'paddleocr'.
        logger: Optional logger instance.
    """
    log = logger.info if logger else print

    gc.collect()
    log("gc.collect() done")

    if parser_name in ("docling", "mineru"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                log("torch.cuda.empty_cache() done")
        except ImportError:
            pass

    elif parser_name == "paddleocr":
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                log("paddle.device.cuda.empty_cache() done")
        except ImportError:
            pass
        # PaddleOCR may also use torch internally
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log("torch.cuda.empty_cache() done (paddle env)")
        except ImportError:
            pass

    gc.collect()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_pdf_files(pdf_dir):
    """Recursively find all PDF files under a directory.

    Returns:
        Sorted list of Path objects.
    """
    pdf_dir = Path(pdf_dir)
    return sorted(pdf_dir.rglob("*.pdf"))


def save_json(data, path):
    """Save a dict/list as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def parse_args_single_pdf(description):
    """Common argument parser for individual runner scripts.

    Returns:
        argparse.Namespace with .pdf_path and .output_dir
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (default: pdf_parser_arena/results)",
    )
    return parser.parse_args()
