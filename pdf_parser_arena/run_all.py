"""
Master Orchestration Script for PDF Parser Arena.

Scans a directory for PDFs and runs all three parsers (Docling, MinerU,
PaddleOCR) sequentially on each PDF. Each parser runs in its own conda
environment via subprocess, with GPU cleanup between runs.

Usage:
    python run_all.py [--pdf-dir <dir>] [--output-dir <dir>] [--parsers docling mineru paddleocr]
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_pdf_files, save_json, setup_logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARENA_DIR = Path(__file__).parent
DEFAULT_PDF_DIR = ARENA_DIR.parent / "pdfs"
DEFAULT_OUTPUT_DIR = ARENA_DIR / "results"

# Parser name -> conda env name
PARSER_CONFIGS = {
    "docling": {
        "env": "docling",
        "script": "run_docling.py",
        "cleanup_framework": "torch",
    },
    "mineru": {
        "env": "mineru",
        "script": "run_mineru.py",
        "cleanup_framework": "torch",
    },
    "paddleocr": {
        "env": "paddleocr",
        "script": "run_paddleocr.py",
        "cleanup_framework": "paddle",
    },
}

# Max time per parser per PDF (seconds)
TIMEOUT_SECONDS = 30 * 60  # 30 minutes


# ---------------------------------------------------------------------------
# GPU cleanup helper
# ---------------------------------------------------------------------------

def _build_cleanup_code(framework):
    """Build a Python one-liner to clear GPU memory for a given framework."""
    lines = ["import gc; gc.collect()"]
    if framework == "torch":
        lines.append(
            "exec('try:\\n import torch\\n"
            " if torch.cuda.is_available():\\n"
            "  torch.cuda.empty_cache()\\n"
            "  torch.cuda.ipc_collect()\\n"
            "except: pass')"
        )
    elif framework == "paddle":
        lines.append(
            "exec('try:\\n import paddle\\n"
            " if paddle.device.is_compiled_with_cuda():\\n"
            "  paddle.device.cuda.empty_cache()\\n"
            "except: pass')"
        )
        lines.append(
            "exec('try:\\n import torch\\n"
            " if torch.cuda.is_available():\\n"
            "  torch.cuda.empty_cache()\\n"
            "except: pass')"
        )
    lines.append("gc.collect()")
    return "; ".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_parser_subprocess(parser_name, pdf_path, output_dir, logger):
    """Run a single parser in its conda environment via subprocess.

    Args:
        parser_name: 'docling', 'mineru', or 'paddleocr'.
        pdf_path: Absolute path to the PDF.
        output_dir: Output directory for this specific PDF+timestamp.
        logger: Logger instance.

    Returns:
        dict with 'success', 'elapsed_s', 'return_code'.
    """
    config = PARSER_CONFIGS[parser_name]
    env_name = config["env"]
    script = ARENA_DIR / config["script"]

    cmd = [
        "conda", "run", "-n", env_name, "--no-capture-output",
        "python", str(script),
        str(pdf_path),
        "--output-dir", str(output_dir),
    ]

    logger.info("-" * 50)
    logger.info("Starting [%s] (env: %s)", parser_name, env_name)
    logger.info("Command: %s", " ".join(cmd))

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            timeout=TIMEOUT_SECONDS,
            capture_output=False,  # Let output flow to console
            cwd=str(ARENA_DIR),
        )
        elapsed = time.perf_counter() - start
        success = result.returncode == 0

        if success:
            logger.info("[%s] completed in %.1fs", parser_name, elapsed)
        else:
            logger.error("[%s] FAILED (return code %d) in %.1fs",
                         parser_name, result.returncode, elapsed)

        return {
            "success": success,
            "elapsed_s": round(elapsed, 2),
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        logger.error("[%s] TIMED OUT after %.1fs", parser_name, elapsed)
        return {
            "success": False,
            "elapsed_s": round(elapsed, 2),
            "return_code": -1,
            "error": "timeout",
        }

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("[%s] subprocess error: %s", parser_name, e)
        return {
            "success": False,
            "elapsed_s": round(elapsed, 2),
            "return_code": -1,
            "error": str(e),
        }


def run_gpu_cleanup(parser_name, logger):
    """Run GPU cleanup in the parser's conda env after it finishes.

    Since the parser runs as a subprocess, its GPU memory is mostly freed
    on exit. This is an extra safety measure to release any cached memory.
    """
    config = PARSER_CONFIGS[parser_name]
    cleanup_code = _build_cleanup_code(config["cleanup_framework"])

    cmd = [
        "conda", "run", "-n", config["env"],
        "python", "-c", cleanup_code,
    ]

    try:
        subprocess.run(cmd, timeout=30, capture_output=True)
        logger.info("[%s] GPU cleanup done", parser_name)
    except Exception as e:
        logger.warning("[%s] GPU cleanup failed: %s", parser_name, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all PDF parsers on all PDFs in a directory"
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(DEFAULT_PDF_DIR),
        help=f"Directory containing PDF files (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--parsers",
        nargs="+",
        default=list(PARSER_CONFIGS.keys()),
        choices=list(PARSER_CONFIGS.keys()),
        help="Which parsers to run (default: all)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Setup orchestrator log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        "orchestrator",
        output_base / f"orchestrator_{timestamp}.log",
    )

    logger.info("=" * 60)
    logger.info("PDF Parser Arena - Orchestrator")
    logger.info("PDF directory: %s", pdf_dir)
    logger.info("Output directory: %s", output_base)
    logger.info("Parsers: %s", args.parsers)
    logger.info("Timeout: %ds per parser per PDF", TIMEOUT_SECONDS)

    # Find PDFs
    pdf_files = get_pdf_files(pdf_dir)
    if not pdf_files:
        logger.error("No PDF files found in %s", pdf_dir)
        sys.exit(1)

    logger.info("Found %d PDF(s):", len(pdf_files))
    for p in pdf_files:
        logger.info("  %s", p)

    # Run all parsers on all PDFs
    all_results = {}

    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        pdf_stem = pdf_path.stem
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_base / f"{pdf_stem}_{run_timestamp}"

        logger.info("=" * 60)
        logger.info("PDF [%d/%d]: %s", pdf_idx, len(pdf_files), pdf_path.name)
        logger.info("Run directory: %s", run_dir)

        pdf_results = {}

        for parser_name in args.parsers:
            # Run the parser
            result = run_parser_subprocess(
                parser_name, pdf_path, str(run_dir), logger
            )
            pdf_results[parser_name] = result

            # GPU cleanup between parsers
            run_gpu_cleanup(parser_name, logger)

            # Brief pause to let system stabilize
            time.sleep(2)

        all_results[str(pdf_path)] = pdf_results

    # -- Print summary --
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("%-30s %-12s %-10s %-10s", "PDF", "Parser", "Status", "Time(s)")
    logger.info("-" * 65)
    for pdf_path, parsers in all_results.items():
        pdf_name = Path(pdf_path).name
        for parser_name, result in parsers.items():
            status = "OK" if result["success"] else "FAIL"
            logger.info("%-30s %-12s %-10s %-10.1f",
                        pdf_name, parser_name, status, result["elapsed_s"])

    # Save summary JSON
    save_json(all_results, output_base / f"summary_{timestamp}.json")
    logger.info("Summary saved to: %s", output_base / f"summary_{timestamp}.json")


if __name__ == "__main__":
    main()
