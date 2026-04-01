"""
MinerU PDF Parser Runner.

Converts a PDF to Markdown using OpenDataLab MinerU, saving all
intermediate results (middle JSON, model output, layout/span PDFs,
content lists, images) and detailed logs.

Usage:
    python run_mineru.py <pdf_path> [--output-dir <dir>]
"""

import logging
import os
import sys
import traceback
from pathlib import Path

# Set debug logging BEFORE any MinerU import
os.environ["MINERU_LOG_LEVEL"] = "DEBUG"

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    cleanup_gpu,
    create_output_dir,
    parse_args_single_pdf,
    save_json,
    setup_logger,
    TimingContext,
)


def run_mineru(pdf_path, output_dir=None):
    """Run MinerU parser on a single PDF with full intermediate output.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Base output directory (auto-generated if None).

    Returns:
        Path to the output directory, or None on failure.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return None

    # -- Create output directory --
    out = create_output_dir(pdf_path, "mineru", base_dir=output_dir)
    logger = setup_logger("mineru", out / "run.log")
    logger.info("=" * 60)
    logger.info("MinerU runner started")
    logger.info("PDF: %s", pdf_path)
    logger.info("Output: %s", out)

    # Redirect loguru (MinerU's logger) to our log file as well
    try:
        from loguru import logger as loguru_logger
        loguru_logger.add(
            str(out / "run.log"),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
            filter=lambda record: True,
        )
    except ImportError:
        logger.warning("loguru not available; MinerU internal logs won't be captured to file")

    timings = {}

    try:
        # -- Import and read PDF --
        with TimingContext("import_and_read", timings, logger):
            from mineru.cli.common import do_parse, read_fn

            pdf_bytes = read_fn(str(pdf_path))
            pdf_stem = pdf_path.stem
            logger.info("PDF read: %d bytes", len(pdf_bytes))

        # -- MinerU output goes into a subdirectory it manages --
        mineru_out = out / "mineru_raw"
        mineru_out.mkdir(exist_ok=True)

        # -- Run parsing with all intermediate dumps enabled --
        with TimingContext("conversion", timings, logger):
            do_parse(
                output_dir=str(mineru_out),
                pdf_file_names=[pdf_stem],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["en"],
                backend="pipeline",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                f_draw_layout_bbox=True,    # Save layout visualization PDF
                f_draw_span_bbox=True,      # Save span visualization PDF
                f_dump_md=True,             # Save markdown
                f_dump_middle_json=True,    # Save intermediate JSON
                f_dump_model_output=True,   # Save model output JSON
                f_dump_orig_pdf=True,       # Save original PDF copy
                f_dump_content_list=True,   # Save content list JSON
            )

        logger.info("MinerU parsing completed")

        # -- Log what files were produced --
        with TimingContext("inventory", timings, logger):
            produced = []
            for f in sorted(mineru_out.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(mineru_out)
                    size_kb = f.stat().st_size / 1024
                    produced.append({"file": str(rel), "size_kb": round(size_kb, 1)})
                    logger.info("  Produced: %-50s (%.1f KB)", rel, size_kb)
            save_json(produced, out / "produced_files.json")
            logger.info("Total files produced: %d", len(produced))

        # -- Save arena timings --
        save_json(timings, out / "arena_timings.json")

        logger.info("=" * 60)
        logger.info("MinerU runner completed successfully")
        logger.info("Timings: %s", timings)
        return out

    except Exception:
        logger.error("MinerU runner FAILED:\n%s", traceback.format_exc())
        save_json(timings, out / "arena_timings.json")
        return None

    finally:
        cleanup_gpu("mineru", logger)


if __name__ == "__main__":
    args = parse_args_single_pdf("Run MinerU PDF parser with full intermediate output")
    result_dir = run_mineru(args.pdf_path, args.output_dir)
    sys.exit(0 if result_dir else 1)
