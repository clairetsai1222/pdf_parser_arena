"""
PaddleOCR PDF Parser Runner.

Converts a PDF to Markdown using PaddleOCR PPStructureV3, saving all
intermediate results (per-page markdown, layout detections, OCR results,
table HTML, images) and detailed logs.

Usage:
    python run_paddleocr.py <pdf_path> [--output-dir <dir>]
"""

import json
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    cleanup_gpu,
    create_output_dir,
    parse_args_single_pdf,
    save_json,
    setup_logger,
    TimingContext,
)


def _serialize_ocr_result(ocr_res):
    """Convert OCR result arrays (numpy) to JSON-serializable lists."""
    if not ocr_res:
        return ocr_res
    out = {}
    for k, v in ocr_res.items():
        try:
            if hasattr(v, "tolist"):
                out[k] = v.tolist()
            elif isinstance(v, list):
                out[k] = [
                    item.tolist() if hasattr(item, "tolist") else item for item in v
                ]
            else:
                out[k] = v
        except Exception:
            out[k] = str(v)
    return out


def _serialize_layout_dets(layout_dets):
    """Convert layout detection results to JSON-serializable format."""
    if not layout_dets:
        return layout_dets
    results = []
    for det in layout_dets:
        item = {}
        for k, v in det.items():
            if hasattr(v, "tolist"):
                item[k] = v.tolist()
            else:
                item[k] = v
        results.append(item)
    return results


def run_paddleocr(pdf_path, output_dir=None):
    """Run PaddleOCR parser on a single PDF with full intermediate output.

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
    out = create_output_dir(pdf_path, "paddleocr", base_dir=output_dir)
    logger = setup_logger("paddleocr", out / "run.log")
    logger.info("=" * 60)
    logger.info("PaddleOCR runner started")
    logger.info("PDF: %s", pdf_path)
    logger.info("Output: %s", out)

    # Set PaddleOCR logging to verbose
    try:
        from paddleocr._utils.logging import logger as paddle_logger
        paddle_logger.setLevel(logging.DEBUG)
        # Add our file handler to PaddleOCR's logger
        fh = logging.FileHandler(str(out / "run.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        paddle_logger.addHandler(fh)
    except Exception as e:
        logger.warning("Could not configure PaddleOCR logger: %s", e)

    # Create subdirectories
    per_page_dir = out / "per_page"
    layout_dir = out / "layout"
    ocr_dir = out / "ocr"
    tables_dir = out / "tables"
    for d in [per_page_dir, layout_dir, ocr_dir, tables_dir]:
        d.mkdir(exist_ok=True)

    timings = {}

    try:
        # -- Initialize pipeline --
        with TimingContext("init_pipeline", timings, logger):
            from paddleocr import PPStructureV3
            pipeline = PPStructureV3(
                use_table_recognition=True,
                use_formula_recognition=True,
            )
            logger.info("PPStructureV3 initialized")

        # -- Run prediction --
        with TimingContext("conversion", timings, logger):
            results = pipeline.predict(str(pdf_path))
            logger.info("Prediction completed: %d page(s)", len(results))

        # -- Process per-page results --
        all_markdown_pages = []

        with TimingContext("save_intermediates", timings, logger):
            for page_idx, page_res in enumerate(results):
                page_num = page_idx + 1
                page_prefix = f"page_{page_num:03d}"
                logger.info("Processing page %d / %d", page_num, len(results))

                # Inspect available keys
                available_keys = list(page_res.keys()) if isinstance(page_res, dict) else []
                logger.info("  Page %d keys: %s", page_num, available_keys)

                # -- Save markdown --
                md_text = ""
                if isinstance(page_res, dict) and "markdown" in page_res:
                    md_info = page_res["markdown"]
                    if isinstance(md_info, dict):
                        md_text = md_info.get("markdown_texts", "")
                        md_images = md_info.get("markdown_images", {})
                    else:
                        md_text = str(md_info)
                        md_images = {}

                    # Save per-page markdown
                    (per_page_dir / f"{page_prefix}.md").write_text(
                        md_text, encoding="utf-8"
                    )
                    all_markdown_pages.append(md_text)

                    # Save images referenced in markdown
                    if md_images:
                        img_dir = per_page_dir / page_prefix
                        img_dir.mkdir(exist_ok=True)
                        for img_key, img_obj in md_images.items():
                            try:
                                img_path = img_dir / f"{img_key}.png"
                                if hasattr(img_obj, "save"):
                                    img_obj.save(str(img_path))
                                else:
                                    logger.debug("  Image %s is not a PIL image", img_key)
                            except Exception as e:
                                logger.warning("  Failed to save image %s: %s", img_key, e)
                        logger.info("  Saved %d images for page %d", len(md_images), page_num)

                # -- Save layout detections --
                if isinstance(page_res, dict) and "layout_det" in page_res:
                    layout_data = _serialize_layout_dets(page_res["layout_det"])
                    save_json(layout_data, layout_dir / f"{page_prefix}_layout.json")
                    logger.info("  Layout detections: %d regions", len(layout_data) if layout_data else 0)

                # -- Save OCR results --
                if isinstance(page_res, dict) and "overall_ocr_res" in page_res:
                    ocr_data = _serialize_ocr_result(page_res["overall_ocr_res"])
                    save_json(ocr_data, ocr_dir / f"{page_prefix}_ocr.json")
                    text_count = len(ocr_data.get("rec_texts", []))
                    logger.info("  OCR text segments: %d", text_count)

                # -- Save table results --
                if isinstance(page_res, dict) and "table_res_list" in page_res:
                    for t_idx, table_html in enumerate(page_res["table_res_list"]):
                        table_path = tables_dir / f"{page_prefix}_table_{t_idx + 1:02d}.html"
                        if isinstance(table_html, dict):
                            table_str = table_html.get("html", str(table_html))
                        else:
                            table_str = str(table_html)
                        table_path.write_text(table_str, encoding="utf-8")
                    logger.info("  Tables: %d", len(page_res["table_res_list"]))

                # -- Save raw page result keys/metadata --
                page_meta = {}
                for k in available_keys:
                    v = page_res[k]
                    if isinstance(v, (str, int, float, bool, type(None))):
                        page_meta[k] = v
                    elif isinstance(v, dict):
                        page_meta[k] = f"<dict with {len(v)} keys>"
                    elif isinstance(v, list):
                        page_meta[k] = f"<list with {len(v)} items>"
                    else:
                        page_meta[k] = f"<{type(v).__name__}>"
                save_json(page_meta, per_page_dir / f"{page_prefix}_meta.json")

        # -- Concatenate all pages into single markdown --
        with TimingContext("concatenate_markdown", timings, logger):
            full_md = "\n\n---\n\n".join(all_markdown_pages)
            (out / "output.md").write_text(full_md, encoding="utf-8")
            logger.info("Saved concatenated markdown: %d pages, %d chars",
                        len(all_markdown_pages), len(full_md))

        # -- Close pipeline --
        try:
            pipeline.close()
            logger.info("Pipeline closed")
        except Exception:
            pass

        # -- Save arena timings --
        save_json(timings, out / "arena_timings.json")

        logger.info("=" * 60)
        logger.info("PaddleOCR runner completed successfully")
        logger.info("Timings: %s", timings)
        return out

    except Exception:
        logger.error("PaddleOCR runner FAILED:\n%s", traceback.format_exc())
        save_json(timings, out / "arena_timings.json")
        return None

    finally:
        cleanup_gpu("paddleocr", logger)


if __name__ == "__main__":
    args = parse_args_single_pdf("Run PaddleOCR PDF parser with full intermediate output")
    result_dir = run_paddleocr(args.pdf_path, args.output_dir)
    sys.exit(0 if result_dir else 1)
