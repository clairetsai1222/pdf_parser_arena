"""
Docling PDF Parser Runner.

Converts a PDF to Markdown/JSON/HTML using IBM Docling, saving all
intermediate results (page images, table exports, layout visualizations,
profiling data) and detailed logs.

Usage:
    python run_docling.py <pdf_path> [--output-dir <dir>]
"""

import logging
import sys
import traceback
from pathlib import Path

# Ensure the arena package is importable
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    cleanup_gpu,
    create_output_dir,
    parse_args_single_pdf,
    save_json,
    setup_logger,
    TimingContext,
)


def run_docling(pdf_path, output_dir=None):
    """Run Docling parser on a single PDF with full intermediate output.

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
    out = create_output_dir(pdf_path, "docling", base_dir=output_dir)
    logger = setup_logger("docling", out / "run.log")
    logger.info("=" * 60)
    logger.info("Docling runner started")
    logger.info("PDF: %s", pdf_path)
    logger.info("Output: %s", out)

    timings = {}

    try:
        # -- Configure debug output before importing heavy modules --
        with TimingContext("import_and_config", timings, logger):
            from docling.datamodel.settings import settings

            debug_dir = out / "debug"
            debug_dir.mkdir(exist_ok=True)
            settings.debug.visualize_layout = True
            settings.debug.visualize_ocr = True
            settings.debug.visualize_tables = True
            settings.debug.visualize_cells = True
            settings.debug.profile_pipeline_timings = True
            settings.debug.debug_output_path = str(debug_dir)

            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 2.0
            pipeline_options.do_table_structure = True
            pipeline_options.do_ocr = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
            logger.info("Docling configured successfully")

        # -- Run conversion --
        with TimingContext("conversion", timings, logger):
            result = converter.convert(str(pdf_path))

        logger.info("Conversion status: %s", result.status)

        if result.errors:
            for err in result.errors:
                logger.error("Conversion error: %s", err)

        # -- Save final outputs --
        with TimingContext("save_outputs", timings, logger):
            from docling_core.types.doc import ImageRefMode

            # Markdown
            md_path = out / "output.md"
            result.document.save_as_markdown(
                md_path, image_mode=ImageRefMode.PLACEHOLDER
            )
            logger.info("Saved markdown: %s", md_path)

            # JSON
            json_path = out / "output.json"
            result.document.save_as_json(
                json_path, image_mode=ImageRefMode.PLACEHOLDER
            )
            logger.info("Saved JSON: %s", json_path)

            # HTML
            html_path = out / "output.html"
            result.document.save_as_html(
                html_path, image_mode=ImageRefMode.PLACEHOLDER
            )
            logger.info("Saved HTML: %s", html_path)

        # -- Save page images --
        with TimingContext("save_page_images", timings, logger):
            pages_dir = out / "pages"
            pages_dir.mkdir(exist_ok=True)
            page_count = 0
            for page_no, page in result.document.pages.items():
                try:
                    if page.image and page.image.pil_image:
                        img_path = pages_dir / f"page_{page_no:03d}.png"
                        page.image.pil_image.save(str(img_path))
                        page_count += 1
                except Exception as e:
                    logger.warning("Could not save page %s image: %s", page_no, e)
            logger.info("Saved %d page images", page_count)

        # -- Save table and figure images --
        with TimingContext("save_tables_figures", timings, logger):
            from docling_core.types.doc import PictureItem, TableItem

            tables_dir = out / "tables"
            tables_dir.mkdir(exist_ok=True)
            figures_dir = out / "figures"
            figures_dir.mkdir(exist_ok=True)

            table_idx = 0
            figure_idx = 0
            for element, _level in result.document.iterate_items():
                if isinstance(element, TableItem):
                    table_idx += 1
                    # Save table image
                    try:
                        img = element.get_image(result.document)
                        if img:
                            img.save(str(tables_dir / f"table_{table_idx:03d}.png"))
                    except Exception as e:
                        logger.warning("Could not save table %d image: %s", table_idx, e)
                    # Save table as CSV
                    try:
                        df = element.export_to_dataframe(doc=result.document)
                        df.to_csv(
                            tables_dir / f"table_{table_idx:03d}.csv",
                            index=False,
                            encoding="utf-8",
                        )
                    except Exception as e:
                        logger.warning("Could not export table %d CSV: %s", table_idx, e)
                    # Save table as HTML
                    try:
                        html = element.export_to_html(doc=result.document)
                        (tables_dir / f"table_{table_idx:03d}.html").write_text(
                            html, encoding="utf-8"
                        )
                    except Exception as e:
                        logger.warning("Could not export table %d HTML: %s", table_idx, e)

                elif isinstance(element, PictureItem):
                    figure_idx += 1
                    try:
                        img = element.get_image(result.document)
                        if img:
                            img.save(str(figures_dir / f"figure_{figure_idx:03d}.png"))
                    except Exception as e:
                        logger.warning("Could not save figure %d: %s", figure_idx, e)

            logger.info("Saved %d tables, %d figures", table_idx, figure_idx)

        # -- Save profiling data --
        with TimingContext("save_profiling", timings, logger):
            profiling_data = {}
            for stage_name, timing_item in result.timings.items():
                profiling_data[stage_name] = {
                    "count": timing_item.count,
                    "total_s": timing_item.total(),
                    "avg_s": timing_item.avg(),
                }
            save_json(profiling_data, out / "profiling.json")
            logger.info("Profiling stages: %s", list(profiling_data.keys()))

        # -- Save arena timings --
        save_json(timings, out / "arena_timings.json")

        logger.info("=" * 60)
        logger.info("Docling runner completed successfully")
        logger.info("Timings: %s", timings)
        return out

    except Exception:
        logger.error("Docling runner FAILED:\n%s", traceback.format_exc())
        save_json(timings, out / "arena_timings.json")
        return None

    finally:
        cleanup_gpu("docling", logger)


if __name__ == "__main__":
    args = parse_args_single_pdf("Run Docling PDF parser with full intermediate output")
    result_dir = run_docling(args.pdf_path, args.output_dir)
    sys.exit(0 if result_dir else 1)
