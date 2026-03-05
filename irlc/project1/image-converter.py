import argparse
import sys
from pathlib import Path

def pdf_to_images(pdf_path: str, output_dir: str = None, fmt: str = "png", dpi: int = 150):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image is not installed.")
        print("Install it with: pip install pdf2image pillow")
        sys.exit(1)

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Error: Input file must be a PDF, got: {pdf_path.suffix}")
        sys.exit(1)

    # Determine output directory
    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {pdf_path}")
    print(f"Output dir: {output_dir}")
    print(f"Format: {fmt.upper()}  |  DPI: {dpi}")

    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("Make sure poppler is installed on your system.")
        sys.exit(1)

    total = len(pages)
    pad = len(str(total))  # zero-pad width based on page count

    saved = []
    for i, page in enumerate(pages, start=1):
        filename = f"page_{str(i).zfill(pad)}.{fmt}"
        out_path = output_dir / filename

        # jpeg doesn't support RGBA, convert if necessary
        if fmt.lower() in ("jpg", "jpeg") and page.mode == "RGBA":
            page = page.convert("RGB")

        page.save(str(out_path), fmt.upper() if fmt.lower() != "jpg" else "JPEG")
        saved.append(out_path)
        print(f"  [{i}/{total}] Saved {filename}")

    print(f"\nDone! {total} image(s) saved to: {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Convert each page of a PDF to an individual image file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory to save images (default: <pdf_name>/ next to the PDF)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "tiff"],
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in DPI (default: 150; use 300 for high quality)",
    )

    args = parser.parse_args()
    pdf_to_images(args.pdf, args.output_dir, args.format, args.dpi)


if __name__ == "__main__":
    main()