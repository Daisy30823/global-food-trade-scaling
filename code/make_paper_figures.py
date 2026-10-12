from __future__ import annotations

import argparse

from analysis.paper_figures import generate_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-file-per-panel manuscript figures.")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh ignored work/figure_data from the validated analysis outputs.")
    args = parser.parse_args()
    index = generate_paper_figures(refresh_data=args.refresh_data)
    print(f"Generated {len(index)} independent figure panels ({len(index) * 2} PDF/PNG files).")


if __name__ == "__main__":
    main()
