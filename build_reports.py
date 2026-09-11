"""Build actual LaTeX PDFs with Tectonic or XeLaTeX; verify page bounds."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NAMES = ("algorithm_article_portrait_ru", "experiment_article_portrait_ru")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", help="Path to tectonic.exe or xelatex")
    parser.add_argument("--bundle", help="Optional Tectonic bundle URL or local path")
    args = parser.parse_args()
    engine = args.engine or shutil.which("tectonic") or shutil.which("xelatex")
    if not engine:
        raise SystemExit(
            "Install Tectonic or XeLaTeX, or supply --engine PATH. See README.md."
        )
    # Committed vector figures and LaTeX tables allow rebuilding without MNIST.
    for name in NAMES:
        if "tectonic" in Path(engine).name.lower():
            command = [engine, "--keep-logs", "--keep-intermediates", name + ".tex"]
            if args.bundle:
                command[1:1] = ["--bundle", args.bundle]
            subprocess.run(command, cwd=ROOT / "reports", check=True)
        else:
            for _ in range(2):
                subprocess.run(
                    [
                        engine,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        name + ".tex",
                    ],
                    cwd=ROOT / "reports",
                    check=True,
                )
        shutil.copyfile(ROOT / "reports" / (name + ".pdf"), ROOT / (name + ".pdf"))
    import pymupdf

    check = {"documents": {}}
    for name in NAMES:
        doc = pymupdf.open(ROOT / (name + ".pdf"))
        out = ROOT / "checks" / name
        out.mkdir(parents=True, exist_ok=True)
        issues = []
        for i, page in enumerate(doc):
            page.get_pixmap(matrix=pymupdf.Matrix(1.1, 1.1)).save(
                out / f"page_{i + 1}.png"
            )
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            x0, y0, x1, y1 = span["bbox"]
                            if (
                                x0 < 18
                                or y0 < 10
                                or x1 > page.rect.width - 18
                                or y1 > page.rect.height - 10
                            ):
                                issues.append(
                                    {
                                        "page": i + 1,
                                        "text": span["text"],
                                        "bbox": span["bbox"],
                                    }
                                )
        log = (ROOT / "reports" / (name + ".log")).read_text(
            encoding="utf-8", errors="replace"
        )
        warnings = [
            line
            for line in log.splitlines()
            if "Overfull" in line or "Missing character" in line
        ]
        check["documents"][name] = {
            "pages": len(doc),
            "out_of_bounds": issues,
            "layout_warnings": warnings,
        }
    (ROOT / "checks/pdf_validation.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
