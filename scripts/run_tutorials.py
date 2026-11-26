#!/usr/bin/env python
"""Run the bundled visualization notebooks without changing their saved originals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


def check_data() -> int:
    data = TUTORIALS / "data"
    records = json.loads((data / "provenance.json").read_text())["files"]
    if not records:
        raise ValueError("The example-data manifest is empty")
    for name, record in records.items():
        if Path(name).name != name:
            raise ValueError(f"Invalid data filename: {name}")
        file = data / name
        if not file.is_file():
            raise FileNotFoundError(f"Missing bundled example data: {name}")
        if file.stat().st_size != record["bytes"]:
            raise ValueError(f"Example-data size mismatch: {name}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Example-data checksum mismatch: {name}")
    return len(records)


def main() -> int:
    catalog = json.loads((TUTORIALS / "catalog.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Execute all tutorials")
    selection.add_argument("--notebook", choices=[item["id"] for item in catalog],
                           action="append", help="Select a tutorial; may be repeated")
    parser.add_argument("--list", action="store_true", help="List available tutorials")
    parser.add_argument("--check-data", action="store_true", help="Verify bundled data only, unless a tutorial is selected")
    parser.add_argument("--output-dir", type=Path, default=Path("tutorials/outputs/executed"),
                        help="Executed-copy destination, relative to repository root")
    parser.add_argument("--html", action="store_true", help="Also export self-contained HTML")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per code cell")
    args = parser.parse_args()
    if args.list:
        for item in catalog:
            print(f"{item['id']:12} {item['title']}  [{item['notebook']}]")
        return 0
    if not args.all and not args.notebook and not args.check_data:
        parser.print_help()
        return 0
    try:
        count = check_data()
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Data check failed: {error}\n")
    print(f"Verified {count} example-data files", flush=True)
    if args.check_data and not args.all and not args.notebook:
        return 0
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        import nbformat
        from nbclient import NotebookClient
        if args.html:
            from nbconvert import HTMLExporter
    except ImportError as error:
        parser.exit(1, f"Missing tutorial dependency: {error}.\n"
                    "Run: python -m pip install -r tutorials/requirements.txt\n")

    output = (ROOT / args.output_dir).resolve()
    selected = [item for item in catalog if args.all or item["id"] in args.notebook]
    destinations = [output / item["notebook"] for item in selected]
    protected = {(TUTORIALS / item["notebook"]).resolve() for item in catalog}
    if any(path.resolve() in protected for path in destinations):
        parser.error("Choose an output directory different from the original tutorials directory")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(TUTORIALS / "outputs/.matplotlib"))
    # Some cluster shells export nonnumeric values that prevent library startup.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if not os.environ.get(variable, "").isdigit() or int(os.environ[variable]) < 1:
            os.environ[variable] = "1"

    report = {"python": platform.python_version(), "data_files_checked": count,
              "kernel": args.kernel, "notebooks": []}
    failed = False
    for item in selected:
        started = time.monotonic()
        notebook = nbformat.read(TUTORIALS / item["notebook"], as_version=4)
        record = {"id": item["id"], "notebook": item["notebook"]}
        print(f"Executing {item['notebook']} ...", flush=True)
        try:
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            NotebookClient(notebook, timeout=args.timeout, kernel_name=args.kernel,
                           resources={"metadata": {"path": str(TUTORIALS)}}).execute()
            nbformat.validate(notebook)
            outputs = [result for cell in notebook.cells if cell.cell_type == "code"
                       for result in cell.outputs]
            images = sum("image/png" in result.get("data", {}) for result in outputs)
            if any(result.output_type == "error" for result in outputs):
                raise RuntimeError("Notebook contains an error output")
            if images < item["minimum_figures"]:
                raise RuntimeError(f"Expected at least {item['minimum_figures']} figures; found {images}")
            if args.html:
                html, _ = HTMLExporter().from_notebook_node(notebook)
                (output / Path(item["notebook"]).with_suffix(".html")).write_text(html, encoding="utf-8")
            record.update(status="passed", figures=images)
            print(f"  Passed: {images} saved figures", flush=True)
        except Exception as error:
            failed = True
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"  Failed: {error}", file=sys.stderr, flush=True)
        finally:
            nbformat.write(notebook, output / item["notebook"])
            record["seconds"] = round(time.monotonic() - started, 2)
            report["notebooks"].append(record)
            (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Executed copies and run summary: {output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
