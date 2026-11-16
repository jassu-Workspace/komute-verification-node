"""Build training/finetune_w600k_kaggle.ipynb from finetune_w600k_kaggle.py, inlining komute_ft.py.

    python training/build_notebook.py
Cells are split on "# %%" lines; "# %% [markdown]" cells become Markdown. Needs nbformat.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent


def main() -> None:
    src = (HERE / "finetune_w600k_kaggle.py").read_text(encoding="utf-8")
    module = (HERE / "komute_ft.py").read_text(encoding="utf-8")
    if "'''" in module or module.rstrip().endswith("\\"):
        raise SystemExit("komute_ft.py cannot be inlined as a raw triple-quoted string")
    block = re.compile(r"# BEGIN-MODULE.*?\n.*?# END-MODULE\n", re.S)
    if len(block.findall(src)) != 1:
        raise SystemExit("expected exactly one BEGIN-MODULE/END-MODULE block")
    src = block.sub(lambda _: "MODULE_SOURCE = r'''" + module + "'''\n", src)
    cells = []
    for part in re.split(r"^# %%", src, flags=re.M)[1:]:
        head, _, body = part.partition("\n")
        if head.strip() == "[markdown]":
            md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.strip().splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            code = body.strip("\n")
            compile(code, "<cell>", "exec")  # syntax check only, nothing runs
            cells.append(nbformat.v4.new_code_cell(code))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}})
    nbformat.validate(nb)
    out = HERE / "finetune_w600k_kaggle.ipynb"
    nbformat.write(nb, out)
    print(f"{len(cells)} cells -> {out}")


if __name__ == "__main__":
    main()
