# -*- coding: utf-8 -*-
"""Surgical editor for word/document.xml.

Replaces the visible text of a paragraph (or a table cell) while leaving the
paragraph's own formatting, the MDPI template, styles, numbering and embedded
images untouched.
"""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

PARA_RE = re.compile(r"<w:p(?: [^>]*)?>.*?</w:p>", re.S)
TEXT_RE = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)
RUN_RE = re.compile(r"<w:r(?: [^>]*)?>.*?</w:r>", re.S)
RPR_RE = re.compile(r"<w:rPr>.*?</w:rPr>", re.S)
PPR_RE = re.compile(r"<w:pPr>.*?</w:pPr>", re.S)


def para_text(p: str) -> str:
    return "".join(TEXT_RE.findall(p))


def _new_run(rpr: str, text: str) -> str:
    return (f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>")


def set_para_text(p: str, new_text: str) -> str:
    """Return the paragraph with all its runs replaced by one run of new_text."""
    ppr_m = PPR_RE.search(p)
    ppr = ppr_m.group(0) if ppr_m else ""
    runs = RUN_RE.findall(p)
    rpr = ""
    for r in runs:
        if "<w:t" in r:
            m = RPR_RE.search(r)
            rpr = m.group(0) if m else ""
            break
    open_tag = p[: p.index(">") + 1]
    return f"{open_tag}{ppr}{_new_run(rpr, new_text)}</w:p>"


class Doc:
    def __init__(self, path: str) -> None:
        self.path = path
        self.xml = open(path, encoding="utf-8").read()
        self.log: list[tuple[str, str]] = []

    # -- paragraph-level ------------------------------------------------
    def find_paras(self, needle: str) -> list[tuple[int, int, str]]:
        out = []
        for m in PARA_RE.finditer(self.xml):
            if needle in para_text(m.group(0)):
                out.append((m.start(), m.end(), m.group(0)))
        return out

    def replace_para(self, needle: str, new_text: str, occurrence: int = 0,
                     must_be_unique: bool = True) -> None:
        hits = self.find_paras(needle)
        if not hits:
            raise KeyError(f"paragraph not found: {needle!r}")
        if must_be_unique and len(hits) > 1:
            raise ValueError(f"{len(hits)} paragraphs match {needle!r}; pass must_be_unique=False")
        s, e, p = hits[occurrence]
        self.xml = self.xml[:s] + set_para_text(p, new_text) + self.xml[e:]
        self.log.append(("para", needle[:60]))

    def replace_text(self, old: str, new: str, count: int = 0) -> int:
        """Replace a literal string inside <w:t> content (runs already merged)."""
        old_e, new_e = escape(old), escape(new)
        n = self.xml.count(old_e)
        if n == 0:
            raise KeyError(f"text not found: {old!r}")
        self.xml = self.xml.replace(old_e, new_e) if count == 0 else \
            self.xml.replace(old_e, new_e, count)
        self.log.append(("text", f"{old[:40]} -> {new[:40]}"))
        return n

    # -- table-level ----------------------------------------------------
    def table_cells(self, table_index: int) -> list[list[str]]:
        tbls = re.findall(r"<w:tbl>.*?</w:tbl>", self.xml, re.S)
        t = tbls[table_index]
        rows = re.findall(r"<w:tr(?: [^>]*)?>.*?</w:tr>", t, re.S)
        return [[para_text(c) for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)] for r in rows]

    def set_table(self, table_index: int, values: list[list[str]]) -> None:
        """Rewrite every cell of a table; `values` must match its shape."""
        tbls = list(re.finditer(r"<w:tbl>.*?</w:tbl>", self.xml, re.S))
        m = tbls[table_index]
        t = m.group(0)
        rows = list(re.finditer(r"<w:tr(?: [^>]*)?>.*?</w:tr>", t, re.S))
        if len(rows) != len(values):
            raise ValueError(f"table {table_index} has {len(rows)} rows, got {len(values)}")
        new_t, cursor = [], 0
        for r_m, row_vals in zip(rows, values):
            new_t.append(t[cursor:r_m.start()])
            r = r_m.group(0)
            cells = list(re.finditer(r"<w:tc>.*?</w:tc>", r, re.S))
            if len(cells) != len(row_vals):
                raise ValueError(f"row has {len(cells)} cells, got {len(row_vals)}")
            new_r, c_cursor = [], 0
            for c_m, val in zip(cells, row_vals):
                new_r.append(r[c_cursor:c_m.start()])
                c = c_m.group(0)
                paras = list(PARA_RE.finditer(c))
                if paras:
                    first = paras[0]
                    rebuilt = (c[:first.start()] + set_para_text(first.group(0), val)
                               + c[paras[-1].end():])
                else:
                    rebuilt = c
                new_r.append(rebuilt)
                c_cursor = c_m.end()
            new_r.append(r[c_cursor:])
            new_t.append("".join(new_r))
            cursor = r_m.end()
        new_t.append(t[cursor:])
        self.xml = self.xml[:m.start()] + "".join(new_t) + self.xml[m.end():]
        self.log.append(("table", str(table_index)))

    def insert_para_after(self, needle: str, new_text: str) -> None:
        """Clone the paragraph containing `needle` and insert a copy after it,
        carrying the same style, then set the copy's text."""
        hits = self.find_paras(needle)
        if not hits:
            raise KeyError(f"paragraph not found: {needle!r}")
        s, e, p = hits[0]
        clone = set_para_text(p, new_text)
        self.xml = self.xml[:e] + clone + self.xml[e:]
        self.log.append(("insert", needle[:60]))

    def save(self) -> None:
        open(self.path, "w", encoding="utf-8").write(self.xml)
