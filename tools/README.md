# Document-generation tools

These scripts write the reported numbers from `results/` into the manuscript and
cover letter, so that no value in either document is transcribed by hand. They
are provided for transparency about how the submitted documents were produced.

**They are not part of the reproduction pipeline** (`scripts/reproduce_all.py`)
and are not needed to reproduce any scientific result. They require the authors'
`.docx` template files, which are the manuscript itself and are therefore **not
redistributed in this repository**. Without those templates the scripts cannot
run; the code is included so that a reader can verify what they do.

| File | Purpose |
|---|---|
| `edit_doc.py` | minimal `word/document.xml` editor: replaces the visible text of a paragraph or a table cell while preserving the template, styles, numbering and embedded images |
| `update_manuscript.py` | reads `results/metrics/*.csv` and `results/tables/*.csv` and writes every number, table and figure into the manuscript |
| `update_cover_letter.py` | same, for the cover letter |
| `make_korean_review.py` | builds a Korean-language review copy of the manuscript and cover letter from the same result CSVs; for internal supervisor review only — the English `.docx` is the version of record |

## Usage

```bash
# 1. unpack the manuscript template next to these scripts
cd tools
unzip -q <manuscript_template>.docx -d unpacked

# 2. write the current results into it
python update_manuscript.py          # -> tools/Atmosphere_eng_v9_persistence_lit.docx

# 3. cover letter (template path via environment variable or tools/cover_letter_template.docx)
COVER_LETTER_TEMPLATE=<path>.docx python update_cover_letter.py

# 4. Korean review copies (needs the figure PNGs extracted from the manuscript)
FIGDIR=<dir with image1.png ... image7.png> python make_korean_review.py
```

`make_korean_review.py` builds the Korean documents from scratch rather than
editing a template, and reads every number from the same result CSVs as the
English builder, so the two language versions cannot drift apart numerically.

## Verifying the output

`scripts/audit_consistency.py` cross-checks a finished `.docx` against
`configs/default.yaml` and `results/metrics/`:

```bash
python scripts/audit_consistency.py --manuscript tools/Atmosphere_eng_v9_persistence_lit.docx
```

Note that `scripts/reproduce_all.py` runs `audit_consistency.py` **without** a
manuscript argument, so the single-command pipeline checks configuration values
and generated result files only. Manuscript-level consistency is a separate,
manual invocation with the `--manuscript` flag.
