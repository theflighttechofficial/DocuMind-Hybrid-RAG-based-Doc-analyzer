"""Layout-aware extraction for PDF, DOCX and TXT.

Every format is turned into the same structure: an ordered list of blocks,
each with its page, kind (heading | paragraph | table), text, bounding box
(PDF points, top-left origin; None when the format has no geometry), font
size, heading level and section path. Page text is rebuilt from the cleaned
blocks, so repeated headers/footers and bare page numbers never reach
chunking, NLP or the viewer.

PDF   PyMuPDF text spans (font size/weight -> headings), tables from ruling
      lines or from "Table N:" captions + word columns, header/footer removal,
      de-hyphenation, OCR for pages without a text layer.
DOCX  Real heading styles (Title, Heading 1..9) and Word tables; page numbers
      come from Word's rendered page breaks when the file has them.
TXT   Paragraphs split on blank lines; short title-like lines become headings.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_WS = re.compile(r"[ \t    ]+")
_NUMBERED = re.compile(r"^((?:\d+\.)*\d+)\.?\s+(\S.*)$")
_CAPTION = re.compile(r"^\s*table\s+\d+\s*[:.]", re.I)
_ROMAN_OR_NUM = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlc]{1,6})(?:\s*(?:/|of)\s*\d{1,4})?$", re.I)


@dataclass
class Block:
    page: int
    kind: str  # heading | paragraph | table
    text: str
    bbox: list[float] | None = None
    font_size: float | None = None
    level: int | None = None  # headings only: 0 = document title, 1 = chapter ...
    section_path: list[str] = field(default_factory=list)
    table: dict | None = None  # {"rows": [[...]], "caption": str}
    char_start: int = 0  # offsets into the page text
    char_end: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractedDocument:
    pages: list[tuple[int, str]]
    blocks: list[Block]
    toc: list[dict]  # [{"title", "level", "page", "path"}]
    title: str | None
    page_sizes: dict[int, tuple[float, float]]
    ocr_pages: list[int]
    notes: list[str]


# ----------------------------------------------------------------- shared

def normalize_ws(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = _WS.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _dehyphenate_lines(lines: list[str], vocab: Counter) -> str:
    """Joins lines; a trailing '-' is removed when the joined word is a real word
    in this document (or neither half is), kept when it's a compound whose
    halves are both words used elsewhere ("sequence-aligned")."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        m = re.search(r"([A-Za-z]+)-$", out)
        n = re.match(r"([a-z][A-Za-z]*)", line)
        if m and n:
            left, right = m.group(1), n.group(1)
            joined = (left + right).lower()
            compound = f"{left}-{right}".lower()
            if vocab.get(joined, 0) > 0:
                keep = False
            elif vocab.get(compound, 0) > 0:
                keep = True
            else:
                keep = len(left) >= 3 and len(right) >= 3 and vocab.get(left.lower(), 0) > 0 and vocab.get(right.lower(), 0) > 0
            out = out + line if keep else out[:-1] + line
        else:
            out += " " + line
    return out


def _vocab(texts) -> Counter:
    c = Counter()
    for t in texts:
        c.update(w.lower() for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", t))
        c.update(w.lower() for w in re.findall(r"[A-Za-z]+", t))
    return c


_NUMERIC_CELL = re.compile(r"^[\s(]*[-+]?\d[\d.,·×x%\s^eE()+-]*[)\s]*$|^\d+[KMB]$")


def _merge_header_rows(rows: list[list[str]]) -> list[list[str]]:
    """PDF tables often have 2-3 stacked header rows with spanning cells
    ("BLEU" over "EN-DE | EN-FR"). Collapse every row above the first data
    row (one with a numeric cell) into a single header, carrying a spanning
    cell's label right across the blank cells it covers."""
    first_data = next((i for i, r in enumerate(rows) if any(_NUMERIC_CELL.match(c) for c in r[1:] if c)), None)
    if not first_data or first_data < 2:
        return rows
    width = len(rows[0])
    header = [""] * width
    for r in rows[:first_data]:
        filled = [i for i, c in enumerate(r) if c]
        spread = list(r)
        # sparse row = spanning labels: fill each label rightwards until the next label
        if 0 < len(filled) < width / 2 + 1:
            for j, i in enumerate(filled):
                if i == 0:
                    continue  # the row-label column never spans
                end = filled[j + 1] if j + 1 < len(filled) else width
                for k in range(i + 1, end):
                    if any(rows[x][k] for x in range(first_data) if x != rows.index(r)):
                        spread[k] = r[i]
        header = [(h + " " + c).strip() if c else h for h, c in zip(header, spread)]
    return [header] + rows[first_data:]


def rows_to_markdown(rows: list[list[str]]) -> str:
    rows = [[normalize_ws(c or "").replace("\n", " ").replace("|", "\\|") for c in r] for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = _merge_header_rows(rows)
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _assign_sections(blocks: list[Block]) -> tuple[list[dict], str | None]:
    """Walks blocks in reading order keeping a heading stack, so every block
    gets its chapter > section > subsection path."""
    stack: list[tuple[int, str]] = []
    toc, title = [], None
    for b in blocks:
        if b.kind == "heading":
            if b.level == 0:
                title = title or b.text
                b.section_path = []
                continue
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            b.section_path = [t for _, t in stack]
            toc.append({"title": b.text, "level": b.level, "page": b.page, "path": list(b.section_path)})
        else:
            b.section_path = [t for _, t in stack]
    return toc, title


def _build_pages(blocks: list[Block], page_numbers: list[int]) -> list[tuple[int, str]]:
    by_page: dict[int, list[Block]] = {p: [] for p in page_numbers}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    pages = []
    for p in sorted(by_page):
        parts, offset = [], 0
        for b in by_page[p]:
            if parts:
                offset += 2
            b.char_start = offset
            b.char_end = offset + len(b.text)
            offset = b.char_end
            parts.append(b.text)
        pages.append((p, "\n\n".join(parts)))
    return pages


def extract_document(file_path: str, file_type: str) -> ExtractedDocument:
    if file_type == "pdf":
        return _extract_pdf(file_path)
    if file_type == "docx":
        return _extract_docx(file_path)
    if file_type == "txt":
        return _extract_txt(file_path)
    raise ValueError(f"Unsupported file type: {file_type}")


def extract_pages(file_path: str, file_type: str) -> list[tuple[int, str]]:
    """Backwards-compatible (page_number, text) view."""
    return extract_document(file_path, file_type).pages


# ----------------------------------------------------------------- OCR

def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def ocr_engine() -> str | None:
    """The OCR engine that will actually be used, or None."""
    choice = get_settings().ocr_engine
    if choice in ("auto", "tesseract") and _tesseract_available():
        return "tesseract"
    if choice in ("auto", "easyocr") and _easyocr_available():
        return "easyocr"
    return None


def _ocr_page(page, engine: str) -> list[tuple[str, list[float]]]:
    """Returns [(line_text, bbox_in_pdf_points)] for one PyMuPDF page."""
    import numpy as np

    zoom = 2.0  # 144 dpi: good accuracy/speed trade-off on CPU
    pix = page.get_pixmap(matrix=__import__("fitz").Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]

    lines = []
    if engine == "tesseract":
        import pytesseract
        from PIL import Image
        data = pytesseract.image_to_data(Image.fromarray(img), output_type=pytesseract.Output.DICT)
        groups: dict[tuple, list[int]] = {}
        for i, word in enumerate(data["text"]):
            if word.strip():
                groups.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
        for idx in groups.values():
            x0 = min(data["left"][i] for i in idx); y0 = min(data["top"][i] for i in idx)
            x1 = max(data["left"][i] + data["width"][i] for i in idx); y1 = max(data["top"][i] + data["height"][i] for i in idx)
            lines.append((" ".join(data["text"][i] for i in idx), [x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom]))
    else:
        from app.core.model_loader import load_model
        import easyocr
        reader = load_model("ocr", [("easyocr-en", lambda: easyocr.Reader(["en"], gpu=False, verbose=False))])
        for box, text, _conf in reader.readtext(img, paragraph=False):
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            lines.append((text, [min(xs) / zoom, min(ys) / zoom, max(xs) / zoom, max(ys) / zoom]))
    lines.sort(key=lambda l: (round(l[1][1] / 6), l[1][0]))
    return lines


# ----------------------------------------------------------------- PDF

@dataclass
class _Line:
    text: str
    bbox: list[float]
    size: float
    bold: bool
    font: str
    horizontal: bool


def _pdf_lines(block: dict) -> list[_Line]:
    out = []
    for line in block.get("lines", []):
        spans = [s for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = "".join(s["text"] for s in line["spans"])
        chars = sum(len(s["text"].strip()) for s in spans)
        size = sum(s["size"] * len(s["text"].strip()) for s in spans) / max(chars, 1)
        bold_chars = sum(len(s["text"].strip()) for s in spans if (s["flags"] & 16) or re.search(r"bold|black|heavy|medi|semibold", s["font"], re.I))
        out.append(_Line(
            text=text, bbox=list(line["bbox"]), size=size, bold=bold_chars >= 0.8 * chars,
            font=max(spans, key=lambda s: len(s["text"]))["font"],
            horizontal=abs(line.get("dir", (1, 0))[1]) < 0.1,
        ))
    return out


def _furniture_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _detect_furniture(raw_pages: list[list[dict]], heights: list[float]) -> set[tuple[int, int]]:
    """(page_index, block_index) of running headers/footers and page numbers:
    blocks in the top/bottom 8% of the page that are bare numbers or whose
    digit-normalized text repeats on at least half of the pages."""
    n_pages = len(raw_pages)
    margin_blocks = []
    for pi, blocks in enumerate(raw_pages):
        h = heights[pi]
        for bi, b in enumerate(blocks):
            y0, y1 = b["bbox"][1], b["bbox"][3]
            if y1 <= h * 0.08 or y0 >= h * 0.92:
                margin_blocks.append((pi, bi, b["text"]))
    counts = Counter(_furniture_key(t) for _, _, t in margin_blocks)
    drop = set()
    for pi, bi, text in margin_blocks:
        if _ROMAN_OR_NUM.match(text.strip()) or (n_pages >= 3 and counts[_furniture_key(text)] >= max(2, n_pages // 2)):
            drop.add((pi, bi))
    return drop


def _row_cells(words: list[tuple], gap: float) -> list[tuple[float, float, str]]:
    words = sorted(words, key=lambda w: w[0])
    cells, cur = [], None
    for w in words:
        if cur and w[0] - cur[1] <= gap:
            cur = (cur[0], w[2], cur[2] + " " + w[4])
        else:
            if cur:
                cells.append(cur)
            cur = (w[0], w[2], w[4])
    if cur:
        cells.append(cur)
    return cells


def _words_to_rows(words: list[tuple]) -> list[list[str]]:
    """Builds a grid from positioned words: rows by baseline, columns anchored
    on the row(s) with the most cells, other rows' cells snapped to the
    nearest anchor column."""
    if not words:
        return []
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur, cur_y = [], [], None
    for w in words:
        yc = (w[1] + w[3]) / 2
        if cur and abs(yc - cur_y) > 3:
            rows.append(cur)
            cur = []
        if not cur:
            cur_y = yc
        cur.append(w)
    if cur:
        rows.append(cur)

    row_cells = [_row_cells(r, gap=5.5) for r in rows]
    max_cells = max(len(c) for c in row_cells)
    anchors_rows = [c for c in row_cells if len(c) == max_cells]
    ncols = max_cells
    centers = [sum((r[i][0] + r[i][1]) / 2 for r in anchors_rows) / len(anchors_rows) for i in range(ncols)]
    lefts = [min(r[i][0] for r in anchors_rows) for i in range(ncols)]

    grid = []
    for cells in row_cells:
        row = [""] * ncols
        for x0, x1, text in cells:
            if len(cells) == ncols:
                col = cells.index((x0, x1, text))
            else:
                # left-aligned label columns snap by left edge, numbers by center
                col = min(range(ncols), key=lambda i: min(abs((x0 + x1) / 2 - centers[i]), abs(x0 - lefts[i])))
            row[col] = (row[col] + " " + text).strip()
        grid.append(row)
    return grid


def _caption_tables(page, blocks: list[dict], body_right: float, body_left: float) -> list[dict]:
    """Tables without full ruling lines (booktabs style): anchor on a
    "Table N:" caption and collect the tabular lines directly below it
    (or above it, if nothing tabular follows)."""
    import fitz

    words = page.get_text("words")
    found = []
    text_width = max(body_right - body_left, 1)
    for bi, b in enumerate(blocks):
        if not _CAPTION.match(b["text"]):
            continue
        cap_rect = fitz.Rect(b["bbox"])
        for direction in (1, -1):
            order = [x for x in blocks if (x["bbox"][1] >= cap_rect.y1 - 1 if direction == 1 else x["bbox"][3] <= cap_rect.y0 + 1)]
            order.sort(key=lambda x: x["bbox"][1] * direction)
            region, last_edge = None, (cap_rect.y1 if direction == 1 else cap_rect.y0)
            for x in order:
                lines = x["lines"]
                if not lines:
                    continue
                # prose: any full-width line of running text ends the table
                is_prose = any((l.bbox[2] - l.bbox[0]) > 0.8 * text_width and len(l.text) > 60 for l in lines)
                gap = (x["bbox"][1] - last_edge) if direction == 1 else (last_edge - x["bbox"][3])
                # row groups inside a table (Table 3 style) are separated by
                # ~15pt; only a much larger gap means the table has ended
                if is_prose or x.get("heading") or _CAPTION.match(x["text"]) or (region is not None and gap > 36):
                    break
                r = fitz.Rect(x["bbox"])
                region = r if region is None else region | r
                last_edge = x["bbox"][3] if direction == 1 else x["bbox"][1]
            if region is None:
                continue
            tw = [w for w in words if fitz.Rect(w[:4]).intersects(region) and fitz.Rect(w[:4]).get_area() > 0
                  and (fitz.Rect(w[:4]) & region).get_area() / fitz.Rect(w[:4]).get_area() > 0.5]
            grid = _words_to_rows(tw)
            if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                found.append({"bbox": list(region | cap_rect), "rows": grid, "caption": normalize_ws(b["text"]).replace("\n", " "),
                              "caption_index": bi, "region": list(region)})
                break
    return found


def _ruled_tables(page) -> list[dict]:
    try:
        tables = page.find_tables()
    except Exception:
        return []
    out = []
    for t in tables.tables:
        rows = t.extract()
        cells = [c for r in rows for c in r]
        filled = sum(1 for c in cells if c and c.strip())
        # figures drawn with lines (attention maps, diagrams) also produce
        # "tables": reject sparse or implausibly wide grids
        if 2 <= t.col_count <= 15 and t.row_count >= 2 and filled >= 4 and filled / max(len(cells), 1) >= 0.5:
            out.append({"bbox": list(t.bbox), "rows": rows, "caption": None, "region": list(t.bbox)})
    return out


def _extract_pdf(file_path: str) -> ExtractedDocument:
    import fitz

    settings = get_settings()
    doc = fitz.open(file_path)
    notes, ocr_pages = [], []
    engine = None
    raw_pages: list[list[dict]] = []
    heights, page_sizes = [], {}

    for pi, page in enumerate(doc):
        page_sizes[pi + 1] = (page.rect.width, page.rect.height)
        heights.append(page.rect.height)
        blocks = []
        for b in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
            if b.get("type") != 0:
                continue
            lines = [l for l in _pdf_lines(b) if l.horizontal]
            if not lines:
                continue
            text = normalize_ws("\n".join(l.text for l in lines))
            if text:
                blocks.append({"bbox": list(b["bbox"]), "lines": lines, "text": text, "ocr": False})

        if sum(len(b["text"]) for b in blocks) < settings.ocr_min_chars:
            engine = engine or ocr_engine()
            if engine:
                try:
                    ocr_lines = _ocr_page(page, engine)
                except Exception as exc:
                    logger.exception("OCR failed on page %d", pi + 1)
                    notes.append(f"OCR failed on page {pi + 1}: {exc}")
                    ocr_lines = []
                if ocr_lines:
                    ocr_pages.append(pi + 1)
                    blocks = _group_ocr_lines(ocr_lines)
        raw_pages.append(blocks)

    if not any(raw_pages):
        if not ocr_engine():
            raise ValueError(
                "No readable text was found in this PDF, and no OCR engine is available "
                "(install tesseract-ocr, or `pip install easyocr`) — this looks like a "
                "scanned/image-only PDF."
            )
        raise ValueError(
            "No readable text was found in this PDF, even after OCR on every page — "
            "the document appears to be empty or too low quality to read."
        )

    furniture = _detect_furniture(raw_pages, heights)
    vocab = _vocab(b["text"] for blocks in raw_pages for b in blocks)

    # body font size = the size carrying the most characters
    size_chars = Counter()
    left_edges = Counter()
    for blocks in raw_pages:
        for b in blocks:
            for l in b["lines"]:
                size_chars[round(l.size * 2) / 2] += len(l.text)
                left_edges[round(l.bbox[0])] += 1
    body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
    margins = [x for x, c in left_edges.items() if c >= 5] or [min(left_edges) if left_edges else 0]
    body_left = min(margins)
    body_right = max((l.bbox[2] for blocks in raw_pages for b in blocks for l in b["lines"]), default=body_left + 400)

    def heading_info(b) -> tuple[bool, float]:
        lines = b["lines"]
        text = b["text"].replace("\n", " ").strip()
        if b.get("ocr") or len(lines) > 3 or len(text) > 120 or len(text.split()) > 14:
            return False, 0
        if not re.search(r"[A-Za-z]{2}", text) or text.endswith((".", ",", ";", ":")) or _CAPTION.match(text):
            return False, 0
        if re.match(r"^(figure|fig\.)\s*\d", text, re.I):
            return False, 0
        size = max(l.size for l in lines)
        bigger = size >= body_size * 1.15
        bold = all(l.bold for l in lines)
        if not (bigger or bold):
            return False, 0
        at_margin = any(abs(lines[0].bbox[0] - m) <= 4 for m in margins)
        centered = abs(((lines[0].bbox[0] + lines[-1].bbox[2]) / 2) - (body_left + body_right) / 2) < 30
        if not (at_margin or (centered and bigger)):
            return False, 0  # figure labels, sidebars
        return True, size

    for blocks in raw_pages:
        for b in blocks:
            b["heading"], b["hsize"] = heading_info(b)

    # heading levels: numbering depth when present ("3.2.1" -> 3), otherwise the
    # rank of the heading's font size among numbered headings of that size
    heading_sizes = sorted({round(b["hsize"] * 2) / 2 for bl in raw_pages for b in bl if b["heading"]}, reverse=True)
    title_size = heading_sizes[0] if heading_sizes and heading_sizes[0] >= body_size * 1.3 else None
    size_level: dict[float, int] = {}
    for bl in raw_pages:
        for b in bl:
            if b["heading"]:
                m = _NUMBERED.match(b["text"].replace("\n", " "))
                if m:
                    size_level.setdefault(round(b["hsize"] * 2) / 2, m.group(1).count(".") + 1)
    title_taken = False

    blocks_out: list[Block] = []
    for pi, page in enumerate(doc):
        blocks = raw_pages[pi]
        # caption-anchored grids first: on booktabs-style tables PyMuPDF's
        # line-based detector merges columns, the word grid doesn't
        tables = _caption_tables(page, blocks, body_right, body_left) if blocks and not blocks[0].get("ocr") else []
        table_rects = [fitz.Rect(t["region"]) for t in tables]
        for t in (_ruled_tables(page) if blocks and not blocks[0].get("ocr") else []):
            if not any(fitz.Rect(t["region"]).intersects(r) for r in table_rects):
                tables.append(t)
                table_rects.append(fitz.Rect(t["region"]))

        items: list[tuple[float, float, Block]] = []
        used_captions = {t.get("caption_index") for t in tables}
        for bi, b in enumerate(blocks):
            if (pi, bi) in furniture or bi in used_captions:
                continue
            r = fitz.Rect(b["bbox"])
            if any(r.intersects(tr) and (r & tr).get_area() > 0.5 * max(r.get_area(), 1e-6) for tr in table_rects):
                continue
            text_lines = [l.text for l in b["lines"]] if not b.get("ocr") else b["text"].split("\n")
            if b.get("ocr_heading"):
                # scanned page: no font data, fall back to the plain-text heading rule
                level = 0 if not title_taken and not blocks_out and not items else 1
                title_taken = title_taken or level == 0
                items.append((b["bbox"][1], b["bbox"][0], Block(page=pi + 1, kind="heading", text=b["text"], bbox=b["bbox"], level=level)))
                continue
            if b["heading"]:
                text = normalize_ws(" ".join(text_lines))
                hs = round(b["hsize"] * 2) / 2
                m = _NUMBERED.match(text)
                if m:
                    level = m.group(1).count(".") + 1
                elif title_size and hs == title_size and pi == 0 and not title_taken:
                    level, title_taken = 0, True
                else:
                    level = size_level.get(hs, 1)
                block = Block(page=pi + 1, kind="heading", text=text, bbox=b["bbox"], font_size=round(b["hsize"], 1), level=level)
            else:
                text = normalize_ws(_dehyphenate_lines(text_lines, vocab))
                sizes = [l.size for l in b["lines"]]
                block = Block(page=pi + 1, kind="paragraph", text=text, bbox=b["bbox"],
                              font_size=round(sum(sizes) / len(sizes), 1) if sizes else None)
            items.append((b["bbox"][1], b["bbox"][0], block))

        for t in tables:
            md = rows_to_markdown(t["rows"])
            if not md:
                continue
            caption = t.get("caption")
            text = f"{caption}\n\n{md}" if caption else md
            items.append((t["bbox"][1], t["bbox"][0], Block(
                page=pi + 1, kind="table", text=text, bbox=t["bbox"],
                table={"rows": t["rows"], "caption": caption},
            )))

        items = _reading_order(items, page.rect.width)
        blocks_out.extend(it[2] for it in items)

    toc, title = _assign_sections(blocks_out)
    pages = _build_pages(blocks_out, list(range(1, len(doc) + 1)))
    if ocr_pages:
        notes.append(f"OCR ({engine}) used on pages {ocr_pages}")
    return ExtractedDocument(pages=pages, blocks=blocks_out, toc=toc, title=title or (doc.metadata or {}).get("title") or None,
                             page_sizes=page_sizes, ocr_pages=ocr_pages, notes=notes)


def _reading_order(items, page_width):
    """Top-to-bottom, left-to-right. On two-column pages, runs of column
    blocks between full-width blocks are read left column first."""
    items = sorted(items, key=lambda it: (round(it[0]), it[1]))
    mid = page_width / 2

    def side(block):
        if not block.bbox:
            return None
        if block.bbox[2] <= mid + 8:
            return 0
        if block.bbox[0] >= mid - 8:
            return 1
        return None  # spans the gutter

    columnar = [it for it in items if side(it[2]) is not None and len(it[2].text) > 200]
    if len({side(it[2]) for it in columnar}) < 2 or len(columnar) < 3:
        return items
    out, band = [], []
    for it in items:
        if side(it[2]) is None:
            out.extend(sorted(band, key=lambda b: (side(b[2]), b[0])))
            band = []
            out.append(it)
        else:
            band.append(it)
    out.extend(sorted(band, key=lambda b: (side(b[2]), b[0])))
    return out


def _group_ocr_lines(lines: list[tuple[str, list[float]]]) -> list[dict]:
    """Groups OCR lines into paragraph blocks by vertical gaps."""
    blocks, cur = [], []
    for text, bbox in lines:
        if cur and bbox[1] - cur[-1][1][3] > 0.9 * (cur[-1][1][3] - cur[-1][1][1]):
            blocks.append(cur)
            cur = []
        cur.append((text, bbox))
    if cur:
        blocks.append(cur)
    out = []
    for group in blocks:
        xs0 = [b[0] for _, b in group]; ys0 = [b[1] for _, b in group]
        xs1 = [b[2] for _, b in group]; ys1 = [b[3] for _, b in group]
        text = normalize_ws("\n".join(t for t, _ in group))
        out.append({"bbox": [min(xs0), min(ys0), max(xs1), max(ys1)], "lines": [], "text": text, "ocr": True,
                    "heading": False, "hsize": 0, "ocr_heading": len(group) == 1 and _looks_like_heading(text, 18)})
    return out


# ----------------------------------------------------------------- DOCX

_WORDS_PER_DOCX_PAGE = 500


def _extract_docx(file_path: str) -> ExtractedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(file_path)
    has_rendered_breaks = "lastRenderedPageBreak" in d.element.xml
    page, words_on_page = 1, 0
    blocks: list[Block] = []
    notes = []
    if not has_rendered_breaks:
        notes.append(f"DOCX has no rendered page breaks; pages are approximated at {_WORDS_PER_DOCX_PAGE} words.")

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            xml = child.xml
            if has_rendered_breaks and ("lastRenderedPageBreak" in xml or 'w:type="page"' in xml) and blocks:
                page += 1
            text = normalize_ws(p.text)
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            level = None
            if style == "Title":
                level = 0
            else:
                m = re.match(r"Heading\s+(\d)", style)
                if m:
                    level = int(m.group(1))
            if level is not None:
                blocks.append(Block(page=page, kind="heading", text=text, level=level))
            else:
                blocks.append(Block(page=page, kind="paragraph", text=text))
            words_on_page += len(text.split())
        elif tag == "tbl":
            t = Table(child, d)
            rows = [[normalize_ws(c.text) for c in r.cells] for r in t.rows]
            md = rows_to_markdown(rows)
            if md:
                blocks.append(Block(page=page, kind="table", text=md, table={"rows": rows, "caption": None}))
                words_on_page += len(md.split())
        if not has_rendered_breaks and words_on_page >= _WORDS_PER_DOCX_PAGE:
            page += 1
            words_on_page = 0

    if not blocks:
        raise ValueError("No readable text was found in this document.")
    toc, title = _assign_sections(blocks)
    pages = _build_pages(blocks, sorted({b.page for b in blocks}))
    return ExtractedDocument(pages=pages, blocks=blocks, toc=toc, title=title or d.core_properties.title or None,
                             page_sizes={}, ocr_pages=[], notes=notes)


# ----------------------------------------------------------------- TXT

_TXT_CHARS_PER_PAGE = 3000


def _looks_like_heading(line: str, max_words: int = 10) -> bool:
    line = line.strip()
    if not (2 <= len(line) <= 8 * max_words) or line.endswith((".", ",", ";", ":", "?")):
        return False
    words = line.split()
    if len(words) > max_words or not line[0].isupper() and not line[0].isdigit():
        return False
    if _NUMBERED.match(line) or re.match(r"^(chapter|section|part|appendix)\b", line, re.I):
        return True
    small = {"a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "with", "by", "at", "vs", "from"}
    return all(w[0].isupper() or w.lower() in small or not w[0].isalpha() for w in words)


def _extract_txt(file_path: str) -> ExtractedDocument:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace("\r\n", "\n")
    if not content.strip():
        raise ValueError("No readable text was found in this file.")

    paras = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
    vocab = _vocab(paras)
    blocks: list[Block] = []
    page, chars = 1, 0
    for i, para in enumerate(paras):
        lines = [l for l in para.split("\n") if l.strip()]
        # the first line may be a long title; later headings are short
        if len(lines) == 1 and _looks_like_heading(lines[0], max_words=18 if i == 0 else 10):
            text = normalize_ws(lines[0])
            m = _NUMBERED.match(text)
            level = 0 if i == 0 else (m.group(1).count(".") + 1 if m else 1)
            block = Block(page=page, kind="heading", text=text, level=level)
        else:
            block = Block(page=page, kind="paragraph", text=normalize_ws(_dehyphenate_lines(lines, vocab)))
        # break pages only between paragraphs, never mid-paragraph, and never
        # directly after a heading (keep the heading with its first paragraph)
        if chars and chars + len(block.text) > _TXT_CHARS_PER_PAGE and not (blocks and blocks[-1].kind == "heading"):
            page += 1
            chars = 0
            block.page = page
        blocks.append(block)
        chars += len(block.text)
    toc, title = _assign_sections(blocks)
    pages = _build_pages(blocks, sorted({b.page for b in blocks}))
    return ExtractedDocument(pages=pages, blocks=blocks, toc=toc, title=title, page_sizes={}, ocr_pages=[], notes=[])
