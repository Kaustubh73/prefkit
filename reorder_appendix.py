#!/usr/bin/env python3
"""Reorder ONLY the appendix (paragraphs 141..207, before Limitations) of
docs/Current Report Draft .docx. References, body, tables untouched.

Gathers EVERY meaningful region element by stable index, reorders completely,
renames stale body-subsection numbers to appendix labels, drops 2 junk empty
headings, adds a placeholder D. Reinserts the whole block right before the
Limitations heading. Nothing else is moved.
"""
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = "docs/Current Report Draft .docx"
d = Document(SRC)
ps = d.paragraphs
body = d.element.body

def el(i):
    return ps[i]._p

def rename(idx, new_text):
    e = el(idx)
    ts = e.findall('.//' + qn('w:t'))
    if ts:
        ts[0].text = new_text

def set_style(idx, style_name):
    ps[idx].style = d.styles[style_name]

def blank():
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr'); p.append(pPr)
    return p

# --- rename stale subsection numbers -> appendix sub-labels ---
rename(170, "C.1 Frozen Menus for M3");        set_style(170, "Heading 4")
rename(173, "C.2 Frozen Menus for M4");        set_style(173, "Heading 4")
rename(186, "E.1 check-axioms");               set_style(186, "Heading 4")
rename(178, "F.1 Inference Procedure");        set_style(178, "Heading 4")
rename(198, "F.2 Experimental Procedure");     set_style(198, "Heading 4")

# --- new placeholder D ---
hD = OxmlElement('w:p'); pPr = OxmlElement('w:pPr')
pStyle = OxmlElement('w:pStyle'); pStyle.set(qn('w:val'), "Heading3"); pPr.append(pStyle)
hD.append(pPr)
r = OxmlElement('w:r'); t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve')
t.text = "Appendix D: Excluded / Debug Runs"; r.append(t); hD.append(r)
pDnote = OxmlElement('w:p'); r2 = OxmlElement('w:r'); t2 = OxmlElement('w:t'); t2.set(qn('xml:space'), 'preserve')
t2.text = "To be added: Run 1 (protocol debug), Ollama-local backend, discarded_aborted runs."
r2.append(t2); pDnote.append(r2)

# --- full ordered region (every meaningful element, new order) ---
B = blank
ordered = [
    el(141),                      # Appendix (H2) title
    B(), el(143),                 # A
    B(), el(144),                 # B
    B(), el(145),                 # C
    el(149), el(150), el(151), el(152), el(153),   # M1 + prompts
    el(154), el(155), el(156),                     # M2 + prompts
    el(157), el(158), el(159), el(160), el(161),   # M3 + prompts
    el(162), el(163), el(164), el(165), el(166), el(167), el(168),  # M4 + prompts
    el(170), el(171),             # C.1
    el(173), el(174), el(176),     # C.2
    hD, pDnote,                   # D placeholder
    B(), el(146),                 # E heading
    el(186), el(187), el(188), el(189), el(190), el(191), el(192), el(193),
    el(194), el(195), el(196), el(197),           # E.1 walkthrough
    B(), el(147),                 # F heading
    el(178), el(180), el(182), el(184),           # F.1 Inference Procedure
    el(198), el(199), el(200), el(201), el(202), el(203), el(204), el(205),  # F.2
]

# Junk to drop (not reinserted): 142 (empty H3), 207 (empty H2), and the many
# empty Normal spacers. Drop = every region element NOT in ordered.
region_idx = set(range(141, 208))
ordered_idx = {141,143,144,145,149,150,151,152,153,154,155,156,157,158,159,160,161,
               162,163,164,165,166,167,168,170,171,173,174,176,146,186,187,188,189,
               190,191,192,193,194,195,196,197,147,178,180,182,184,198,199,200,201,
               202,203,204,205}
drop_idx = region_idx - ordered_idx   # includes 142,148,169,172,175,177,179,181,183,185,206,207

lim = el(next(i for i,p in enumerate(ps) if p.text.strip().startswith("Limitations and Dual-Use")))
# remove all region elements first (drop + ordered)
for i in region_idx:
    body.remove(el(i))
# reinsert ordered (incl. new D) right before Limitations
for e in ordered:
    body.insert(body.index(lim), e)

d.save(SRC)
print("Saved. Full appendix reorder complete; region fully relocated, junk dropped.")
