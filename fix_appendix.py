#!/usr/bin/env python3
"""Appendix-only follow-up fix (no other section touched):
 - Set 'Appendix D' heading to Heading 3.
 - Rename 'Inference Procedure' -> 'F.1 Inference Procedure' (Heading 4).
 - Rename 'F.1 Experimental Procedure' -> 'F.2 Experimental Procedure' (Heading 4).
"""
from docx import Document
from docx.oxml.ns import qn

SRC = "docs/Current Report Draft .docx"
d = Document(SRC)

def set_style(p, style_name):
    p.style = d.styles[style_name]

for p in d.paragraphs:
    t = p.text.strip()
    if t == "Appendix D: Excluded / Debug Runs":
        set_style(p, "Heading 3")
    elif t == "Inference Procedure":
        set_style(p, "Heading 4")
        # rename first run text
        p.runs[0].text = "F.1 Inference Procedure"
    elif t.startswith("F.1 Experimental Procedure"):
        set_style(p, "Heading 4")
        p.runs[0].text = "F.2 Experimental Procedure"

d.save(SRC)
print("Fix applied.")
