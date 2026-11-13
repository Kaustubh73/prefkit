import zipfile, re, os
from xml.etree import ElementTree as ET

SRC='docs/Current Report Draft .docx'
OUT='docs/Current Report Draft - Reorganized.docx'

W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

# Register all namespaces so ET round-trips attribute prefixes
NSMAP={
 'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
 'c':'http://schemas.openxmlformats.org/drawingml/2006/chart',
 'cr':'http://schemas.openxmlformats.org/officeDocument/2006/custom-properties',
 'dgm':'http://schemas.openxmlformats.org/drawingml/2006/diagram',
 'lc':'http://schemas.openxmlformats.org/drawingml/2006/lock',
 'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
 'mc':'http://schemas.openxmlformats.org/markup-compatibility/2006',
 'o':'urn:schemas-microsoft-com:office:office',
 'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture',
 'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
 'sl':'http://schemas.openxmlformats.org/schemaLibrary/2006/main',
 'v':'urn:schemas-microsoft-com:vml',
 'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
 'w10':'urn:schemas-microsoft-com:office:word',
 'w14':'http://schemas.microsoft.com/office/word/2010/wordml',
 'w15':'http://schemas.microsoft.com/office/word/2012/wordml',
 'w16':'http://schemas.microsoft.com/office/word/2013/wordml',
 'w16cex':'http://schemas.microsoft.com/office/word/2014/wordml',
 'w16cid':'http://schemas.microsoft.com/office/word/2016/wordml',
 'wne':'http://schemas.microsoft.com/office/word/2006/wordml',
 'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
 'wpg':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing/2010',
 'wps':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingShape',
}
for p,u in NSMAP.items(): ET.register_namespace(p,u)

z=zipfile.ZipFile(SRC)
xml=z.read('word/document.xml')
root=ET.fromstring(xml)
body=root.find(W+'body')

def text_of(el):
    return ''.join(t.text or '' for t in el.iter(W+'t'))

def pstyle(el):
    pPr=el.find(W+'pPr')
    if pPr is None: return None
    ps=pPr.find(W+'pStyle')
    return ps.get(W+'val') if ps is not None else None

def find_all(pred):
    return [el for el in body if pred(el)]

# Identify heading paragraphs by exact text start (ignore style)
def hidx(start_text):
    for i,el in enumerate(body):
        if text_of(el).strip().startswith(start_text):
            return i
    return None

# Map sections by body index
idx_methods=hidx('3. Methods')
idx_341=hidx('3.4.1 Frozen Menus for M3')
idx_342=hidx('3.4.2 Frozen Menus for M4')
idx_35=hidx('3.5 Models')
idx_36=hidx('3.6 Inference Procedure')
idx_37=hidx('3.7 Sampling Configuration')
idx_311=hidx('3.11 Empirical Verification of Axioms')
idx_312=hidx('3.12 Experimental Procedure')
idx_results=hidx('4. Results')
idx_appendix=hidx('Appendix')
idx_llm=hidx('LLM Usage Statement')
print('anchors:', dict(m=idx_methods,_341=idx_341,_342=idx_342,_35=idx_35,_36=idx_36,_37=idx_37,_311=idx_311,_312=idx_312,r=idx_results,app=idx_appendix,llm=idx_llm))

# Helper: make a pointer paragraph copying style from an existing body paragraph
def make_pointer(ref_el, txt):
    p=ET.SubElement(body, W+'p')  # placeholder, will be moved
    body.remove(p)
    # clone pPr
    pPr=ref_el.find(W+'pPr')
    if pPr is not None:
        p.append(pPr)
    r=ET.SubElement(p, W+'r')
    # copy rPr if present in ref run
    ref_r=ref_el.find(W+'r')
    if ref_r is not None:
        ref_rpr=ref_r.find(W+'rPr')
        if ref_rpr is not None:
            r.append(ref_rpr)
    t=ET.SubElement(r, W+'t')
    t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    t.text=txt
    return p

def set_heading_text(el, newtext):
    # replace text of first run (keep formatting)
    r=el.find(W+'r')
    t=r.find(W+'t')
    t.text=newtext

# Build the new body order as a list of elements
new_order=[]

# 1) front matter: everything before 3. Methods
new_order += body[:idx_methods]

# 2) Methods 3.1..3.4 (+ Fig1) up to (not incl) 3.4.1
methods_top=body[idx_methods:idx_341]
# insert pointer after 3.4 block (right before idx_341) -> pointer as separate element
ptrA=make_pointer(body[idx_methods+1], 'The frozen-menu construction for M3 and M4 is detailed in Appendix A\u2013B.')
methods_block=list(methods_top)+[ptrA]
new_order += methods_block

# 3) 3.5 Models (+ table) up to 3.6
new_order += body[idx_35:idx_36]
ptrC=make_pointer(body[idx_methods+1], 'Inference and decode details are in Appendix C.')
new_order += [ptrC]

# 4) 3.7..3.10 (up to 3.11)
new_order += body[idx_37:idx_311]
ptrDE=make_pointer(body[idx_methods+1], 'Empirical A1\u2013A5 verification and the full experimental run procedure are in Appendix D\u2013E.')
new_order += [ptrDE]

# 5) Results .. Appendix heading (exclusive of appendix content placeholder)
new_order += body[idx_results:idx_appendix]
new_order += [body[idx_appendix]]

# 6) Relocated + renumbered methods (rename headings)
def clone_range(a,b):
    return list(body[a:b])

reloc_A=clone_range(idx_341, idx_342)
reloc_B=clone_range(idx_342, idx_35)
reloc_C=clone_range(idx_36, idx_37)
reloc_D=clone_range(idx_311, idx_312)
reloc_E=clone_range(idx_312, idx_results)

set_heading_text(reloc_A[0], 'A. Frozen Menus for M3')
set_heading_text(reloc_B[0], 'B. Frozen Menus for M4')
set_heading_text(reloc_C[0], 'C. Inference Procedure')
set_heading_text(reloc_D[0], 'D. Empirical Verification of Axioms')
set_heading_text(reloc_E[0], 'E. Experimental Procedure')

new_order += reloc_A + reloc_B + reloc_C + reloc_D + reloc_E

# 7) LLM usage + trailing sectPr
new_order += body[idx_llm:]

# Replace body children
for c in list(body):
    body.remove(c)
for c in new_order:
    body.append(c)

# Fix internal cross-refs everywhere in the retained body + moved sections
for el in list(body):
    for t in el.iter(W+'t'):
        if t.text:
            t.text=t.text.replace('\u00a73.4.1','Appendix A').replace('\u00a73.4.2','Appendix B')
            t.text=t.text.replace('\u00a73.6','Appendix C').replace('\u00a73.11','Appendix D').replace('\u00a73.12','Appendix E')

# Serialize
out_xml=ET.tostring(root, encoding='unicode')
# wrap with xml declaration + standalone like original
header='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
new_doc=header+out_xml

# Validate by re-parsing
ET.fromstring(new_doc)
print('NEW DOC PARSES OK')

if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as zout:
    for item in zin.namelist():
        data=zin.read(item)
        if item=='word/document.xml':
            data=new_doc.encode('utf-8')
        zout.writestr(item,data)

full=''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', new_doc, re.DOTALL))
print('bytes', os.path.getsize(OUT))
for kw in ['Prefkit','Kaustubh Gupta','1. Introduction','2. Related Work','3. Methods','4. Results','5. Discussion','6. Conclusion','References','Appendix','LLM Usage Statement','A. Frozen Menus for M3','B. Frozen Menus for M4','C. Inference Procedure','D. Empirical Verification of Axioms','E. Experimental Procedure']:
    print(('OK ' if kw in full else 'MISSING '), kw)
print('Heading2 count:', len(re.findall(r'w:val="Heading2"', new_doc)))
print('pointers present:', full.count('is detailed in Appendix'), full.count('are in Appendix'))
