#!/usr/bin/env python3
"""Check the built manuscript with standard Poppler tools."""
from pathlib import Path
import re, subprocess
root=Path(__file__).resolve().parents[1]
log=(root/'main.log').read_text(errors='replace')
for pattern in (r'^!', r'undefined references', r'Citation .+ undefined', r'Reference .+ undefined', r'Overfull \\hbox', r'Font shape .* undefined', r'defaults substituted'):
    assert not re.search(pattern,log,re.M), pattern
text=subprocess.check_output(['pdftotext','-layout',str(root/'main.pdf'),'-'],text=True)
pages=text.split('\f')
if not pages[-1].strip():pages.pop()
assert len(pages)<=8, f'{len(pages)} total pages; inspect excluded-content boundary'
assert 'LLM Usage Statement' in text
assert 'Anonymous' in text
assert not re.search(r'/home/|/data/|@|Institution1|First Author',text)
assert '?'*2 not in text
assert '2027' in text
print(f'PASS: {len(pages)} total PDF pages (including disclosure and references); no unresolved references, overfull boxes, or identifying paths/emails.')
