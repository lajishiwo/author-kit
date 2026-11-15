#!/usr/bin/env python3
"""Check the language-only pass against its evidence-updated baseline."""
from collections import Counter
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
baseline=json.loads((root/'docs/editorial/CVPR2027_EVIDENCE_UPDATED_BASELINE.json').read_text())
patterns={
    'numbers':r'(?<![A-Za-z])\d[\d,]*(?:\.\d+)?',
    'math':r'\\begin\{(equation|align)\}.*?\\end\{\1\}|\$[^$]*\$',
    'citations':r'\\cite\{[^}]+\}',
    'labels':r'\\label\{[^}]+\}',
    'references':r'\\ref\{[^}]+\}',
}
for name,before in baseline.items():
    after=(root/name).read_text()
    for kind,pattern in patterns.items():
        count=lambda text:Counter(m.group() for m in re.finditer(pattern,text,re.S))
        assert count(before)==count(after),(name,kind)
    assert len(re.split(r'\n\s*\n',before))==len(re.split(r'\n\s*\n',after)),(name,'paragraph count')
    assert after.count('\\textbf{')==before.count('\\textbf{'),(name,'bold formatting')
    assert after.count('\\emph{')==before.count('\\emph{'),(name,'emphasis formatting')
    assert '\u2014' not in after and '---' not in after,(name,'em dash')
    assert not re.search(r"\b\w+'s\b",after),(name,'possessive')
print('PASS: language-only edits preserve numeric tokens, formulas, citations, references, paragraph counts and emphasis.')
