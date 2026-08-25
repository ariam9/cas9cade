#!/usr/bin/env python
"""Decode a `kaggle kernels logs` JSON stream into readable text.

Kaggle returns logs as a JSON array of {stream_name, time, data} records, so
piping them straight to `tail` shows escaped JSON rather than the traceback.
"""
import json, re, sys
raw = sys.stdin.read()
for m in re.finditer(r'"data":"((?:[^"\\]|\\.)*)"', raw):
    t = json.loads('"' + m.group(1) + '"')
    if t.strip():
        sys.stdout.write(t if t.endswith("\n") else t + "\n")
