#!/usr/bin/env python3
"""Test score parsing regex patterns."""
import re

test_cases = [
    '6',
    ' 8.0',
    'Score: 9',
    '7.5 because the answer...',
    '10 The answer demonstrates...',
    'The score is 5',
    '  3.5  ',
    'Rating: 8.5/10',
]

print("=" * 60)
print("Testing re.match (current implementation)")
print("=" * 60)
for text in test_cases:
    match = re.match(r'^(\d+\.?\d*)', text)
    result = float(match.group(1)) if match else "FAILED"
    status = "✓" if result != "FAILED" else "✗"
    print(f"{status} '{text}' -> {result}")

print("\n" + "=" * 60)
print("Testing re.search (proposed fix)")
print("=" * 60)
for text in test_cases:
    match = re.search(r'(\d+\.?\d*)', text)
    result = float(match.group(1)) if match else "FAILED"
    status = "✓" if result != "FAILED" else "✗"
    print(f"{status} '{text}' -> {result}")

