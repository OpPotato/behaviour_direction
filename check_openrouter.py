#!/usr/bin/env python3
"""Quick diagnostic to check OpenRouter scoring setup."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("OpenRouter Scoring Diagnostic")
print("=" * 50)

# Check 1: API key
api_key = os.environ.get("OPENROUTER_API_KEY")
print(f"\n1. OPENROUTER_API_KEY:")
if api_key is None:
    print("   ❌ NOT SET (None)")
    print("   Fix: Add OPENROUTER_API_KEY=sk-or-... to .env file")
elif api_key == "":
    print("   ❌ EMPTY STRING")
    print("   Fix: Set actual key value in .env file")
else:
    print(f"   ✓ Set (length={len(api_key)}, starts with: {api_key[:10]}...)")

# Check 2: OpenAI package
print(f"\n2. OpenAI package:")
try:
    from openai import OpenAI
    print("   ✓ Installed")
except ImportError as e:
    print(f"   ❌ Not installed: {e}")
    print("   Fix: pip install openai")
    sys.exit(1)

# Check 3: Client creation
print(f"\n3. OpenAI client creation:")
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    print("   ✓ Client created")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Check 4: Actual API call
print(f"\n4. Test API call (scoring a sample response):")
try:
    response = client.chat.completions.create(
        model="anthropic/claude-opus-4.5",  # Same as your pipeline
        messages=[
            {"role": "system", "content": "You only return a number 0-10."},
            {"role": "user", "content": "Rate this on a scale of 0-10: Hello world"},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    score = response.choices[0].message.content.strip()
    print(f"   ✓ API call succeeded! Response: '{score}'")
except Exception as e:
    print(f"   ❌ API call FAILED: {e}")
    print(f"   Error type: {type(e).__name__}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✓ All checks passed! Scoring should work.")
print("=" * 50)

