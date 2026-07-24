"""Investigate coding validation→test collapse."""
import json
import re

# Load coding data
val = [json.loads(l) for l in open('data/coding/validation.jsonl')]
test = [json.loads(l) for l in open('data/coding/test.jsonl')]
train = [json.loads(l) for l in open('data/coding/train.jsonl')]

print(f"Validation: {len(val)} samples")
print(f"Test: {len(test)} samples")
print(f"Train: {len(train)} samples")

# Check for hash overlap
val_hashes = set(v['hash'] for v in val)
test_hashes = set(t['hash'] for t in test)
train_hashes = set(t['hash'] for t in train)
overlap_vt = val_hashes & test_hashes
overlap_vtr = val_hashes & train_hashes
overlap_ttr = test_hashes & train_hashes
print(f"\nHash overlap val-test: {len(overlap_vt)}")
print(f"Hash overlap val-train: {len(overlap_vtr)}")
print(f"Hash overlap test-train: {len(overlap_ttr)}")

# Check for prefix overlap (50 chars)
val_prefixes = set(v['text'][:50] for v in val)
test_prefixes = set(t['text'][:50] for t in test)
prefix_overlap = val_prefixes & test_prefixes
print(f"Prefix overlap (50 chars): {len(prefix_overlap)}")

# Analyze prompt-code structure
def analyze_coding_sample(text):
    """Extract prompt and code from a coding sample."""
    if '```' in text:
        parts = text.split('```')
        prompt = parts[0].strip()
        # Code is between first ``` and next ```
        if len(parts) >= 3:
            code = parts[1].strip()
            # Remove language identifier
            if code.startswith(('python', 'bash', 'typescript', 'javascript', 'java', 'sql')):
                code = code.split('\n', 1)[1] if '\n' in code else ''
            return prompt, code
    return text, ''

# Check for prompt-code mismatches
def check_mismatch(prompt, code):
    """Check for obvious prompt-code mismatches."""
    p = prompt.lower()
    c = code.lower()
    mismatches = []

    # Function name mismatch
    if 'sum' in p and 'sum' not in c and ('max' in c or 'min' in c):
        mismatches.append('sum_vs_max_min')
    if 'max' in p and 'max' not in c and 'sum' in c:
        mismatches.append('max_vs_sum')
    if 'flatten' in p and 'flatten' not in c:
        mismatches.append('flatten_missing')

    # SSH vs pip
    if 'ssh' in p and 'pip install' in c and 'ssh' not in c:
        mismatches.append('ssh_vs_pip')

    # JSON parse mismatch
    if 'parse json' in p and 'json.loads' in c:
        # Check if field extraction matches
        pass

    return mismatches

print("\n=== VALIDATION: Prompt-Code Analysis ===")
val_mismatches = 0
for i, v in enumerate(val):
    prompt, code = analyze_coding_sample(v['text'])
    mm = check_mismatch(prompt, code)
    if mm:
        val_mismatches += 1
        if val_mismatches <= 5:
            print(f"  [{i}] MISMATCH: {mm}")
            print(f"      Prompt: {prompt[:80]}")
            print(f"      Code: {code[:80]}")

print(f"Validation mismatches: {val_mismatches}/{len(val)}")

print("\n=== TEST: Prompt-Code Analysis ===")
test_mismatches = 0
for i, t in enumerate(test):
    prompt, code = analyze_coding_sample(t['text'])
    mm = check_mismatch(prompt, code)
    if mm:
        test_mismatches += 1
        if test_mismatches <= 10:
            print(f"  [{i}] MISMATCH: {mm}")
            print(f"      Prompt: {prompt[:80]}")
            print(f"      Code: {code[:80]}")

print(f"Test mismatches: {test_mismatches}/{len(test)}")

# Compare text length distributions
val_lens = [len(v['text']) for v in val]
test_lens = [len(t['text']) for t in test]
print(f"\nValidation text length: mean={sum(val_lens)/len(val_lens):.0f}, min={min(val_lens)}, max={max(val_lens)}")
print(f"Test text length: mean={sum(test_lens)/len(test_lens):.0f}, min={min(test_lens)}, max={max(test_lens)}")

# Check for templated patterns
val_templates = {}
for v in val:
    # Extract template pattern (replace numbers with X)
    template = re.sub(r'\d+', 'X', v['text'][:100])
    val_templates[template] = val_templates.get(template, 0) + 1

test_templates = {}
for t in test:
    template = re.sub(r'\d+', 'X', t['text'][:100])
    test_templates[template] = test_templates.get(template, 0) + 1

# Find shared templates
shared = set(val_templates.keys()) & set(test_templates.keys())
print(f"\nShared templates (first 100 chars, numbers replaced): {len(shared)}")
if shared:
    for t in list(shared)[:5]:
        print(f"  Val count: {val_templates[t]}, Test count: {test_templates[t]}")
        print(f"  Template: {t[:80]}")

# Check language distribution
def get_language(text):
    if '```python' in text:
        return 'python'
    elif '```bash' in text:
        return 'bash'
    elif '```typescript' in text:
        return 'typescript'
    elif '```javascript' in text:
        return 'javascript'
    elif '```java' in text:
        return 'java'
    elif '```sql' in text:
        return 'sql'
    elif '```' in text:
        return 'other'
    return 'none'

val_langs = {}
for v in val:
    lang = get_language(v['text'])
    val_langs[lang] = val_langs.get(lang, 0) + 1

test_langs = {}
for t in test:
    lang = get_language(t['text'])
    test_langs[lang] = test_langs.get(lang, 0) + 1

print(f"\nValidation languages: {val_langs}")
print(f"Test languages: {test_langs}")
