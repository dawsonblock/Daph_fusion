"""Fix ALL prompt-code mismatches in coding data by generating correct code."""
import json
import re
import hashlib
from pathlib import Path


def extract_parts(text):
    """Extract (prompt, lang, code, prefix, suffix) from a coding sample."""
    if '```' not in text:
        return None
    parts = text.split('```')
    prompt = parts[0].strip()
    code_block = parts[1] if len(parts) > 1 else ''
    suffix = '```'.join(parts[2:]) if len(parts) > 2 else ''

    # Detect language
    lang = None
    for l in ['python', 'bash', 'typescript', 'javascript', 'java', 'sql', 'sh']:
        if code_block.lower().startswith(l):
            lang = l
            code_block = code_block[len(l):].lstrip('\n')
            break
    if lang is None:
        if 'def ' in code_block:
            lang = 'python'
        elif 'function ' in code_block:
            lang = 'javascript'
        elif 'SELECT ' in code_block.upper():
            lang = 'sql'
        elif 'interface ' in code_block:
            lang = 'typescript'
        else:
            lang = 'python'

    return prompt, lang, code_block.strip(), suffix


# Correct implementations indexed by (language, prompt_pattern)
IMPLEMENTATIONS = {
    # Python implementations
    ('python', 'sum'): 'def sum_list(values):\n    return sum(values)',
    ('python', 'reverse a string'): 'def reverse_string(s):\n    return s[::-1]',
    ('python', 'flatten nested list'): 'def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result',
    ('python', 'flatten'): 'def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result',
    ('python', 'count vowels'): 'def count_vowels(s):\n    return sum(1 for c in s if c in \'aeiouAEIOU\')',
    ('python', 'check palindrome'): 'def is_palindrome(s):\n    return s == s[::-1]',
    ('python', 'palindrome'): 'def is_palindrome(s):\n    return s == s[::-1]',
    ('python', 'compute factorial'): 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
    ('python', 'factorial'): 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
    ('python', 'generate fibonacci'): 'def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a',
    ('python', 'fibonacci'): 'def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a',
    ('python', 'check prime'): 'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True',
    ('python', 'prime'): 'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True',
    ('python', 'remove duplicates'): 'def remove_duplicates(lst):\n    return list(dict.fromkeys(lst))',
    ('python', 'duplicate'): 'def remove_duplicates(lst):\n    return list(dict.fromkeys(lst))',
    ('python', 'merge dictionaries'): 'def merge_dicts(d1, d2):\n    return {**d1, **d2}',
    ('python', 'merge dicts'): 'def merge_dicts(d1, d2):\n    return {**d1, **d2}',
    ('python', 'binary search'): 'def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1',
    ('python', 'search'): 'def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1',
    ('python', 'sort'): 'def sort_list(lst):\n    return sorted(lst)',
    ('python', 'capitalize words'): "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())",
    ('python', 'capitalize'): "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())",
    ('python', 'filter even'): 'def filter_even(lst):\n    return [x for x in lst if x % 2 == 0]',
    ('python', 'filter'): 'def filter_even(lst):\n    return [x for x in lst if x % 2 == 0]',
    ('python', 'average'): 'def average(values):\n    return sum(values) / len(values) if values else 0',
    ('python', 'mean'): 'def average(values):\n    return sum(values) / len(values) if values else 0',
    ('python', 'max'): 'def find_max(lst):\n    return max(lst) if lst else None',
    ('python', 'minimum'): 'def find_min(lst):\n    return min(lst) if lst else None',
    ('python', 'min'): 'def find_min(lst):\n    return min(lst) if lst else None',
    ('python', 'count'): 'def count_items(lst):\n    return len(lst)',

    # JavaScript implementations
    ('javascript', 'sum'): 'function sumAll(arr) {\n    return arr.reduce((a, b) => a + b, 0);\n}',
    ('javascript', 'sum all elements'): 'function sumAll(arr) {\n    return arr.reduce((a, b) => a + b, 0);\n}',
    ('javascript', 'flatten array'): 'function flatten(arr) {\n    return arr.flat(Infinity);\n}',
    ('javascript', 'flatten'): 'function flatten(arr) {\n    return arr.flat(Infinity);\n}',
    ('javascript', 'find maximum'): 'function findMax(arr) {\n    return Math.max(...arr);\n}',
    ('javascript', 'maximum'): 'function findMax(arr) {\n    return Math.max(...arr);\n}',
    ('javascript', 'max'): 'function findMax(arr) {\n    return Math.max(...arr);\n}',
    ('javascript', 'find minimum'): 'function findMin(arr) {\n    return Math.min(...arr);\n}',
    ('javascript', 'minimum'): 'function findMin(arr) {\n    return Math.min(...arr);\n}',
    ('javascript', 'min'): 'function findMin(arr) {\n    return Math.min(...arr);\n}',
    ('javascript', 'sort descending'): 'function sortDesc(arr) {\n    return arr.sort((a, b) => b - a);\n}',
    ('javascript', 'sort'): 'function sortDesc(arr) {\n    return arr.sort((a, b) => b - a);\n}',
    ('javascript', 'remove duplicates'): 'function unique(arr) {\n    return [...new Set(arr)];\n}',
    ('javascript', 'duplicate'): 'function unique(arr) {\n    return [...new Set(arr)];\n}',
    ('javascript', 'filter even'): 'function filterEven(arr) {\n    return arr.filter(n => n % 2 === 0);\n}',
    ('javascript', 'filter'): 'function filterEven(arr) {\n    return arr.filter(n => n % 2 === 0);\n}',
    ('javascript', 'reverse'): "function reverseString(str) {\n    return str.split('').reverse().join('');\n}",
    ('javascript', 'debounce'): 'function debounce(fn, ms = 300) {\n    let timer;\n    return (...args) => {\n        clearTimeout(timer);\n        timer = setTimeout(() => fn(...args), ms);\n    };\n}',
    ('javascript', 'group by'): 'function groupBy(arr, key) {\n    return arr.reduce((acc, obj) => {\n        (acc[obj[key]] = acc[obj[key]] || []).push(obj);\n        return acc;\n    }, {});\n}',
    ('javascript', 'groupby'): 'function groupBy(arr, key) {\n    return arr.reduce((acc, obj) => {\n        (acc[obj[key]] = acc[obj[key]] || []).push(obj);\n        return acc;\n    }, {});\n}',
    ('javascript', 'map double'): 'function mapDouble(arr) {\n    return arr.map(n => n * 2);\n}',
    ('javascript', 'mapdouble'): 'function mapDouble(arr) {\n    return arr.map(n => n * 2);\n}',
    ('javascript', 'count'): 'function countItems(arr) {\n    return arr.length;\n}',
    ('javascript', 'average'): 'function average(arr) {\n    return arr.reduce((a, b) => a + b, 0) / arr.length;\n}',
    ('javascript', 'palindrome'): "function isPalindrome(str) {\n    return str === str.split('').reverse().join('');\n}",
    ('javascript', 'factorial'): 'function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}',
    ('javascript', 'fibonacci'): 'function fibonacci(n) {\n    let a = 0, b = 1;\n    for (let i = 0; i < n; i++) {\n        [a, b] = [b, a + b];\n    }\n    return a;\n}',
    ('javascript', 'prime'): 'function isPrime(n) {\n    if (n < 2) return false;\n    for (let i = 2; i <= Math.sqrt(n); i++) {\n        if (n % i === 0) return false;\n    }\n    return true;\n}',

    # TypeScript implementations
    ('typescript', 'interface'): None,  # Keep as-is, these are usually correct

    # Bash implementations
    ('bash', 'ssh'): 'ssh user@server \'df -h\'',
    ('bash', 'disk'): 'df -h',
    ('bash', 'git'): 'git checkout -b feature/branch && git commit -m "feat: implement" && git push origin HEAD',
}


def detect_mismatch(prompt, code, lang):
    """Detect if prompt and code are mismatched."""
    p = prompt.lower()
    c = code.lower()
    func_names = re.findall(r'def\s+(\w+)', c)
    func_names += re.findall(r'function\s+(\w+)', c)
    class_names = re.findall(r'class\s+(\w+)', c)

    prompt_keywords = {
        'sum': ['sum', 'add', 'total'],
        'max': ['max', 'maximum', 'largest'],
        'min': ['min', 'minimum', 'smallest'],
        'flatten': ['flatten', 'flat'],
        'sort': ['sort', 'sorted'],
        'reverse': ['reverse'],
        'filter': ['filter'],
        'count': ['count'],
        'average': ['average', 'mean'],
        'factorial': ['factorial'],
        'fibonacci': ['fibonacci', 'fib'],
        'prime': ['prime'],
        'palindrome': ['palindrome'],
        'duplicate': ['duplicate', 'deduplicate', 'unique'],
        'merge': ['merge'],
        'split': ['split'],
        'search': ['search', 'find', 'lookup', 'binary_search'],
        'ssh': ['ssh'],
        'disk': ['disk', 'df'],
    }

    for kw, related in prompt_keywords.items():
        if kw in p:
            found = any(r in fn for fn in func_names for r in related)
            found = found or any(r in c for r in related)
            if not found and (func_names or class_names):
                return kw

    return None


def fix_sample(text):
    """Fix a single coding sample if it has a mismatch."""
    parts = extract_parts(text)
    if parts is None:
        return text, False

    prompt, lang, code, suffix = parts
    mismatch_kw = detect_mismatch(prompt, code, lang)

    if mismatch_kw is None:
        return text, False

    # Find correct implementation
    p_lower = prompt.lower()
    fixed_code = None

    # Try exact keyword match first
    for (impl_lang, impl_key), impl_code in IMPLEMENTATIONS.items():
        if impl_lang == lang and impl_key in p_lower:
            if impl_code is not None:
                fixed_code = impl_code
                break

    if fixed_code is None:
        # Try the mismatch keyword
        for (impl_lang, impl_key), impl_code in IMPLEMENTATIONS.items():
            if impl_lang == lang and impl_key == mismatch_kw:
                if impl_code is not None:
                    fixed_code = impl_code
                    break

    if fixed_code is None:
        return text, False  # Can't fix

    # Reconstruct
    fixed_text = f"{prompt}\n```{lang}\n{fixed_code}\n```{suffix}"
    return fixed_text, True


def process_file(filepath):
    """Process a single JSONL file."""
    records = []
    fixed_count = 0
    total = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            text = rec.get('text', '')
            fixed_text, was_fixed = fix_sample(text)

            if was_fixed:
                rec['text'] = fixed_text
                rec['hash'] = hashlib.sha256(fixed_text.encode('utf-8')).hexdigest()
                fixed_count += 1

            records.append(rec)

    with open(filepath, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return total, fixed_count


def main():
    data_dir = Path("data/coding")
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl",
              "calibration.jsonl", "qualification.jsonl"]

    print("Fixing coding data prompt-code mismatches (comprehensive)...")
    print()

    total_fixed = 0
    total_samples = 0

    for split in splits:
        filepath = data_dir / split
        if not filepath.exists():
            continue
        total, fixed = process_file(filepath)
        total_fixed += fixed
        total_samples += total
        print(f"  {split}: {fixed}/{total} fixed ({100*fixed/total:.1f}%)")

    print(f"\nTotal: {total_fixed}/{total_samples} fixed ({100*total_fixed/total_samples:.1f}%)")

    # Verify remaining mismatches
    print("\nVerifying remaining mismatches...")
    for split in splits:
        filepath = data_dir / split
        if not filepath.exists():
            continue
        records = [json.loads(l) for l in open(filepath) if l.strip()]
        remaining = 0
        for rec in records:
            parts = extract_parts(rec.get('text', ''))
            if parts is None:
                continue
            prompt, lang, code, _ = parts
            if detect_mismatch(prompt, code, lang):
                remaining += 1
        print(f"  {split}: {remaining}/{len(records)} remaining mismatches")


if __name__ == "__main__":
    main()
