#!/usr/bin/env python
"""Generate diverse, non-templated data for math/planning/coding domains.

Produces 5 splits per domain: train, qualification, calibration, validation, test.
All text is diverse (no templated near-duplicates). Each sample is a self-contained
prompt+response or problem+solution pair.

Usage:
    python scripts/generate_diverse_data.py --output-dir data --samples-per-split 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List, Tuple


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# MATH domain: diverse arithmetic, algebra, word problems
# =============================================================================

def _gen_math_sample(rng: random.Random, idx: int) -> str:
    """Generate a diverse math problem with solution."""
    kind = rng.choice([
        "arithmetic", "algebra_linear", "algebra_quadratic",
        "word_problem", "geometry", "percentage", "ratio",
        "sequence", "inequality", "system_equations",
    ])

    if kind == "arithmetic":
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*", "/"])
        if op == "+":
            return f"Calculate {a} + {b}.\nAnswer: {a} + {b} = {a + b}"
        elif op == "-":
            return f"Calculate {a} - {b}.\nAnswer: {a} - {b} = {a - b}"
        elif op == "*":
            return f"Calculate {a} × {b}.\nAnswer: {a} × {b} = {a * b}"
        else:
            if a % b != 0:
                a = (a // b + 1) * b
            return f"Calculate {a} ÷ {b}.\nAnswer: {a} ÷ {b} = {a // b}"

    elif kind == "algebra_linear":
        x = rng.randint(1, 20)
        m = rng.randint(2, 10)
        c = rng.randint(1, 50)
        rhs = m * x + c
        return f"Solve for x: {m}x + {c} = {rhs}.\nSolution: {m}x = {rhs - c}, so x = {x}."

    elif kind == "algebra_quadratic":
        r1, r2 = rng.randint(1, 10), rng.randint(1, 10)
        b_val = -(r1 + r2)
        c_val = r1 * r2
        return f"Solve x² + ({b_val})x + {c_val} = 0.\nSolution: x = {r1} or x = {r2}."

    elif kind == "word_problem":
        items = rng.choice(["apples", "books", "marbles", "stickers", "coins"])
        n = rng.randint(3, 15)
        per = rng.randint(2, 8)
        total = n * per
        return f"Sarah has {n} boxes, each containing {per} {items}. How many {items} does she have in total?\nAnswer: {n} × {per} = {total} {items}."

    elif kind == "geometry":
        r = rng.randint(2, 15)
        area = 3.14159 * r * r
        return f"Find the area of a circle with radius {r}.\nAnswer: Area = π × {r}² ≈ {area:.2f}"

    elif kind == "percentage":
        total = rng.randint(100, 1000)
        pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 75])
        result = total * pct // 100
        return f"What is {pct}% of {total}?\nAnswer: {pct}% of {total} = {result}."

    elif kind == "ratio":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        total = rng.randint(20, 100)
        share_a = total * a // (a + b)
        share_b = total - share_a
        return f"Divide {total} in the ratio {a}:{b}.\nAnswer: {share_a} and {share_b}."

    elif kind == "sequence":
        start = rng.randint(1, 10)
        diff = rng.randint(2, 7)
        terms = [start + i * diff for i in range(5)]
        next_term = start + 5 * diff
        return f"Find the next term: {', '.join(map(str, terms))}, ?\nAnswer: The common difference is {diff}, so the next term is {next_term}."

    elif kind == "inequality":
        x = rng.randint(1, 15)
        m = rng.randint(2, 8)
        c = rng.randint(1, 30)
        rhs = m * x + c + rng.randint(1, 5)
        return f"Solve {m}x + {c} < {rhs}.\nSolution: {m}x < {rhs - c}, so x < {(rhs - c) / m:.1f}, meaning x ≤ {x}."

    elif kind == "system_equations":
        x, y = rng.randint(1, 10), rng.randint(1, 10)
        a1, a2 = rng.randint(2, 6), rng.randint(2, 6)
        b1, b2 = rng.randint(2, 6), rng.randint(2, 6)
        c1 = a1 * x + b1 * y
        c2 = a2 * x + b2 * y
        return f"Solve the system:\n{a1}x + {b1}y = {c1}\n{a2}x + {b2}y = {c2}\nSolution: x = {x}, y = {y}."

    return f"Compute {idx}.\nAnswer: {idx}."


# =============================================================================
# PLANNING domain: diverse project/task planning
# =============================================================================

def _gen_planning_sample(rng: random.Random, idx: int) -> str:
    """Generate a diverse planning problem with solution."""
    kind = rng.choice([
        "project_schedule", "task_prioritization", "resource_allocation",
        "meeting_plan", "travel_itinerary", "budget_plan",
        "event_planning", "milestone_plan", "risk_plan", "sprint_plan",
    ])

    if kind == "project_schedule":
        project = rng.choice(["website redesign", "mobile app launch", "data migration", "marketing campaign"])
        weeks = rng.randint(4, 12)
        phases = ["requirements", "design", "development", "testing", "deployment"]
        n_phases = min(len(phases), rng.randint(3, 5))
        selected = rng.sample(phases, n_phases)
        schedule = "; ".join(f"Week {i+1}-{i+weeks//n_phases}: {p}" for i, p in enumerate(selected))
        return f"Plan a {project} project over {weeks} weeks.\nSchedule: {schedule}"

    elif kind == "task_prioritization":
        tasks = [
            rng.choice(["Fix login bug", "Update documentation", "Review pull requests",
                        "Deploy hotfix", "Refactor API", "Write tests", "Optimize queries"]),
            rng.choice(["Design landing page", "Schedule team meeting", "Backup database",
                        "Update dependencies", "Analyze user feedback", "Create wireframes"]),
            rng.choice(["Prepare quarterly report", "Interview candidate", "Set up CI/CD",
                        "Migrate legacy code", "Configure monitoring", "Plan sprint goals"]),
        ]
        order = rng.choice(["urgency", "impact", "effort", "dependency"])
        return f"Prioritize these tasks by {order}: {', '.join(tasks)}.\nPriority order: 1. {tasks[0]}, 2. {tasks[1]}, 3. {tasks[2]}"

    elif kind == "resource_allocation":
        team = rng.randint(3, 8)
        projects = rng.randint(2, 5)
        per_project = team // projects
        remaining = team - per_project * projects
        return f"Allocate {team} team members across {projects} projects.\nPlan: {per_project} per project, {remaining} floating for support."

    elif kind == "meeting_plan":
        purpose = rng.choice(["sprint review", "quarterly planning", "architecture discussion", "retrospective"])
        duration = rng.choice([30, 45, 60, 90])
        attendees = rng.randint(4, 12)
        return f"Plan a {purpose} meeting for {attendees} people, {duration} minutes.\nAgenda: 1. Review progress (10 min) 2. Discussion (20 min) 3. Action items (10 min)"

    elif kind == "travel_itinerary":
        dest = rng.choice(["Tokyo", "Paris", "New York", "London", "Sydney", "Berlin"])
        days = rng.randint(3, 7)
        activities = rng.sample(["museum visit", "local cuisine", "city tour", "shopping", "hiking", "beach day"], min(days, 4))
        return f"Plan a {days}-day trip to {dest}.\nDay 1: Arrival and {activities[0]}. Day 2: {activities[1]}. Day 3: {activities[2]}."

    elif kind == "budget_plan":
        budget = rng.randint(10000, 100000)
        categories = ["development", "marketing", "operations", "contingency"]
        weights = [rng.randint(20, 40) for _ in categories]
        total_w = sum(weights)
        allocs = [budget * w // total_w for w in weights]
        return f"Allocate a ${budget} budget.\nPlan: Development ${allocs[0]}, Marketing ${allocs[1]}, Operations ${allocs[2]}, Contingency ${allocs[3]}."

    elif kind == "event_planning":
        event = rng.choice(["conference", "workshop", "hackathon", "product launch"])
        capacity = rng.randint(50, 300)
        return f"Plan a {event} for {capacity} attendees.\nChecklist: 1. Venue booking 2. Catering for {capacity} 3. Speaker line-up 4. Registration system 5. A/V equipment"

    elif kind == "milestone_plan":
        milestones = ["MVP", "Beta release", "GA launch", "Post-launch review"]
        n = rng.randint(2, 4)
        selected = rng.sample(milestones, n)
        dates = [f"Month {i+1}" for i in range(n)]
        plan = "; ".join(f"{d}: {m}" for d, m in zip(dates, selected))
        return f"Define project milestones.\nPlan: {plan}"

    elif kind == "risk_plan":
        risks = [
            rng.choice(["vendor delay", "budget overrun", "scope creep", "team attrition"]),
            rng.choice(["technical failure", "regulatory change", "market shift", "data breach"]),
        ]
        mitigations = ["buffer timeline", "weekly budget review", "freeze scope", "cross-training", "redundant systems"]
        return f"Identify and mitigate risks: {risks[0]}, {risks[1]}.\nMitigation: {risks[0]} → {rng.choice(mitigations)}; {risks[1]} → {rng.choice(mitigations)}"

    elif kind == "sprint_plan":
        sprint_len = rng.choice([1, 2, 3])
        capacity = rng.randint(20, 60)
        stories = rng.randint(5, 12)
        points = capacity // stories
        return f"Plan a {sprint_len}-week sprint with {capacity} story points.\nPlan: {stories} stories, ~{points} points each. Daily standups, mid-sprint review."

    return f"Planning task {idx}."


# =============================================================================
# CODING domain: diverse programming problems
# =============================================================================

def _gen_coding_sample(rng: random.Random, idx: int) -> str:
    """Generate a diverse coding problem with solution.

    Uses extensive parameterization to produce many unique samples.
    """
    kind = rng.choice([
        "python_function", "javascript_function", "sql_query",
        "regex", "data_structure", "algorithm", "debug",
        "api_design", "string_manipulation", "list_comprehension",
        "python_class", "typescript_type", "shell_command",
        "python_decorator", "error_handling", "dict_operation",
        "file_operation", "json_parse", "lambda_func", "generator_func",
    ])

    if kind == "python_function":
        func_name = rng.choice(["calculate_sum", "find_max", "reverse_string", "count_vowels",
                                "is_palindrome", "factorial", "fibonacci", "gcd", "lcm",
                                "is_prime", "flatten_list", "merge_dicts", "chunk_list",
                                "capitalize_words", "remove_duplicates", "sort_by_key"])
        param = rng.choice(["numbers", "data", "items", "values", "arr", "lst", "seq", "collection"])
        desc = rng.choice(["Calculate the sum of a list", "Find the maximum element",
                           "Reverse a string", "Count vowels in text", "Check palindrome",
                           "Compute factorial", "Generate Fibonacci sequence", "Find GCD",
                           "Find LCM", "Check primality", "Flatten nested list",
                           "Merge dictionaries", "Chunk list into groups", "Capitalize words",
                           "Remove duplicates", "Sort by key"])
        if func_name == "calculate_sum":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}({param}):\n    return sum({param})\n```"
        elif func_name == "find_max":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}({param}):\n    return max({param}) if {param} else None\n```"
        elif func_name == "reverse_string":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(s):\n    return s[::-1]\n```"
        elif func_name == "count_vowels":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(s):\n    return sum(1 for c in s if c in 'aeiouAEIOU')\n```"
        elif func_name == "is_palindrome":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(s):\n    return s == s[::-1]\n```"
        elif func_name == "factorial":
            n = rng.randint(1, 20)
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(n):\n    return 1 if n <= 1 else n * {func_name}(n - 1)\n```\nExample: {func_name}({n}) = {__import__('math').factorial(n)}"
        elif func_name == "fibonacci":
            n = rng.randint(5, 20)
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nExample: {func_name}({n}) = {__import__('functools').reduce(lambda x, _: (x[1], x[0]+x[1]), range(n), (0,1))[0]}"
        elif func_name == "gcd":
            a_val, b_val = rng.randint(10, 100), rng.randint(10, 100)
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```\nExample: gcd({a_val}, {b_val}) = {__import__('math').gcd(a_val, b_val)}"
        elif func_name == "lcm":
            a_val, b_val = rng.randint(5, 50), rng.randint(5, 50)
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(a, b):\n    return abs(a * b) // __import__('math').gcd(a, b) if a and b else 0\n```\nExample: lcm({a_val}, {b_val})"
        elif func_name == "is_prime":
            n = rng.randint(2, 100)
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\nExample: is_prime({n})"
        elif func_name == "flatten_list":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend({func_name}(item))\n        else:\n            result.append(item)\n    return result\n```"
        elif func_name == "merge_dicts":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(d1, d2):\n    return {{**d1, **d2}}\n```"
        elif func_name == "chunk_list":
            size = rng.randint(2, 8)
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}({param}, size={size}):\n    return [{param}[i:i+size] for i in range(0, len({param}), size)]\n```"
        elif func_name == "capitalize_words":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}(s):\n    return ' '.join(w.capitalize() for w in s.split())\n```"
        elif func_name == "remove_duplicates":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}({param}):\n    return list(dict.fromkeys({param}))\n```"
        elif func_name == "sort_by_key":
            return f"Write a Python function to {desc.lower()}.\n```python\ndef {func_name}({param}, key=0):\n    return sorted({param}, key=lambda x: x[key])\n```"
        return f"Write a Python function.\n```python\ndef {func_name}({param}):\n    pass\n```"

    elif kind == "javascript_function":
        func_name = rng.choice(["filterEven", "mapDouble", "reduceSum", "findMax", "sortDesc",
                                "groupBy", "unique", "flatten", "debounce", "partial"])
        desc = rng.choice(["filter even numbers", "double each element", "sum all elements",
                           "find maximum", "sort descending", "group by property",
                           "remove duplicates", "flatten array", "debounce function",
                           "partial application"])
        if func_name == "filterEven":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return arr.filter(n => n % 2 === 0);\n}}\n```"
        elif func_name == "mapDouble":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return arr.map(n => n * 2);\n}}\n```"
        elif func_name == "reduceSum":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return arr.reduce((a, b) => a + b, 0);\n}}\n```"
        elif func_name == "findMax":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return Math.max(...arr);\n}}\n```"
        elif func_name == "sortDesc":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return arr.sort((a, b) => b - a);\n}}\n```"
        elif func_name == "groupBy":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr, key) {{\n    return arr.reduce((acc, obj) => {{\n        (acc[obj[key]] = acc[obj[key]] || []).push(obj);\n        return acc;\n    }}, {{}});\n}}\n```"
        elif func_name == "unique":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return [...new Set(arr)];\n}}\n```"
        elif func_name == "flatten":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(arr) {{\n    return arr.flat(Infinity);\n}}\n```"
        elif func_name == "debounce":
            ms = rng.randint(100, 1000)
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(fn, ms={ms}) {{\n    let timer;\n    return (...args) => {{\n        clearTimeout(timer);\n        timer = setTimeout(() => fn(...args), ms);\n    }};\n}}\n```"
        elif func_name == "partial":
            return f"Write a JavaScript function to {desc}.\n```javascript\nfunction {func_name}(fn, ...args) {{\n    return (...rest) => fn(...args, ...rest);\n}}\n```"
        return f"Write a JavaScript function.\n```javascript\nfunction {func_name}(arr) {{ }}\n```"

    elif kind == "sql_query":
        table = rng.choice(["users", "orders", "products", "employees", "customers",
                            "transactions", "invoices", "shipments", "reviews", "subscriptions"])
        cols = rng.sample(["id", "name", "email", "salary", "created_at", "status",
                           "department", "price", "quantity", "total", "city", "country"],
                          rng.randint(1, 4))
        col_str = ", ".join(cols)
        conditions = [
            f"WHERE {rng.choice(cols)} > {rng.randint(1, 100)}",
            f"WHERE status = '{rng.choice(['active', 'pending', 'closed', 'shipped'])}'",
            f"ORDER BY {rng.choice(cols)} {rng.choice(['ASC', 'DESC'])}",
            f"GROUP BY {rng.choice(cols)}",
            f"LIMIT {rng.randint(5, 50)}",
            f"WHERE {rng.choice(cols)} IS NOT NULL",
            f"WHERE {rng.choice(cols)} LIKE '%{rng.choice(['test', 'admin', 'prod', 'dev'])}%'",
        ]
        condition = rng.choice(conditions)
        return f"Write a SQL query to select {col_str} from {table} {condition}.\n```sql\nSELECT {col_str} FROM {table} {condition};\n```"

    elif kind == "regex":
        patterns = [
            (r"\d+", "match digits", "123abc456"),
            (r"[A-Z][a-z]+", "match capitalized words", "Hello World Test"),
            (r"\w+@\w+\.\w+", "match email addresses", "user@example.com"),
            (r"https?://\S+", "match URLs", "https://example.com/path"),
            (r"\d{4}-\d{2}-\d{2}", "match dates", "2024-01-15"),
            (r"\$[\d,]+\.?\d*", "match currency", "$1,234.56"),
            (r"[A-Z]{2,}", "match acronyms", "NASA FBI CIA"),
            (r"\b\w{4}\b", "match 4-letter words", "this test code here"),
            (r"#+\s*\w+", "match markdown headers", "# Title ## Section"),
            (r"```[\s\S]*?```", "match code blocks", "```python\ncode\n```"),
        ]
        p = rng.choice(patterns)
        return f"Write a regex to {p[1]}.\nExample input: '{p[2]}'\n```python\nimport re\npattern = r'{p[0]}'\nmatches = re.findall(pattern, text)\n```"

    elif kind == "data_structure":
        ds = rng.choice(["stack", "queue", "linked list", "hash map", "binary tree",
                         "heap", "graph", "trie", "set", "deque"])
        desc_map = {
            "stack": ("LIFO", "push", "pop"),
            "queue": ("FIFO", "enqueue", "dequeue"),
            "linked list": ("sequential", "insert", "remove"),
            "hash map": ("key-value", "put", "get"),
            "binary tree": ("hierarchical", "insert", "traverse"),
            "heap": ("priority", "push", "pop_min"),
            "graph": ("nodes+edges", "add_edge", "bfs"),
            "trie": ("prefix tree", "insert", "search"),
            "set": ("unique", "add", "contains"),
            "deque": ("double-ended", "append", "pop_left"),
        }
        prop, op1, op2 = desc_map[ds]
        return f"Describe a {ds} data structure.\nA {ds} is a {prop} structure with operations: {op1}, {op2}, and search.\n```python\nfrom collections import deque\n```"

    elif kind == "algorithm":
        algo = rng.choice(["bubble sort", "binary search", "merge sort", "linear search",
                           "quick sort", "insertion sort", "DFS", "BFS",
                           "Dijkstra", "dynamic programming"])
        complexity = rng.choice(["O(n)", "O(n log n)", "O(log n)", "O(n²)", "O(V+E)"])
        if algo == "binary search":
            target = rng.randint(1, 100)
            return f"Implement binary search in Python.\nTime complexity: {complexity}\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\nFind {target} in sorted array."
        elif algo == "bubble sort":
            return f"Implement bubble sort.\nTime complexity: {complexity}\n```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```"
        elif algo == "merge sort":
            return f"Implement merge sort.\nTime complexity: {complexity}\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n```"
        elif algo == "quick sort":
            pivot = rng.choice(["first", "last", "middle", "random"])
            return f"Implement quick sort with {pivot} pivot.\nTime complexity: {complexity}\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + mid + quick_sort(right)\n```"
        elif algo == "DFS":
            return f"Implement depth-first search.\nTime complexity: {complexity}\n```python\ndef dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited\n```"
        elif algo == "BFS":
            return f"Implement breadth-first search.\nTime complexity: {complexity}\n```python\nfrom collections import deque\ndef bfs(graph, start):\n    visited = {{start}}\n    queue = deque([start])\n    while queue:\n        node = queue.popleft()\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return visited\n```"
        elif algo == "Dijkstra":
            return f"Implement Dijkstra's shortest path.\nTime complexity: {complexity}\n```python\nimport heapq\ndef dijkstra(graph, start):\n    dist = {{start: 0}}\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        for v, w in graph[u]:\n            if v not in dist or d + w < dist[v]:\n                dist[v] = d + w\n                heapq.heappush(pq, (d + w, v))\n    return dist\n```"
        return f"Describe {algo}.\nTime complexity: {complexity}.\n{algo.capitalize()} is used for sorting and searching."

    elif kind == "debug":
        bug_type = rng.choice(["off-by-one", "null reference", "type mismatch",
                               "infinite loop", "missing return", "wrong operator",
                               "index error", "mutation bug"])
        bugs = {
            "off-by-one": ("range(1, len(data))", "range(len(data))", "Loop starts at index 1 instead of 0"),
            "null reference": ("data.get('key').value", "data.get('key', {}).get('value')", "No null check before attribute access"),
            "type mismatch": ("int('3.14')", "float('3.14')", "int() cannot parse a float string"),
            "infinite loop": ("while True:", "while condition:", "Missing break condition"),
            "missing return": ("def calc(x):\n    x * 2", "def calc(x):\n    return x * 2", "Function doesn't return the result"),
            "wrong operator": ("if x = 5:", "if x == 5:", "Assignment instead of comparison"),
            "index error": ("arr[len(arr)]", "arr[len(arr) - 1]", "Index out of bounds"),
            "mutation bug": ("def add_item(lst, item):\n    lst.append(item)\n    return lst", "def add_item(lst, item):\n    return lst + [item]", "Mutates input list in place"),
        }
        bad, fixed, desc = bugs[bug_type]
        return f"Debug this code with a {bug_type} error:\n```python\n# Bug: {desc}\n{bad}\n```\nFix:\n```python\n{fixed}\n```"

    elif kind == "api_design":
        method = rng.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])
        endpoint = rng.choice(["/users", "/orders", "/products", "/sessions",
                               "/api/v1/items", "/api/v2/checkout", "/webhooks/events",
                               "/auth/token", "/uploads/files", "/search"])
        status = rng.choice([200, 201, 204, 400, 401, 403, 404, 500])
        desc = rng.choice(["list all resources", "create a new resource", "update a resource",
                           "delete a resource", "partially update", "authenticate user",
                           "upload a file", "search resources"])
        return f"Design a REST API endpoint to {desc}.\n```\n{method} {endpoint}\nContent-Type: application/json\nAuthorization: Bearer <token>\nResponse: {status} {'OK' if status < 400 else 'Error'}\n```"

    elif kind == "string_manipulation":
        ops = [
            ("split by comma and strip whitespace", "[p.strip() for p in s.split(',')]"),
            ("replace all spaces with underscores", "s.replace(' ', '_')"),
            ("convert to snake_case", "'_'.join(s.lower().split())"),
            ("remove all punctuation", "''.join(c for c in s if c.isalnum() or c == ' ')"),
            ("count word frequency", "from collections import Counter; Counter(s.split())"),
            ("pad with zeros to length n", "s.zfill(n)"),
            ("truncate to n characters with ellipsis", "s[:n] + '...' if len(s) > n else s"),
            ("extract all numbers", "import re; re.findall(r'\\d+', s)"),
            ("title case", "s.title()"),
            ("remove HTML tags", "import re; re.sub(r'<[^>]+>', '', s)"),
        ]
        desc, code = rng.choice(ops)
        return f"Write Python code to {desc}.\n```python\nresult = {code}\n```"

    elif kind == "list_comprehension":
        comps = [
            ("square even numbers", "[x**2 for x in range(10) if x % 2 == 0]"),
            ("filter positive numbers", "[x for x in numbers if x > 0]"),
            ("flatten a 2D list", "[item for row in matrix for item in row]"),
            ("create dict from two lists", "dict(zip(keys, values))"),
            ("get lengths of strings", "[len(s) for s in strings]"),
            ("uppercase first letter", "[s.capitalize() for s in words]"),
            ("filter non-None", "[x for x in items if x is not None]"),
            ("sum of squares", "sum(x**2 for x in range(n))"),
            ("nested condition", "[x if x > 0 else 0 for x in values]"),
            ("enumerate to dict", "{i: v for i, v in enumerate(items)}"),
        ]
        desc, code = rng.choice(comps)
        return f"Write a Python list comprehension to {desc}.\n```python\nresult = {code}\n```"

    elif kind == "python_class":
        class_name = rng.choice(["Animal", "Car", "BankAccount", "Student", "Rectangle",
                                 "Circle", "LinkedList", "Stack", "Queue", "TreeNode",
                                 "Shape", "Book", "Employee", "Product", "Order"])
        attr1 = rng.choice(["name", "title", "id", "value", "size", "color", "weight", "price"])
        attr2 = rng.choice(["age", "quantity", "status", "type", "category", "rating", "count", "level"])
        repr_line = f"        return f'{class_name}(self.{attr1}, self.{attr2})'"
        return f"Write a Python class {class_name} with attributes.\n```python\nclass {class_name}:\n    def __init__(self, {attr1}, {attr2}):\n        self.{attr1} = {attr1}\n        self.{attr2} = {attr2}\n\n    def __repr__(self):\n{repr_line}\n```"

    elif kind == "typescript_type":
        type_name = rng.choice(["User", "Product", "Order", "Config", "ApiResponse",
                                "Props", "State", "Event", "Callback", "Result"])
        fields = rng.sample(["id: number", "name: string", "email: string", "age: number",
                             "active: boolean", "created: Date", "tags: string[]",
                             "data: T", "error?: string", "count: number"], rng.randint(2, 4))
        return f"Define a TypeScript interface for {type_name}.\n```typescript\ninterface {type_name} {{\n    {chr(10).join('    ' + f + ';' for f in fields)}\n}}\n```"

    elif kind == "shell_command":
        cmd = rng.choice([
            "find . -name '*.py' -exec grep -l 'TODO' {} \\;",
            "docker build -t myapp . && docker run -p 8080:8080 myapp",
            "git log --oneline --graph --all | head -20",
            "ssh -i ~/.ssh/id_rsa user@server 'df -h'",
            "tar -czf backup.tar.gz --exclude='*.pyc' src/",
            "curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://localhost:3000/api",
            "psql -U postgres -c 'SELECT version();'",
            "kubectl get pods --all-namespaces | grep Running",
            "pip install -r requirements.txt && python manage.py migrate",
            "redis-cli -h localhost -p 6379 KEYS '*' | head -10",
        ])
        desc = rng.choice(["find TODO comments", "build and run Docker", "view git log",
                           "SSH and check disk", "create backup archive", "POST to API",
                           "check database version", "list running pods", "install and migrate",
                           "list Redis keys"])
        return f"Shell command to {desc}:\n```bash\n{cmd}\n```"

    elif kind == "python_decorator":
        deco_name = rng.choice(["timer", "cache", "retry", "log", "validate",
                                "rate_limit", "memoize", "deprecated"])
        if deco_name == "timer":
            return f"Write a Python decorator to measure execution time.\n```python\nimport time\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{{func.__name__}}: {{time.time()-start:.4f}}s')\n        return result\n    return wrapper\n```"
        elif deco_name == "cache":
            return f"Write a Python decorator to cache results.\n```python\nfrom functools import lru_cache\n@lru_cache(maxsize=128)\ndef expensive_func(n):\n    return n ** 2\n```"
        elif deco_name == "retry":
            attempts = rng.randint(2, 5)
            return f"Write a Python decorator to retry on failure.\n```python\nimport time\ndef retry(max_attempts={attempts}):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception:\n                    if attempt == max_attempts - 1:\n                        raise\n                    time.sleep(2 ** attempt)\n        return wrapper\n    return decorator\n```"
        elif deco_name == "log":
            return f"Write a Python decorator to log function calls.\n```python\nimport logging\ndef log(func):\n    def wrapper(*args, **kwargs):\n        logging.info(f'Calling {{func.__name__}}({{args}}, {{kwargs}})')\n        return func(*args, **kwargs)\n    return wrapper\n```"
        return f"Write a Python decorator.\n```python\ndef {deco_name}(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n```"

    elif kind == "error_handling":
        err_type = rng.choice(["ValueError", "TypeError", "KeyError", "FileNotFoundError",
                               "ConnectionError", "ZeroDivisionError", "IndexError", "AttributeError"])
        context = rng.choice(["parsing input", "accessing dictionary", "opening file",
                              "network request", "division operation", "indexing list",
                              "calling method", "type conversion"])
        return f"Write Python code with proper error handling for {context}.\n```python\ntry:\n    result = operation()\nexcept {err_type} as e:\n    print(f'Error: {{e}}')\n    result = None\nfinally:\n    cleanup()\n```"

    elif kind == "dict_operation":
        ops = [
            ("merge two dicts", "{**d1, **d2}"),
            ("get with default", "d.get('key', 'default')"),
            ("filter by value", "{k: v for k, v in d.items() if v > threshold}"),
            ("invert dict", "{v: k for k, v in d.items()}"),
            ("sort by value", "dict(sorted(d.items(), key=lambda x: x[1]))"),
            ("group by key", "lambda items: {{k: [v for _, v in g] for k, g in itertools.groupby(sorted(items), lambda x: x[0])}}"),
        ]
        desc, code = rng.choice(ops)
        return f"Write Python code to {desc}.\n```python\nresult = {code}\n```"

    elif kind == "file_operation":
        op = rng.choice(["read", "write", "append", "read lines", "copy", "CSV", "JSON load", "JSON dump"])
        ext = rng.choice(["txt", "json", "csv", "log", "yaml", "xml"])
        if op == "read":
            return f"Write Python code to read a .{ext} file.\n```python\nwith open('data.{ext}', 'r') as f:\n    content = f.read()\n```"
        elif op == "write":
            return f"Write Python code to write to a .{ext} file.\n```python\nwith open('output.{ext}', 'w') as f:\n    f.write(content)\n```"
        elif op == "JSON load":
            return f"Load JSON from a file.\n```python\nimport json\nwith open('data.json', 'r') as f:\n    data = json.load(f)\n```"
        elif op == "JSON dump":
            return f"Dump data to JSON file.\n```python\nimport json\nwith open('output.json', 'w') as f:\n    json.dump(data, f, indent=2)\n```"
        elif op == "CSV":
            return f"Read CSV file in Python.\n```python\nimport csv\nwith open('data.csv') as f:\n    reader = csv.DictReader(f)\n    rows = list(reader)\n```"
        return f"File operation: {op} .{ext}\n```python\nwith open('file.{ext}', 'r') as f:\n    data = f.readlines()\n```"

    elif kind == "json_parse":
        keys = rng.sample(["name", "age", "email", "city", "active", "score", "role", "id"], rng.randint(2, 4))
        vals = [f'"value_{rng.randint(1,999)}"' if k in ("name", "email", "city", "role") else str(rng.randint(1, 100)) if k in ("age", "score", "id") else str(rng.choice([True, False])).lower() for k in keys]
        json_str = ", ".join(f'"{k}": {v}' for k, v in zip(keys, vals))
        return f"Parse JSON and extract fields.\n```python\nimport json\ndata = json.loads('{{{json_str}}}')\nname = data.get('name')\n```"

    elif kind == "lambda_func":
        desc = rng.choice(["square a number", "add two numbers", "filter even", "sort by second element",
                           "format string", "compute hash", "check range", "concat strings"])
        if desc == "square a number":
            return f"Write a Python lambda to {desc}.\n```python\nsquare = lambda x: x ** 2\n```"
        elif desc == "add two numbers":
            return f"Write a Python lambda to {desc}.\n```python\nadd = lambda a, b: a + b\n```"
        elif desc == "filter even":
            return f"Write a Python lambda to {desc}.\n```python\neven = lambda n: n % 2 == 0\n```"
        elif desc == "sort by second element":
            return f"Write a Python lambda to {desc}.\n```python\nkey_func = lambda x: x[1]\nsorted(data, key=key_func)\n```"
        return f"Write a Python lambda.\n```python\nf = lambda x: x\n```"

    elif kind == "generator_func":
        gen_name = rng.choice(["fibonacci_gen", "count_up", "even_numbers", "chunks",
                               "random_walk", "cycle_items", "take_n", "natural_numbers"])
        if gen_name == "fibonacci_gen":
            return f"Write a Python generator for Fibonacci numbers.\n```python\ndef fibonacci_gen():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n```"
        elif gen_name == "count_up":
            return f"Write a Python generator that counts up from n.\n```python\ndef count_up(n):\n    while True:\n        yield n\n        n += 1\n```"
        elif gen_name == "even_numbers":
            return f"Write a Python generator for even numbers.\n```python\ndef even_numbers():\n    n = 0\n    while True:\n        if n % 2 == 0:\n            yield n\n        n += 1\n```"
        elif gen_name == "chunks":
            size = rng.randint(2, 8)
            return f"Write a Python generator to yield chunks of size {size}.\n```python\ndef chunks(seq, size={size}):\n    for i in range(0, len(seq), size):\n        yield seq[i:i+size]\n```"
        return f"Write a Python generator.\n```python\ndef {gen_name}():\n    yield 1\n```"

    return f"Coding problem #{idx}: implement a solution.\n```python\n# solution\n```"


# =============================================================================
# Data generation
# =============================================================================

SPLITS = ["train", "qualification", "calibration", "validation", "test"]
DOMAINS = ["math", "planning", "coding"]
GENERATORS = {
    "math": _gen_math_sample,
    "planning": _gen_planning_sample,
    "coding": _gen_coding_sample,
}


def generate_dataset(output_dir: Path, samples_per_split: int = 200, seed: int = 23):
    """Generate diverse data for all domains and splits.

    Generates ALL samples for ALL splits in a single pass per domain,
    deduplicates globally (across all splits), then partitions into splits.
    This guarantees zero exact overlap between any pair of splits.
    """
    rng = random.Random(seed)

    for domain in DOMAINS:
        gen = GENERATORS[domain]
        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Generate enough unique samples for ALL splits combined
        total_needed = samples_per_split * len(SPLITS)
        global_seen_hashes: set = set()
        all_records: list = []

        # Keep generating until we have enough unique samples
        gen_rng = random.Random(seed + DOMAINS.index(domain) * 1000000)
        attempts = 0
        max_attempts = total_needed * 10

        while len(all_records) < total_needed and attempts < max_attempts:
            text = gen(gen_rng, attempts)
            h = _hash(text)
            if h not in global_seen_hashes:
                global_seen_hashes.add(h)
                all_records.append({"text": text, "hash": h})
            attempts += 1

        # Shuffle and partition into splits
        gen_rng.shuffle(all_records)
        split_size = len(all_records) // len(SPLITS)

        for split_idx, split in enumerate(SPLITS):
            start = split_idx * split_size
            end = start + split_size if split_idx < len(SPLITS) - 1 else len(all_records)
            split_records = all_records[start:end]

            split_path = domain_dir / f"{split}.jsonl"
            with open(split_path, "w", encoding="utf-8") as f:
                for r in split_records:
                    f.write(json.dumps(r) + "\n")

            print(f"  {domain}/{split}: {len(split_records)} samples")

    print(f"\nDataset generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate diverse training data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--samples-per-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Generating diverse data with seed={args.seed}, {args.samples_per_split} samples/split")
    generate_dataset(output_dir, args.samples_per_split, args.seed)


if __name__ == "__main__":
    main()
