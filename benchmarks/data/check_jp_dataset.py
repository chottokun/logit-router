import json
from collections import Counter
from pathlib import Path


def main():
    dataset_path = Path(__file__).parent / "jp_large_eval_cases.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_cases = len(data)

    # Check for duplicates
    ids = [case["id"] for case in data]
    duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]

    # Check labels/choices
    domains = Counter()
    difficulties = Counter()
    tags = Counter()

    invalid_labels = []

    for case in data:
        domains[case["domain"]] += 1
        difficulties[case["difficulty"]] += 1

        for tag in case.get("tags", []):
            tags[tag] += 1

        if case["expected_choice"] not in case["choices"]:
            invalid_labels.append(case["id"])

    stats_md = f"""# JP Large Eval Cases Statistics

## Overview
- **Total Cases:** {total_cases}
- **Duplicate IDs:** {len(duplicate_ids)}
- **Invalid Expected Choices (not in choices):** {len(invalid_labels)}

## Domain Distribution
"""
    for domain, count in domains.most_common():
        stats_md += f"- **{domain}:** {count}\n"

    stats_md += "\n## Difficulty Distribution\n"
    for difficulty, count in difficulties.most_common():
        stats_md += f"- **{difficulty}:** {count}\n"

    stats_md += "\n## Tag Distribution\n"
    for tag, count in tags.most_common():
        stats_md += f"- **{tag}:** {count}\n"

    stats_path = Path(__file__).parent / "jp_large_eval_stats.md"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_md)

    print(f"Stats saved to {stats_path}")
    if duplicate_ids:
        print(f"Found {len(duplicate_ids)} duplicate IDs.")
    if invalid_labels:
        print(f"Found {len(invalid_labels)} cases with invalid expected_choice.")


if __name__ == "__main__":
    main()
