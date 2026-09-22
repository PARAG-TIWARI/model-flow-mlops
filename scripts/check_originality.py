"""Originality and clean-room audit script for ModelFlow MLOps."""

import os
import sys
from pathlib import Path

FORBIDDEN_TERMS = [
    "MOHD-OMER",
    "omer022",
    "mlops-news-classifier",
    "news-classification",
    "ag_news",
    "truthlens",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def audit_project():
    """Scan all source, configuration, and documentation files for reference remnants."""
    print("=" * 60)
    print("  ModelFlow MLOps — Clean-Room & Originality Audit")
    print("=" * 60)
    print(f"Scanning directory: {PROJECT_ROOT}\n")

    violations = []
    scanned_files = 0

    ignore_dirs = {".git", ".pytest_cache", "__pycache__", "node_modules", "dist", ".dvc"}

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            # Skip binary artifacts or lock files
            if file.endswith((".joblib", ".pkl", ".png", ".jpg", ".db", ".whl", ".pyc")):
                continue
            filepath = Path(root) / file
            scanned_files += 1
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                for term in FORBIDDEN_TERMS:
                    if term.lower() in content.lower():
                        # Exclude self check script itself
                        if file == "check_originality.py":
                            continue
                        violations.append((str(filepath.relative_to(PROJECT_ROOT)), term))
            except Exception as e:
                print(f"Warning reading {filepath}: {e}")

    print(f"Audit completed: {scanned_files} files scanned.")
    if violations:
        print(f"\n[FAIL] Found {len(violations)} forbidden reference remnants:")
        for f, term in violations:
            print(f"  - {f} (contains '{term}')")
        sys.exit(1)
    else:
        print("\n[SUCCESS] 0 reference remnants detected! Project is 100% original.")
        sys.exit(0)


if __name__ == "__main__":
    audit_project()
