from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_import_contact_alignment():
    import contact_alignment  # noqa: F401


def test_scan_helpers_available():
    from contact_alignment import db200k_scan

    assert hasattr(db200k_scan, "AlignmentResult")
