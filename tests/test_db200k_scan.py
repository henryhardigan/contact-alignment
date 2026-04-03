from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contact_alignment import db200k_scan


def make_profile(
    position: int,
    center_residue: str,
    favored: dict[str, float],
) -> db200k_scan.PositionProfile:
    energies = np.full(len(db200k_scan.CENTER_ALPHABET), 1.0, dtype=float)
    for residue, energy in favored.items():
        energies[db200k_scan.CENTER_ALPHABET.index(residue)] = energy
    return db200k_scan.PositionProfile(
        position=position,
        center_residue=center_residue,
        center_residue_index=db200k_scan.RESIDUE_INDEX[center_residue],
        query_context=center_residue,
        energies=energies,
        energy_stds=np.ones_like(energies),
        count_5x5=0,
        count_3x3=0,
        count_1x1=1,
        geometry_buckets_5x5=0,
        geometry_buckets_3x3=0,
        geometry_buckets_1x1=1,
        weight_5x5=0.0,
        weight_3x3=0.0,
        weight_1x1=1.0,
        support_mode="test",
        effective_support=1.0,
        allows_charge_rescue=center_residue in db200k_scan.OFFSET_CHARGE_RESCUE_PAIRS,
        allows_aromatic_proline_rescue=center_residue in db200k_scan.AROMATIC_PROLINE_DIPEPTIDE_RESCUE_RESIDUES,
    )


def test_shared_donor_rescue_marks_second_claim_with_asterisk():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]

    score, breakdown, donor_indices = db200k_scan.score_window_with_donor_trace("AEA", profiles)

    assert score == -2.7
    assert breakdown[0] == (1, "A<-E@2", -2.0)
    assert breakdown[1] == (2, "E", 0.0)
    assert breakdown[2] == (3, "A<-E@2*", -0.7)
    assert donor_indices == [1, 1, 1]


def test_score_window_fast_matches_traced_score():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]

    traced_score, _ = db200k_scan.score_window("AEA", profiles)
    fast_score = db200k_scan.score_window_fast("AEA", profiles)

    assert fast_score == traced_score


def test_scan_records_rigid_recovers_expected_best_window():
    profiles = [
        make_profile(1, "E", {"L": -1.5}),
        make_profile(2, "T", {"R": -2.0}),
        make_profile(3, "S", {"L": -1.25}),
    ]

    records = [
        (">decoy", "AAAKKKAAA"),
        (">target", "QQQLRLQQQ"),
    ]

    hits, windows_scanned = db200k_scan.scan_records(
        records,
        profiles,
        top_k=3,
        alignment_mode="rigid",
    )

    assert windows_scanned == 14
    assert hits[0]["header"] == ">target"
    assert hits[0]["start"] == 4
    assert hits[0]["end"] == 6
    assert hits[0]["window"] == "LRL"
    assert hits[0]["score"] == -4.75


def test_scan_records_fast_scan_matches_standard_scores():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]
    records = [
        (">target", "QQQAEAQQQ"),
        (">decoy", "QQQAAAQQQ"),
    ]

    standard_hits, standard_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=4,
        alignment_mode="rigid",
    )
    fast_hits, fast_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=4,
        alignment_mode="rigid",
        fast_scan=True,
    )

    assert fast_windows == standard_windows
    assert [(hit["header"], hit["start"], hit["end"], hit["score"]) for hit in fast_hits] == [
        (hit["header"], hit["start"], hit["end"], hit["score"]) for hit in standard_hits
    ]
    assert all(hit["breakdown"] == [] for hit in fast_hits)


def test_scan_records_fast_scan_skips_invalid_windows_like_standard():
    profiles = [
        make_profile(1, "E", {"L": -1.5}),
        make_profile(2, "T", {"R": -2.0}),
        make_profile(3, "S", {"L": -1.25}),
    ]
    records = [
        (">mixed", "QQQLXRLQQQ"),
    ]

    standard_hits, standard_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=10,
        alignment_mode="rigid",
    )
    fast_hits, fast_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=10,
        alignment_mode="rigid",
        fast_scan=True,
    )

    assert fast_windows == standard_windows
    assert [(hit["start"], hit["end"], hit["window"], hit["score"]) for hit in fast_hits] == [
        (hit["start"], hit["end"], hit["window"], hit["score"]) for hit in standard_hits
    ]
