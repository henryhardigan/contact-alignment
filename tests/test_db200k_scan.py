from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contact_alignment import db200k_scan, design


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


def test_rescue_improves_center_without_zeroing_donor():
    # R-profile at positions 0 and 2, A-profile at position 1.
    # Window "AEA": A at pos 0 and 2 score 1.0 (unfavorable); E at pos 1 scores 0.5.
    # R-profiles allow rescue from D/E neighbors. E at pos 1 is a valid donor for pos 0.
    # Pos 0 rescues: center → -2.0 (E in R-profile). E@1 marked consumed, keeps its 0.5 score.
    # Pos 1 is consumed (was claimed as donor) → skipped as rescue center.
    # Pos 2 tries rescue: only neighbor is E@1 (consumed) → no rescue fires.
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]

    score, breakdown, donor_indices = db200k_scan.score_window_with_donor_trace("AEA", profiles)

    assert score == pytest.approx(-0.5)
    assert breakdown[0] == (1, "A<-E@2", -2.0)
    assert breakdown[1] == (2, "E", 0.5)
    assert breakdown[2] == (3, "A", 1.0)
    assert donor_indices == [1, 1, 2]


def test_score_window_fast_matches_traced_score():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]

    traced_score, _ = db200k_scan.score_window("AEA", profiles)
    fast_score = db200k_scan.score_window_fast("AEA", profiles)

    assert fast_score == traced_score


def test_score_window_rescue_none_returns_direct_sum():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]

    score, breakdown = db200k_scan.score_window("AEA", profiles, rescue_mode="none")
    fast_score = db200k_scan.score_window_fast("AEA", profiles, rescue_mode="none")

    assert score == 2.5
    assert fast_score == score
    assert breakdown == [(1, "A", 1.0), (2, "E", 0.5), (3, "A", 1.0)]


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


@pytest.mark.skipif(not db200k_scan.NUMBA_AVAILABLE, reason="Numba not installed")
def test_scan_records_numba_fast_matches_python_fast_scores():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]
    records = [
        (">target", "QQQAEAQQQ"),
        (">decoy", "QQQAAAQQQ"),
    ]

    python_hits, python_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=4,
        alignment_mode="rigid",
        fast_scan=True,
    )
    numba_hits, numba_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=4,
        alignment_mode="rigid",
        fast_scan=True,
        use_numba=True,
    )

    assert numba_windows == python_windows
    assert [(hit["header"], hit["start"], hit["end"], hit["score"]) for hit in numba_hits] == [
        (hit["header"], hit["start"], hit["end"], hit["score"]) for hit in python_hits
    ]


@pytest.mark.skipif(not db200k_scan.NUMBA_AVAILABLE, reason="Numba not installed")
def test_scan_records_numba_fast_matches_python_fast_scores_without_rescue():
    profiles = [
        make_profile(1, "R", {"A": 1.0, "E": -2.0}),
        make_profile(2, "A", {"E": 0.5}),
        make_profile(3, "R", {"A": 1.0, "E": -2.0}),
    ]
    records = [
        (">target", "QQQAEAQQQ"),
        (">decoy", "QQQAAAQQQ"),
    ]

    python_hits, python_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=4,
        alignment_mode="rigid",
        fast_scan=True,
        rescue_mode="none",
    )
    numba_hits, numba_windows = db200k_scan.scan_records(
        records,
        profiles,
        top_k=4,
        alignment_mode="rigid",
        fast_scan=True,
        use_numba=True,
        rescue_mode="none",
    )

    assert numba_windows == python_windows
    assert [(hit["header"], hit["start"], hit["end"], hit["score"]) for hit in numba_hits] == [
        (hit["header"], hit["start"], hit["end"], hit["score"]) for hit in python_hits
    ]


def test_scan_records_zero_progress_interval_is_treated_as_disabled():
    profiles = [
        make_profile(1, "E", {"L": -1.5}),
        make_profile(2, "T", {"R": -2.0}),
        make_profile(3, "S", {"L": -1.25}),
    ]
    records = [
        (">target", "QQQLRLQQQ"),
    ]

    hits, windows_scanned = db200k_scan.scan_records(
        records,
        profiles,
        top_k=3,
        alignment_mode="rigid",
        progress_every_windows=0,
    )

    assert windows_scanned == 7
    assert hits[0]["window"] == "LRL"


def test_beam_sample_favored_sequences_prefers_best_direct_choices():
    profiles = [
        make_profile(1, "A", {"L": -2.0, "V": -1.0}),
        make_profile(2, "A", {"R": -3.0, "K": -0.5}),
    ]

    candidates = design.beam_sample_favored_sequences(profiles, top_n=2, beam_width=4)

    assert candidates[0] == ("LR", -5.0)
    assert ("VK", -1.5) in candidates


def test_build_query_profiles_from_resources_matches_direct_builder():
    db_root = "/Users/henryhardigan/Downloads/pisces-cache-scaac"
    query_seq = "ETSEAKGPDGMALPRPR"

    resources = db200k_scan.load_profile_resources(
        db_root,
        profile_strategy="blend_3x3_1x1",
    )
    direct = db200k_scan.build_query_profiles(
        query_seq,
        db_root,
        profile_strategy="blend_3x3_1x1",
    )
    reused = db200k_scan.build_query_profiles_from_resources(
        query_seq,
        resources,
        profile_strategy="blend_3x3_1x1",
    )

    assert len(reused) == len(direct)
    for left, right in zip(reused, direct):
        assert left.position == right.position
        assert left.center_residue == right.center_residue
        assert left.support_mode == right.support_mode
        assert np.array_equal(left.energies, right.energies)
