import argparse
import csv
import random
import sys
import math
import heapq
import itertools
from typing import Dict, Iterable, List, Optional, Set, Tuple
import unittest

# Standard 20 amino acids (canonical order for reproducibility)
AA20_STR = "ACDEFGHIKLMNPQRSTVWY"
AA20 = set(AA20_STR)
AROMATICS = {"W", "F", "Y"}
SMALL_NEUTRAL = {"A", "G"}
POS_CHARGES = {"K", "R"}
NEG_CHARGES = {"D", "E"}
HYDROPHOBES = {"A", "I", "L", "M", "V", "F", "W", "Y"}
HYDROPHOBE_OFFSET_WEIGHT = 0.6
APPLY_MJ_OVERRIDES = True

CLUSTAL_STRONG_GROUPS = [
    set("STA"),
    set("NEQK"),
    set("NHQK"),
    set("NDEQ"),
    set("QHRK"),
    set("MILV"),
    set("MILF"),
    set("HY"),
    set("FYW"),
]
CLUSTAL_WEAK_GROUPS = [
    set("CSA"),
    set("ATV"),
    set("SAG"),
    set("STNK"),
    set("STPA"),
    set("SGND"),
    set("SNDEQK"),
    set("NDEQHK"),
    set("NEQHRK"),
    set("FVLIM"),
    set("HFY"),
]


def apply_mj_overrides(a: str, b: str, val: Optional[float]) -> Optional[float]:
    """Apply custom MJ overrides (e.g., aromatics vs small residues)."""
    if not APPLY_MJ_OVERRIDES:
        return val
    if a in AROMATICS and b in SMALL_NEUTRAL:
        return -8.0
    if b in AROMATICS and a in SMALL_NEUTRAL:
        return -8.0
    return val


def proline_run_mask(seq: str) -> List[bool]:
    """Return a mask for positions in proline runs of length >= 2."""
    s = seq.strip().upper()
    mask = [False] * len(s)
    i = 0
    while i < len(s):
        if s[i] != "P":
            i += 1
            continue
        j = i + 1
        while j < len(s) and s[j] == "P":
            j += 1
        if j - i >= 2:
            for k in range(i, j):
                mask[k] = True
        i = j
    return mask


def proline_run_ids(seq: str) -> List[int]:
    """Return run IDs per position for proline runs (>=2), -1 otherwise."""
    s = seq.strip().upper()
    ids = [-1] * len(s)
    i = 0
    run_id = 0
    while i < len(s):
        if s[i] != "P":
            i += 1
            continue
        j = i + 1
        while j < len(s) and s[j] == "P":
            j += 1
        if j - i >= 2:
            for k in range(i, j):
                ids[k] = run_id
            run_id += 1
        i = j
    return ids


def alignment_gaps_cover_all_proline_runs_seq2(
    aln1: str,
    aln2: str,
    *,
    gap_char: str = "-",
) -> Tuple[bool, int]:
    """Return (ok, gap_count) requiring a gap on every PP+ run in seq2."""
    s1 = aln1.strip().upper()
    s2 = aln2.strip().upper()
    s1_nogap = "".join(c for c in s1 if c != gap_char)
    s2_nogap = "".join(c for c in s2 if c != gap_char)
    ids2 = proline_run_ids(s2_nogap)
    need2 = set(i for i in ids2 if i >= 0)
    i1 = 0
    i2 = 0
    gaps = 0
    for a, b in zip(s1, s2):
        if a == gap_char and b == gap_char:
            continue
        if a == gap_char:
            if i2 < len(ids2) and ids2[i2] >= 0:
                need2.discard(ids2[i2])
                gaps += 1
            i2 += 1
            continue
        if b == gap_char:
            i1 += 1
            continue
        i1 += 1
        i2 += 1
    if need2:
        return False, gaps
    return True, gaps


def force_gap_per_proline_run_seq2(
    aln1: str,
    aln2: str,
    *,
    gap_char: str = "-",
) -> Tuple[str, str, bool]:
    """Insert gaps in aln1 so every PP+ run in seq2 has at least one gap."""
    s1 = aln1.strip().upper()
    s2 = aln2.strip().upper()
    s2_nogap = "".join(c for c in s2 if c != gap_char)
    ids2 = proline_run_ids(s2_nogap)
    if not ids2:
        return s1, s2, False
    run_positions: Dict[int, List[int]] = {}
    i2 = 0
    for idx, b in enumerate(s2):
        if b == gap_char:
            continue
        if i2 < len(ids2) and ids2[i2] >= 0:
            run_positions.setdefault(ids2[i2], []).append(idx)
        i2 += 1
    insert_positions: List[int] = []
    for run_id, positions in run_positions.items():
        if not positions:
            continue
        if any(s1[pos] == gap_char for pos in positions):
            continue
        insert_positions.append(positions[0])
    if not insert_positions:
        return s1, s2, False
    s1_list = list(s1)
    s2_list = list(s2)
    for pos in sorted(insert_positions, reverse=True):
        s1_list.insert(pos, gap_char)
        s2_list.insert(pos, s2_list[pos])
    return "".join(s1_list), "".join(s2_list), True


def score_aligned_with_gaps(
    aln1: str,
    aln2: str,
    mj: Dict[Tuple[str, str], float],
    *,
    gap_open: float,
    gap_ext: float,
    gap_char: str = "-",
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> float:
    """Score aligned sequences using gap-open/gap-extend penalties."""
    s1 = aln1.strip().upper()
    s2 = aln2.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")
    total = 0.0
    gap1 = 0
    gap2 = 0
    for i, (a, b) in enumerate(zip(s1, s2), start=1):
        if a == gap_char and b == gap_char:
            continue
        if a == gap_char:
            total += gap_ext if gap1 else gap_open
            gap1 = 1
            gap2 = 0
            continue
        if b == gap_char:
            total += gap_ext if gap2 else gap_open
            gap2 = 1
            gap1 = 0
            continue
        gap1 = 0
        gap2 = 0
        val = _mj_pair_score(a, b, mj, unknown_policy)
        if val is None:
            continue
        total += float(val)
    if context_bonus:
        bonuses = context_bonus_aligned(
            s1, s2, mj, gap_char=gap_char, unknown_policy=unknown_policy
        )
        total += sum(bonuses)
    return total


def fmt_float(x: float, nd: int = 2) -> str:
    """Format a float with rounding but without unnecessary trailing zeros."""
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_pct(x: float, nd: int = 4) -> str:
    """Format a proportion in [0,1] with rounding, trimming trailing zeros."""
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def scanprosite_motif_from_anchors(seq_aln: str, anchors: Iterable[int]) -> str:
    """Build a ScanProsite-style regex motif from an aligned sequence and 1-based anchor indices.

    - Anchor positions: emit the concrete residue (single-letter AA)
    - Non-anchor (or gap/unknown) positions: emit 'x'
    - Consecutive 'x' runs are compressed as x(n) for n>1

    Example: R-x-R-x(2)-F-P
    """
    s = seq_aln.strip().upper()
    a_set = set(int(i) for i in anchors)

    tokens: List[str] = []
    for i, ch in enumerate(s, start=1):
        if i in a_set and ch in AA20:
            tokens.append(ch)
        else:
            tokens.append("x")

    # Trim leading/trailing x's so the motif doesn't start or end with x/x(n).
    while tokens and tokens[0] == "x":
        tokens.pop(0)
    while tokens and tokens[-1] == "x":
        tokens.pop()

    # Compress x-runs
    out: List[str] = []
    run = 0
    for t in tokens + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            out.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            out.append(t)

    return "-".join(out)


def combined_aligned_regex(
    seq1_aln: str,
    seq2_aln: str,
    *,
    gap_char: str = "-",
) -> str:
    """Combine two aligned sequences into a single ScanProsite-style regex."""
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")
    tokens: List[str] = []
    for a, b in zip(s1, s2):
        if a == gap_char or b == gap_char:
            tokens.append("x")
            continue
        if a not in AA20 or b not in AA20:
            tokens.append("x")
            continue
        if a == b:
            tokens.append(a)
        else:
            pair = "".join(sorted({a, b}))
            tokens.append("[" + pair + "]")

    while tokens and tokens[0] == "x":
        tokens.pop(0)
    while tokens and tokens[-1] == "x":
        tokens.pop()

    out: List[str] = []
    run = 0
    for t in tokens + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            out.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            out.append(t)

    return "-".join(out)


def _clustal_strong_pair(a: str, b: str) -> bool:
    return any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS)


def _clustal_any_pair(a: str, b: str) -> bool:
    return any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS + CLUSTAL_WEAK_GROUPS)


def combined_aligned_strong_regex(
    seq1_aln: str,
    seq2_aln: str,
    *,
    gap_char: str = "-",
) -> str:
    """Combine aligned sequences into a regex with strong-sim expansion."""
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")
    tokens: List[str] = []
    for a, b in zip(s1, s2):
        if a == gap_char or b == gap_char:
            tokens.append("x")
            continue
        if a not in AA20 or b not in AA20:
            tokens.append("x")
            continue
        if a == b:
            groups = [g for g in CLUSTAL_STRONG_GROUPS if a in g]
            if groups:
                common = set(groups[0])
                for g in groups[1:]:
                    common &= g
                if common and len(common) > 1:
                    tokens.append("[" + "".join(sorted(common)) + "]")
                else:
                    tokens.append(a)
            else:
                tokens.append(a)
            continue
        base = {a, b}
        if not _clustal_any_pair(a, b):
            tokens.append("x")
            continue
        expanded = set(base)
        use_strong = _clustal_strong_pair(a, b)
        for cand in AA20:
            count = 0
            for aa in base:
                if use_strong:
                    if _clustal_strong_pair(aa, cand):
                        count += 1
                else:
                    if _clustal_any_pair(aa, cand):
                        count += 1
            if count >= 2:
                expanded.add(cand)
        if len(expanded) == 1:
            tokens.append(next(iter(expanded)))
        else:
            tokens.append("[" + "".join(sorted(expanded)) + "]")

    while tokens and tokens[0] == "x":
        tokens.pop(0)
    while tokens and tokens[-1] == "x":
        tokens.pop()

    out: List[str] = []
    run = 0
    for t in tokens + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            out.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            out.append(t)

    return "-".join(out)


def clustal_similarity(
    seq1_aln: str,
    seq2_aln: str,
    *,
    gap_char: str = "-",
) -> Tuple[str, float, float, int]:
    """Compute a Clustal-style similarity line and score.

    Returns (symbols, score, score_norm, n_eligible).
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")
    symbols = []
    score = 0.0
    n_eligible = 0
    for a, b in zip(s1, s2):
        if a == gap_char or b == gap_char:
            symbols.append(" ")
            continue
        if a not in AA20 or b not in AA20:
            symbols.append(" ")
            continue
        n_eligible += 1
        if a == b:
            symbols.append("*")
            score += 1.0
            continue
        strong = any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS)
        if strong:
            symbols.append(":")
            score += 0.5
            continue
        weak = any(a in g and b in g for g in CLUSTAL_WEAK_GROUPS)
        if weak:
            symbols.append(".")
            score += 0.25
            continue
        symbols.append(" ")
    score_norm = score / n_eligible if n_eligible else 0.0
    return "".join(symbols), score, score_norm, n_eligible


def clustal_pair_score(a: str, b: str) -> Optional[float]:
    if a not in AA20 or b not in AA20:
        return None
    if a == b:
        return 1.0
    if any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS):
        return 0.5
    if any(a in g and b in g for g in CLUSTAL_WEAK_GROUPS):
        return 0.25
    return 0.0


def clustal_anchor_positions(
    s1: str, s2: str, mode: str = "strong"
) -> Tuple[List[int], List[int]]:
    """Return (anchors, eligible) 1-based positions for Clustal similarity anchors."""
    anchors: List[int] = []
    eligible: List[int] = []
    for i, (a, b) in enumerate(zip(s1, s2), start=1):
        if a not in AA20 or b not in AA20:
            continue
        eligible.append(i)
        is_id = (a == b)
        is_strong = any(a in g and b in g for g in CLUSTAL_STRONG_GROUPS)
        is_weak = any(a in g and b in g for g in CLUSTAL_WEAK_GROUPS)
        if mode == "identity":
            keep = is_id
        elif mode == "strong":
            keep = is_strong and not is_id
        elif mode == "strong+identity":
            keep = is_strong or is_id
        elif mode == "weak":
            keep = is_weak and not is_strong and not is_id
        elif mode == "any":
            keep = is_weak or is_strong or is_id
        else:
            raise ValueError(f"Unknown clustal anchor mode: {mode}")
        if keep:
            anchors.append(i)
    return anchors, eligible


def clustal_entry_key(header: str) -> Tuple[str, str]:
    """Return (base_id, entry_id) from a FASTA header."""
    head = header.split()[0]
    if "|" in head:
        parts = head.split("|")
        entry = parts[2] if len(parts) >= 3 else head
    else:
        entry = head
    base = entry.split("_")[0]
    return base, entry


def has_charge_run(seq: str, run_len: int = 3) -> bool:
    """Return True if seq contains positive (K/R) or negative (D/E) runs of length >= run_len."""
    if run_len <= 1:
        return False
    s = seq.strip().upper()
    count_pos = 0
    count_neg = 0
    for c in s:
        if c in "KR":
            count_pos += 1
            count_neg = 0
        elif c in "DE":
            count_neg += 1
            count_pos = 0
        else:
            count_pos = 0
            count_neg = 0
        if count_pos >= run_len or count_neg >= run_len:
            return True
    return False


def _kmer_hits(
    seq: str,
    k: int,
    kmer_set: Set[str],
    *,
    unknown_policy: str,
) -> List[bool]:
    hits: List[bool] = []
    if k <= 0 or len(seq) < k:
        return hits
    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in AA20 for c in kmer):
            if unknown_policy == "error":
                raise ValueError("Unknown residue in Clustal k-mer scan")
            hits.append(False)
            continue
        hits.append(kmer in kmer_set)
    return hits


def best_ungapped_window_pair_clustal(
    seq1: str,
    seq2: str,
    *,
    window: int,
    max_evals: int = 0,
    rng_seed: int = 1,
    unknown_policy: str = "error",
    prefilter_min_strong: int = 0,
    prefilter_min_identity: int = 0,
    kmer_len: int = 0,
    kmer_min: int = 1,
    require_positions2: Optional[List[Tuple[int, Set[str]]]] = None,
    rank_by: str = "score",
    filter_charge_runs: bool = False,
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    if window <= 0:
        return None, None, None
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return None, None, None

    prefix = None
    if kmer_len and kmer_len > 0 and kmer_len <= window:
        seq1_kmers = set()
        for i in range(0, n1 - kmer_len + 1):
            kmer = seq1[i : i + kmer_len]
            if any(c not in AA20 for c in kmer):
                if unknown_policy == "error":
                    raise ValueError("Unknown residue in Clustal k-mer scan")
                continue
            seq1_kmers.add(kmer)
        if not seq1_kmers:
            return None, None, None
        hits = _kmer_hits(seq2, kmer_len, seq1_kmers, unknown_policy=unknown_policy)
        if not hits:
            return None, None, None
        prefix = [0]
        for h in hits:
            prefix.append(prefix[-1] + (1 if h else 0))

    total_pairs = (n1 - window + 1) * (n2 - window + 1)
    use_sampling = max_evals and max_evals > 0 and total_pairs > max_evals

    best = float("-inf")
    best_ident = None
    best_i = None
    best_j = None

    if use_sampling:
        rng = random.Random(rng_seed)
        for _ in range(max_evals):
            i = rng.randrange(0, n1 - window + 1)
            j = rng.randrange(0, n2 - window + 1)
            if require_positions2:
                if any(seq2[j + off] not in allowed for off, allowed in require_positions2):
                    continue
            if prefix is not None:
                end = j + (window - kmer_len + 1)
                if end > len(prefix) - 1:
                    continue
                if prefix[end] - prefix[j] < kmer_min:
                    continue
            if filter_charge_runs:
                win1 = seq1[i : i + window]
                win2 = seq2[j : j + window]
                if has_charge_run(win1) or has_charge_run(win2):
                    continue
            s = 0.0
            invalid = False
            strong = 0
            ident = 0
            for k in range(window):
                a = seq1[i + k]
                b = seq2[j + k]
                v = clustal_pair_score(a, b)
                if v is None:
                    if unknown_policy == "error":
                        raise ValueError("Unknown residue in Clustal scan")
                    if unknown_policy == "skip":
                        invalid = True
                        break
                    v = 0.0
                if a == b and a in AA20:
                    ident += 1
                if v >= 0.5:
                    strong += 1
                s += v
            if invalid:
                continue
            if prefilter_min_strong and strong < prefilter_min_strong:
                continue
            if prefilter_min_identity and ident < prefilter_min_identity:
                continue
            if rank_by == "identity":
                if best_ident is None or ident > best_ident or (ident == best_ident and s > best):
                    best = s
                    best_ident = ident
                    best_i = i
                    best_j = j
            else:
                if s > best:
                    best = s
                    best_ident = ident
                    best_i = i
                    best_j = j
    else:
        for i in range(0, n1 - window + 1):
            for j in range(0, n2 - window + 1):
                if require_positions2:
                    if any(seq2[j + off] not in allowed for off, allowed in require_positions2):
                        continue
                if prefix is not None:
                    end = j + (window - kmer_len + 1)
                    if end > len(prefix) - 1:
                        continue
                    if prefix[end] - prefix[j] < kmer_min:
                        continue
                if filter_charge_runs:
                    win1 = seq1[i : i + window]
                    win2 = seq2[j : j + window]
                    if has_charge_run(win1) or has_charge_run(win2):
                        continue
                s = 0.0
                invalid = False
                strong = 0
                ident = 0
                for k in range(window):
                    a = seq1[i + k]
                    b = seq2[j + k]
                    v = clustal_pair_score(a, b)
                    if v is None:
                        if unknown_policy == "error":
                            raise ValueError("Unknown residue in Clustal scan")
                        if unknown_policy == "skip":
                            invalid = True
                            break
                        v = 0.0
                    if a == b and a in AA20:
                        ident += 1
                    if v >= 0.5:
                        strong += 1
                    s += v
                if invalid:
                    continue
                if prefilter_min_strong and strong < prefilter_min_strong:
                    continue
                if prefilter_min_identity and ident < prefilter_min_identity:
                    continue
                if rank_by == "identity":
                    if best_ident is None or ident > best_ident or (ident == best_ident and s > best):
                        best = s
                        best_ident = ident
                        best_i = i
                        best_j = j
                else:
                    if s > best:
                        best = s
                        best_ident = ident
                        best_i = i
                        best_j = j

    if best_i is None or best_j is None:
        return None, None, None, None
    return best, best_i, best_j, best_ident


def best_ungapped_window_pair_clustal_topk(
    seq1: str,
    seq2: str,
    *,
    window: int,
    max_evals: int = 0,
    rng_seed: int = 1,
    unknown_policy: str = "error",
    prefilter_min_strong: int = 0,
    prefilter_min_identity: int = 0,
    kmer_len: int = 0,
    kmer_min: int = 1,
    require_positions2: Optional[List[Tuple[int, Set[str]]]] = None,
    rank_by: str = "identity",
    topk: int = 3,
    filter_charge_runs: bool = False,
) -> List[Tuple[float, int, int, int]]:
    """Return top-k windows ranked by identity (or score) with tie-breaker score."""
    if window <= 0:
        return []
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return []

    prefix = None
    if kmer_len and kmer_len > 0 and kmer_len <= window:
        seq1_kmers = set()
        for i in range(0, n1 - kmer_len + 1):
            kmer = seq1[i : i + kmer_len]
            if any(c not in AA20 for c in kmer):
                if unknown_policy == "error":
                    raise ValueError("Unknown residue in Clustal k-mer scan")
                continue
            seq1_kmers.add(kmer)
        if not seq1_kmers:
            return []
        hits = _kmer_hits(seq2, kmer_len, seq1_kmers, unknown_policy=unknown_policy)
        if not hits:
            return []
        prefix = [0]
        for h in hits:
            prefix.append(prefix[-1] + (1 if h else 0))

    total_pairs = (n1 - window + 1) * (n2 - window + 1)
    use_sampling = max_evals and max_evals > 0 and total_pairs > max_evals
    rng = random.Random(rng_seed)

    heap: List[Tuple[Tuple[int, float], float, int, int, int]] = []

    def push_hit(score: float, ident: int, i: int, j: int) -> None:
        key = (ident, score) if rank_by == "identity" else (int(score * 1000), score)
        item = (key, score, ident, i, j)
        if len(heap) < topk:
            heapq.heappush(heap, item)
        else:
            if key > heap[0][0]:
                heapq.heapreplace(heap, item)

    if use_sampling:
        for _ in range(max_evals):
            i = rng.randrange(0, n1 - window + 1)
            j = rng.randrange(0, n2 - window + 1)
            if require_positions2:
                if any(seq2[j + off] not in allowed for off, allowed in require_positions2):
                    continue
            if prefix is not None:
                end = j + (window - kmer_len + 1)
                if end > len(prefix) - 1:
                    continue
                if prefix[end] - prefix[j] < kmer_min:
                    continue
            if filter_charge_runs:
                win1 = seq1[i : i + window]
                win2 = seq2[j : j + window]
                if has_charge_run(win1) or has_charge_run(win2):
                    continue
            s = 0.0
            invalid = False
            strong = 0
            ident = 0
            for k in range(window):
                a = seq1[i + k]
                b = seq2[j + k]
                v = clustal_pair_score(a, b)
                if v is None:
                    if unknown_policy == "error":
                        raise ValueError("Unknown residue in Clustal scan")
                    if unknown_policy == "skip":
                        invalid = True
                        break
                    v = 0.0
                if a == b and a in AA20:
                    ident += 1
                if v >= 0.5:
                    strong += 1
                s += v
            if invalid:
                continue
            if prefilter_min_strong and strong < prefilter_min_strong:
                continue
            if prefilter_min_identity and ident < prefilter_min_identity:
                continue
            push_hit(s, ident, i, j)
    else:
        for i in range(0, n1 - window + 1):
            for j in range(0, n2 - window + 1):
                if require_positions2:
                    if any(seq2[j + off] not in allowed for off, allowed in require_positions2):
                        continue
                if prefix is not None:
                    end = j + (window - kmer_len + 1)
                    if end > len(prefix) - 1:
                        continue
                    if prefix[end] - prefix[j] < kmer_min:
                        continue
                if filter_charge_runs:
                    win1 = seq1[i : i + window]
                    win2 = seq2[j : j + window]
                    if has_charge_run(win1) or has_charge_run(win2):
                        continue
                s = 0.0
                invalid = False
                strong = 0
                ident = 0
                for k in range(window):
                    a = seq1[i + k]
                    b = seq2[j + k]
                    v = clustal_pair_score(a, b)
                    if v is None:
                        if unknown_policy == "error":
                            raise ValueError("Unknown residue in Clustal scan")
                        if unknown_policy == "skip":
                            invalid = True
                            break
                        v = 0.0
                    if a == b and a in AA20:
                        ident += 1
                    if v >= 0.5:
                        strong += 1
                    s += v
                if invalid:
                    continue
                if prefilter_min_strong and strong < prefilter_min_strong:
                    continue
                if prefilter_min_identity and ident < prefilter_min_identity:
                    continue
                push_hit(s, ident, i, j)

    results = sorted(heap, key=lambda x: x[0], reverse=True)
    return [(score, ident, i, j) for _key, score, ident, i, j in results]


def overlaps_fraction(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Return overlap fraction relative to the shorter interval length."""
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    if right < left:
        return 0.0
    overlap = right - left + 1
    denom = min(a_end - a_start + 1, b_end - b_start + 1)
    if denom <= 0:
        return 0.0
    return overlap / denom


def clustal_search_fasta_global(
    seq1: str,
    fasta_path: str,
    *,
    window: int,
    max_evals: int,
    rng_seed: int,
    unknown_policy: str,
    prefilter_min_strong: int,
    prefilter_min_identity: int,
    kmer_len: int,
    kmer_min: int,
    require_positions2: Optional[List[Tuple[int, Set[str]]]] = None,
    topk: int,
    filter_charge_runs: bool = False,
) -> List[Tuple[float, float, int, int, str, str, str]]:
    rng = random.Random(rng_seed)
    hits: List[Tuple[float, float, int, int, str, str, str]] = []
    if max_evals <= 0:
        max_evals = 1
    if len(seq1) < window:
        return []

    records = []
    for name, seq in read_fasta_all(fasta_path):
        s = seq.upper()
        if len(s) < window:
            continue
        if any(c not in AA20 for c in s):
            if unknown_policy == "error":
                raise ValueError("Unknown residue in FASTA for Clustal scan")
            if unknown_policy == "skip":
                continue
        records.append((name, s))

    if not records:
        return []

    seq1_kmers = None
    if kmer_len and kmer_len > 0 and kmer_len <= window:
        seq1_kmers = set()
        for i in range(0, len(seq1) - kmer_len + 1):
            kmer = seq1[i : i + kmer_len]
            if any(c not in AA20 for c in kmer):
                if unknown_policy == "error":
                    raise ValueError("Unknown residue in Clustal k-mer scan")
                continue
            seq1_kmers.add(kmer)
        if not seq1_kmers:
            return []

    for _ in range(max_evals):
        name2, seq2 = rng.choice(records)
        i = rng.randrange(0, len(seq1) - window + 1)
        j = rng.randrange(0, len(seq2) - window + 1)
        if require_positions2:
            if any(seq2[j + off] not in allowed for off, allowed in require_positions2):
                continue
        if seq1_kmers is not None:
            ok = False
            for k in range(j, j + window - kmer_len + 1):
                kmer = seq2[k : k + kmer_len]
                if kmer in seq1_kmers:
                    ok = True
                    break
            if not ok:
                continue
        if filter_charge_runs:
            win1 = seq1[i : i + window]
            win2 = seq2[j : j + window]
            if has_charge_run(win1) or has_charge_run(win2):
                continue
        strong = 0
        ident = 0
        score = 0.0
        invalid = False
        for k in range(window):
            a = seq1[i + k]
            b = seq2[j + k]
            v = clustal_pair_score(a, b)
            if v is None:
                if unknown_policy == "error":
                    raise ValueError("Unknown residue in Clustal scan")
                if unknown_policy == "skip":
                    invalid = True
                    break
                v = 0.0
            if a == b and a in AA20:
                ident += 1
            if v >= 0.5:
                strong += 1
            score += v
        if invalid:
            continue
        if prefilter_min_strong and strong < prefilter_min_strong:
            continue
        if prefilter_min_identity and ident < prefilter_min_identity:
            continue
        win1 = seq1[i : i + window]
        win2 = seq2[j : j + window]
        _sym, _s, _sn, _n = clustal_similarity(win1, win2)
        hits.append((score, _sn, i, j, name2, win1, win2))

    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[: max(1, topk)]


def parse_scanprosite_pattern(pattern: str) -> List[str]:
    """Parse a simple ScanProsite-style pattern into per-position tokens.

    Supported tokens (dash-separated):
      - single AA letter (e.g., K)
      - x
      - x(n) where n is an integer > 0
    """
    tokens: List[str] = []
    for raw in pattern.strip().upper().split("-"):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "X":
            tokens.append("X")
            continue
        if raw.startswith("X(") and raw.endswith(")"):
            n_str = raw[2:-1].strip()
            if not n_str.isdigit():
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            n = int(n_str)
            if n <= 0:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            tokens.extend(["X"] * n)
            continue
        if len(raw) == 1 and raw in AA20:
            tokens.append(raw)
            continue
        raise ValueError(f"Unsupported ScanProsite token: {raw}")
    return tokens


def parse_clustal_require(s: Optional[str]) -> Optional[List[Tuple[int, Set[str]]]]:
    """Parse required positions like '3=W,4=L' into 0-based offsets and allowed sets."""
    if not s:
        return None
    items: List[Tuple[int, Set[str]]] = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid clustal require token: {raw}")
        pos_str, aa_str = raw.split("=", 1)
        pos_str = pos_str.strip()
        aa_str = aa_str.strip().upper()
        if not pos_str.isdigit():
            raise ValueError(f"Invalid clustal require position: {pos_str}")
        if not aa_str:
            raise ValueError("Empty clustal require residues")
        if aa_str.startswith("[") and aa_str.endswith("]"):
            aa_str = aa_str[1:-1].strip()
        if not aa_str or any(c not in AA20 for c in aa_str):
            raise ValueError(f"Invalid clustal require residues: {aa_str}")
        pos0 = int(pos_str) - 1
        if pos0 < 0:
            raise ValueError(f"Invalid clustal require position: {pos_str}")
        items.append((pos0, set(aa_str)))
    return items or None


def parse_scanprosite_pattern_with_sets(
    pattern: str,
) -> List[Optional[Set[str]]]:
    """Parse ScanProsite-style pattern into per-position residue sets.

    Supported tokens (dash-separated):
      - single AA letter (e.g., K)
      - [RK] bracket set
      - x
      - x(n) where n is an integer > 0
    Returns list where None means "any residue".
    """
    tokens: List[Optional[Set[str]]] = []
    for raw in pattern.strip().upper().split("-"):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "X":
            tokens.append(None)
            continue
        if raw.startswith("X(") and raw.endswith(")"):
            n_str = raw[2:-1].strip()
            if not n_str.isdigit():
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            n = int(n_str)
            if n <= 0:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            tokens.extend([None] * n)
            continue
        if raw.startswith("[") and raw.endswith("]") and len(raw) >= 3:
            body = raw[1:-1].strip()
            if not body:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            allowed = {c for c in body if c in AA20}
            if not allowed:
                raise ValueError(f"Invalid ScanProsite token: {raw}")
            tokens.append(allowed)
            continue
        if len(raw) == 1 and raw in AA20:
            tokens.append({raw})
            continue
        raise ValueError(f"Unsupported ScanProsite token: {raw}")
    return tokens


def scanprosite_forms_in_fasta(
    pattern: str,
    fasta_path: str,
    *,
    unknown_policy: str,
    name_filter: Optional[str] = None,
) -> Dict[str, int]:
    """Search a FASTA for all instances of a ScanProsite pattern.

    Returns a dict: matched form -> count.
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    if not tokens:
        return {}
    window = len(tokens)
    counts: Dict[str, int] = {}
    for _name, seq in read_fasta_all(fasta_path):
        if name_filter and name_filter not in _name:
            continue
        s = seq.upper()
        if len(s) < window:
            continue
        for i in range(0, len(s) - window + 1):
            ok = True
            for k, allowed in enumerate(tokens):
                aa = s[i + k]
                if aa not in AA20:
                    if unknown_policy == "skip":
                        ok = False
                        break
                    if unknown_policy == "error":
                        raise ValueError(
                            f"Unknown residue '{aa}' in FASTA at {fasta_path}"
                        )
                    if unknown_policy == "zero":
                        if allowed is not None:
                            ok = False
                            break
                        continue
                if allowed is not None and aa not in allowed:
                    ok = False
                    break
            if ok:
                form = s[i : i + window]
                counts[form] = counts.get(form, 0) + 1
    return counts


def scanprosite_expected_forms(pattern: str) -> Optional[Set[str]]:
    """Enumerate all possible concrete forms for a pattern.

    Returns None if the pattern contains any 'x' positions.
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    if not tokens:
        return set()
    for t in tokens:
        if t is None:
            return None
    pools: List[List[str]] = []
    for t in tokens:
        pools.append(sorted(t))  # type: ignore[arg-type]
    return {"".join(p) for p in itertools.product(*pools)}


def scanprosite_complement_motif(
    pattern: str,
    mj: Dict[Tuple[str, str], float],
    *,
    thr: float,
    set_mode: str = "all",
    context_offset: bool = False,
    top_k: int = 0,
) -> str:
    """Build a ScanProsite-style complement motif from a pattern and MJ threshold.

    For each concrete residue or bracket set in the pattern, emit a residue or
    bracket set that satisfies MJ(res, b) <= thr. For x positions, emit x.
    For bracket sets, the complement residue must satisfy all or any residues
    in the set, controlled by set_mode.
    """
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    out: List[str] = []
    aa_list = sorted(AA20)
    if set_mode not in {"all", "any"}:
        raise ValueError("set_mode must be 'all' or 'any'")
    def allowed_for_residue(res: str) -> Set[str]:
        allowed_set: Set[str] = set()
        for b in aa_list:
            val = mj.get((res, b), mj.get((b, res)))
            val = apply_mj_overrides(res, b, val)
            if val is None:
                continue
            if float(val) <= thr:
                allowed_set.add(b)
        return allowed_set

    def allowed_for_set(res_set: Set[str]) -> Set[str]:
        if set_mode == "all":
            allowed = set(aa_list)
            for aa in res_set:
                allowed &= allowed_for_residue(aa)
            return allowed
        allowed_any: Set[str] = set()
        for aa in res_set:
            allowed_any |= allowed_for_residue(aa)
        return allowed_any

    for idx, t in enumerate(tokens):
        if t is None:
            out.append("x")
            continue
        allowed = allowed_for_set(t)

        if context_offset:
            neighbor_sets: List[Set[str]] = []
            if idx - 1 >= 0 and tokens[idx - 1] is not None:
                neighbor_sets.append(tokens[idx - 1])  # type: ignore[arg-type]
            if idx + 1 < len(tokens) and tokens[idx + 1] is not None:
                neighbor_sets.append(tokens[idx + 1])  # type: ignore[arg-type]
            for nset in neighbor_sets:
                for aa in nset:
                    if aa in POS_CHARGES or aa in NEG_CHARGES or aa == "P" or aa in AROMATICS:
                        allowed |= allowed_for_residue(aa)

        if top_k and allowed:
            sources = set(t)
            if context_offset:
                for nset in neighbor_sets:
                    for aa in nset:
                        if aa in POS_CHARGES or aa in NEG_CHARGES or aa == "P" or aa in AROMATICS:
                            sources.add(aa)

            scored = []
            for b in sorted(allowed):
                best = None
                for aa in sources:
                    val = mj.get((aa, b), mj.get((b, aa)))
                    val = apply_mj_overrides(aa, b, val)
                    if val is None:
                        continue
                    v = float(val)
                    if best is None or v < best:
                        best = v
                if best is not None:
                    scored.append((best, b))

            scored.sort(key=lambda x: (x[0], x[1]))
            allowed = {b for _, b in scored[:top_k]}
        if not allowed:
            out.append("x")
        elif len(allowed) == 1:
            out.append(sorted(allowed)[0])
        else:
            out.append("[" + "".join(sorted(allowed)) + "]")
    # compress consecutive x into x(n)
    compressed: List[str] = []
    run = 0
    for t in out + ["__END__"]:
        if t == "x":
            run += 1
            continue
        if run:
            compressed.append("x" if run == 1 else f"x({run})")
            run = 0
        if t != "__END__":
            compressed.append(t)
    # Trim leading/trailing x runs so complement doesn't start/end with x/x(n)
    while compressed and compressed[0].startswith("x"):
        compressed.pop(0)
    while compressed and compressed[-1].startswith("x"):
        compressed.pop()
    return "-".join(compressed)


def avg_mj_score_pattern_to_seq(
    pattern: str,
    seq: str,
    mj: Dict[Tuple[str, str], float],
    *,
    unknown_policy: str,
) -> Optional[float]:
    """Average MJ score of a degenerate pattern against a concrete sequence."""
    tokens = parse_scanprosite_pattern_with_sets(pattern)
    s = seq.strip().upper()
    if len(tokens) != len(s):
        return None
    total = 0.0
    aa_list = sorted(AA20)
    for t, b in zip(tokens, s):
        if b not in AA20:
            if unknown_policy == "error":
                raise ValueError(f"Unknown residue {b!r} in sequence")
            if unknown_policy == "skip":
                return None
            # unknown_policy == "zero"
            continue
        allowed = aa_list if t is None else sorted(t)
        if not allowed:
            return None
        vals = []
        for a in allowed:
            val = mj.get((a, b), mj.get((b, a)))
            val = apply_mj_overrides(a, b, val)
            if val is None:
                continue
            vals.append(float(val))
        if not vals:
            return None
        total += sum(vals) / len(vals)
    return total


def get_mj_scorer(mj):
    """
    Return a callable score(a,b)->float for either:
      - dict keyed by (a,b)
      - object with .score(a,b)
    """
    if hasattr(mj, "score") and callable(getattr(mj, "score")):
        def _score(a, b):
            return apply_mj_overrides(a, b, mj.score(a, b))
        return _score

    if isinstance(mj, dict):
        def _score(a, b):
            # prefer direct key
            if (a, b) in mj:
                return apply_mj_overrides(a, b, mj[(a, b)])
            # fall back to reversed if the dict is stored the other way
            if (b, a) in mj:
                return apply_mj_overrides(a, b, mj[(b, a)])
            raise KeyError((a, b))
        return _score

    raise TypeError(f"Unsupported MJ matrix type: {type(mj)}")


def best_ungapped_window_pair(
    seq1,
    seq2,
    mj,
    window,
    mode="min",
    max_evals=0,
    rng_seed=1,
    unknown_policy="error",
    context_bonus: bool = False,
):
    """
    Scan all ungapped window pairs of fixed length and return the best pair.

    Returns (best_score, best_i, best_j, per_pos_scores) where:
      - best_i is 0-based start in seq1
      - best_j is 0-based start in seq2
      - per_pos_scores is a list of per-position MJ scores for the best window

    If no valid windows exist, returns (None, None, None, None).
    """
    if window <= 0:
        return None, None, None, None
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return None, None, None, None

    total_pairs = (n1 - window + 1) * (n2 - window + 1)
    use_sampling = max_evals and max_evals > 0 and total_pairs > max_evals

    # Localize for speed
    mj_score = get_mj_scorer(mj)
    s1 = seq1
    s2 = seq2
    w = window

    if mode == "min":
        best = float("inf")
        def better(x, y): return x < y
    else:
        best = float("-inf")
        def better(x, y): return x > y

    best_i = None
    best_j = None
    best_per = None
    use_context = context_bonus

    if use_sampling:
        import random
        rng = random.Random(rng_seed)
        for _ in range(max_evals):
            i = rng.randrange(0, n1 - w + 1)
            j = rng.randrange(0, n2 - w + 1)
            per = []
            s = 0.0
            for k in range(w):
                try:
                    v = mj_score(s1[i + k], s2[j + k])
                except KeyError:
                    if unknown_policy == "error":
                        raise
                    if unknown_policy == "skip":
                        per = None
                        break
                    v = 0.0
                per.append(v)
                s += v
            if per is None:
                continue
            if use_context:
                s += sum(
                    context_bonus_aligned(
                        s1[i : i + w],
                        s2[j : j + w],
                        mj,
                        unknown_policy=unknown_policy,
                    )
                )
            if better(s, best):
                best = s
                best_i = i
                best_j = j
                best_per = per
    else:
        for i in range(0, n1 - w + 1):
            for j in range(0, n2 - w + 1):
                per = []
                s = 0.0
                invalid = False
                for k in range(w):
                    try:
                        v = mj_score(s1[i + k], s2[j + k])
                    except KeyError:
                        if unknown_policy == "error":
                            raise
                        if unknown_policy == "skip":
                            invalid = True
                            break
                        v = 0.0
                    per.append(v)
                    s += v
                if invalid:
                    continue
                if use_context:
                    s += sum(
                        context_bonus_aligned(
                            s1[i : i + w],
                            s2[j : j + w],
                            mj,
                            unknown_policy=unknown_policy,
                        )
                    )
                if better(s, best):
                    best = s
                    best_i = i
                    best_j = j
                    best_per = per

    return best, best_i, best_j, best_per


def seed_windows(
    seq1: str,
    seq2: str,
    mj,
    *,
    window: int,
    score_max: float,
    kmax: int,
    kmin: int,
    rank_by_anchors: bool = False,
    anchor_thr: float = -25.0,
    prefilter_len: int = 0,
    prefilter_score_max: Optional[float] = None,
    prefilter_kmax: int = 0,
    prefilter_kmin: int = 0,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> List[Tuple[float, int, int, int]]:
    """Return seed windows (score, i, j, anchors) filtered by score_max and capped by kmax.

    If fewer than kmin seeds pass the score_max filter, the best kmin are kept.
    """
    if window <= 0:
        return []
    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return []

    # Use a faster numpy path when available (especially for large FASTA scans).
    # Anchor-first ranking needs per-position counts, so skip numpy in that case.
    if not rank_by_anchors:
        hits_np = _seed_windows_numpy(
            seq1,
            seq2,
            mj,
            window=window,
            score_max=score_max,
            kmax=kmax,
            kmin=kmin,
            prefilter_len=prefilter_len,
            prefilter_score_max=prefilter_score_max,
            prefilter_kmax=prefilter_kmax,
            prefilter_kmin=prefilter_kmin,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        if hits_np is not None:
            return [(s, i, j, 0) for (s, i, j) in hits_np]

    mj_score = get_mj_scorer(mj)

    def score_window(i: int, j: int, win: int) -> Optional[Tuple[float, int]]:
        s = 0.0
        anchors = 0
        for k in range(win):
            try:
                v = mj_score(seq1[i + k], seq2[j + k])
            except KeyError:
                if unknown_policy == "error":
                    raise
                if unknown_policy == "skip":
                    return None
                v = 0.0
            s += v
            if v <= anchor_thr:
                anchors += 1
        if context_bonus:
            s += sum(
                context_bonus_aligned(
                    seq1[i : i + win],
                    seq2[j : j + win],
                    mj,
                    unknown_policy=unknown_policy,
                )
            )
        return s, anchors

    hits: List[Tuple[float, int, int, int]] = []
    use_prefilter = (
        prefilter_len
        and prefilter_len > 0
        and prefilter_len < window
        and n1 >= prefilter_len
        and n2 >= prefilter_len
    )

    if use_prefilter:
        pf_score_max = prefilter_score_max if prefilter_score_max is not None else score_max
        pf_hits: List[Tuple[float, int, int]] = []
        for i in range(0, n1 - prefilter_len + 1):
            for j in range(0, n2 - prefilter_len + 1):
                r = score_window(i, j, prefilter_len)
                if r is None:
                    continue
                s, _a = r
                pf_hits.append((s, i, j))

        if not pf_hits:
            return []

        pf_hits.sort(key=lambda x: x[0])
        pf_filtered = [h for h in pf_hits if h[0] <= pf_score_max]

        if len(pf_filtered) < prefilter_kmin:
            pf_filtered = pf_hits[: min(prefilter_kmin, len(pf_hits))]

        if prefilter_kmax and len(pf_filtered) > prefilter_kmax:
            pf_filtered = pf_filtered[:prefilter_kmax]

        candidates = [(i, j) for _, i, j in pf_filtered]
        for i, j in candidates:
            if i + window > n1 or j + window > n2:
                continue
            r = score_window(i, j, window)
            if r is None:
                continue
            s, anchors = r
            hits.append((s, i, j, anchors))
    else:
        for i in range(0, n1 - window + 1):
            for j in range(0, n2 - window + 1):
                r = score_window(i, j, window)
                if r is None:
                    continue
                s, anchors = r
                hits.append((s, i, j, anchors))

    if not hits:
        return []

    if rank_by_anchors:
        hits.sort(key=lambda x: (-x[3], x[0]))  # most anchors, then most negative
        filtered = [h for h in hits if h[0] <= score_max]
    else:
        hits.sort(key=lambda x: x[0])  # most negative first
        filtered = [h for h in hits if h[0] <= score_max]

    if len(filtered) < kmin:
        filtered = hits[: min(kmin, len(hits))]

    if kmax and len(filtered) > kmax:
        filtered = filtered[:kmax]

    return filtered


def _seed_unpack(seed):
    score = seed[0]
    i = seed[1]
    j = seed[2]
    anchors = seed[3] if len(seed) > 3 else None
    return score, i, j, anchors


def _seed_windows_numpy(
    seq1: str,
    seq2: str,
    mj,
    *,
    window: int,
    score_max: float,
    kmax: int,
    kmin: int,
    prefilter_len: int,
    prefilter_score_max: Optional[float],
    prefilter_kmax: int,
    prefilter_kmin: int,
    unknown_policy: str,
    context_bonus: bool,
) -> Optional[List[Tuple[float, int, int]]]:
    try:
        import numpy as np
    except Exception:
        return None
    if context_bonus:
        return None

    n1 = len(seq1)
    n2 = len(seq2)
    if n1 < window or n2 < window:
        return []

    aa_to_idx = {aa: i for i, aa in enumerate(AA20_STR)}
    unk_idx = len(AA20_STR)
    s1_idx = np.fromiter(
        (aa_to_idx.get(ch, unk_idx) for ch in seq1.upper()),
        dtype=np.int16,
        count=n1,
    )
    s2_idx = np.fromiter(
        (aa_to_idx.get(ch, unk_idx) for ch in seq2.upper()),
        dtype=np.int16,
        count=n2,
    )

    if unknown_policy == "error":
        if (s1_idx == unk_idx).any() or (s2_idx == unk_idx).any():
            raise KeyError("Unknown residue encountered with --unknown error")

    mj_arr = np.zeros((unk_idx + 1, unk_idx + 1), dtype=np.float32)
    for i, a in enumerate(AA20_STR):
        for j, b in enumerate(AA20_STR):
            val = mj.get((a, b), mj.get((b, a)))
            val = apply_mj_overrides(a, b, val)
            if val is not None:
                mj_arr[i, j] = float(val)

    nwin = n2 - window + 1
    if nwin <= 0:
        return []

    s2_stride = s2_idx.strides[0]
    windows_full = np.lib.stride_tricks.as_strided(
        s2_idx, shape=(nwin, window), strides=(s2_stride, s2_stride)
    )
    invalid_full = None
    if unknown_policy == "skip":
        invalid_full = (windows_full == unk_idx).any(axis=1)

    use_prefilter = (
        prefilter_len
        and prefilter_len > 0
        and prefilter_len < window
        and n1 >= prefilter_len
        and n2 >= prefilter_len
    )
    if use_prefilter:
        pf_len = prefilter_len
        pf_score_max = prefilter_score_max if prefilter_score_max is not None else score_max
        pf_nwin = n2 - pf_len + 1
        windows_pf = np.lib.stride_tricks.as_strided(
            s2_idx, shape=(pf_nwin, pf_len), strides=(s2_stride, s2_stride)
        )
        invalid_pf = None
        if unknown_policy == "skip":
            invalid_pf = (windows_pf == unk_idx).any(axis=1)

    def heap_push(heap, score, i, j, max_size):
        key = -score
        if len(heap) < max_size:
            heapq.heappush(heap, (key, score, i, j))
            return
        if key > heap[0][0]:
            heapq.heapreplace(heap, (key, score, i, j))

    filtered_heap: List[Tuple[float, float, int, int]] = []
    overall_heap: List[Tuple[float, float, int, int]] = []
    all_filtered: List[Tuple[float, int, int]] = []
    filtered_count = 0

    for i in range(0, n1 - window + 1):
        s1w = s1_idx[i : i + window]
        if unknown_policy == "skip" and (s1w == unk_idx).any():
            continue

        if use_prefilter:
            s1_pf = s1_idx[i : i + pf_len]
            if unknown_policy == "skip" and (s1_pf == unk_idx).any():
                continue
            pf_scores = mj_arr[s1_pf[:, None], windows_pf.T].sum(axis=0)
            if invalid_pf is not None:
                pf_scores = pf_scores.copy()
                pf_scores[invalid_pf] = np.inf

            pf_mask = pf_scores <= pf_score_max
            if pf_mask.sum() < prefilter_kmin:
                if pf_nwin <= prefilter_kmin:
                    pf_idx = np.arange(pf_nwin, dtype=np.int64)
                else:
                    pf_idx = np.argpartition(pf_scores, prefilter_kmin - 1)[
                        :prefilter_kmin
                    ]
            else:
                pf_idx = np.nonzero(pf_mask)[0]
                if prefilter_kmax and len(pf_idx) > prefilter_kmax:
                    tmp = pf_scores.copy()
                    tmp[~pf_mask] = np.inf
                    pf_idx = np.argpartition(tmp, prefilter_kmax - 1)[
                        :prefilter_kmax
                    ]

            if len(pf_idx) == 0:
                continue
            pf_idx = pf_idx[pf_idx <= n2 - window]
            if len(pf_idx) == 0:
                continue

            windows_sel = windows_full[pf_idx]
            scores = mj_arr[s1w[:, None], windows_sel.T].sum(axis=0)
            if invalid_full is not None:
                invalid_sel = invalid_full[pf_idx]
                if invalid_sel.any():
                    scores = scores.copy()
                    scores[invalid_sel] = np.inf
            j_candidates = pf_idx
        else:
            scores = mj_arr[s1w[:, None], windows_full.T].sum(axis=0)
            if invalid_full is not None:
                scores = scores.copy()
                scores[invalid_full] = np.inf
            j_candidates = None

        if j_candidates is None:
            j_idx_all = np.arange(nwin, dtype=np.int64)
            j_idx = j_idx_all
        else:
            j_idx = j_candidates

        if kmin > 0:
            if len(scores) <= kmin:
                idx = np.arange(len(scores))
            else:
                idx = np.argpartition(scores, kmin - 1)[:kmin]
            for jj in idx:
                score = float(scores[jj])
                heap_push(overall_heap, score, i, int(j_idx[jj]), kmin)

        if score_max is not None:
            mask = scores <= score_max
            filtered_count += int(mask.sum())
            if kmax == 0:
                idx = np.nonzero(mask)[0]
                for jj in idx:
                    all_filtered.append((float(scores[jj]), i, int(j_idx[jj])))
            elif mask.any():
                if mask.sum() > kmax:
                    tmp = scores.copy()
                    tmp[~mask] = np.inf
                    idx = np.argpartition(tmp, kmax - 1)[:kmax]
                else:
                    idx = np.nonzero(mask)[0]
                for jj in idx:
                    score = float(scores[jj])
                    heap_push(filtered_heap, score, i, int(j_idx[jj]), kmax)

    if filtered_count >= kmin:
        if kmax == 0:
            hits = all_filtered
        else:
            hits = [(s, i, j) for _, s, i, j in filtered_heap]
    else:
        hits = [(s, i, j) for _, s, i, j in overall_heap]

    hits.sort(key=lambda x: x[0])
    return hits


def _mj_pair_score(a: str, b: str, mj, unknown_policy: str) -> Optional[float]:
    val = mj.get((a, b), mj.get((b, a)))
    val = apply_mj_overrides(a, b, val)
    if val is None:
        if unknown_policy == "error":
            raise ValueError(f"Unknown pair {a},{b}")
        if unknown_policy == "skip":
            return None
        return 0.0
    return float(val)


def _stage2_dp_tables(
    s1: str,
    s2: str,
    mj,
    *,
    anchor_i: int,
    anchor_j: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_in_seq1_only: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str,
    enforce_end_delta_zero: bool = False,
) -> Tuple[
    List[List[Optional[float]]],
    List[List[Optional[Tuple[int, int, int, str, int]]]],
    List[Dict[Tuple[int, int, int, str, int], Tuple[Tuple[int, int, int, str, int], str]]],
]:
    n1 = len(s1)
    n2 = len(s2)
    best_scores: List[List[Optional[float]]] = [
        [None] * (max_gaps + 1) for _ in range(max_len + 1)
    ]
    best_keys: List[List[Optional[Tuple[int, int, int, str, int]]]] = [
        [None] * (max_gaps + 1) for _ in range(max_len + 1)
    ]
    back: List[Dict[Tuple[int, int, int, str, int], Tuple[Tuple[int, int, int, str, int], str]]] = []
    dp: List[Dict[Tuple[int, int, int, str, int], float]] = []

    dp0: Dict[Tuple[int, int, int, str, int], float] = {}
    back0: Dict[Tuple[int, int, int, str, int], Tuple[Tuple[int, int, int, str, int], str]] = {}
    dp0[(anchor_i, anchor_j, 0, "S", 0)] = 0.0
    dp.append(dp0)
    back.append(back0)

    for t in range(1, max_len + 1):
        cur: Dict[Tuple[int, int, int, str, int], float] = {}
        cur_back: Dict[Tuple[int, int, int, str, int], Tuple[Tuple[int, int, int, str, int], str]] = {}
        prev = dp[t - 1]

        for (i, j, g, state, gl), prev_score in prev.items():
            delta_prev = j - i

            # Match
            if i < n1 and j < n2:
                delta_cur = (j + 1) - (i + 1)
                if abs(delta_cur) <= band:
                    val = _mj_pair_score(s1[i], s2[j], mj, unknown_policy)
                    if val is not None:
                        score = prev_score + val
                        if state != "S" and delta_cur != delta_prev:
                            score += switch_pen
                        key = (i + 1, j + 1, g, "M", 0)
                        if key not in cur or score < cur[key]:
                            cur[key] = score
                            cur_back[key] = ((i, j, g, state, gl), "M")

            # Gap in seq1 (advance j)
            if j < n2:
                delta_cur = (j + 1) - i
                if abs(delta_cur) <= band:
                    if state == "G1":
                        if gl < max_gap_len:
                            score = prev_score + gap_ext
                            if delta_cur != delta_prev:
                                score += switch_pen
                            key = (i, j + 1, g, "G1", gl + 1)
                            if key not in cur or score < cur[key]:
                                cur[key] = score
                                cur_back[key] = ((i, j, g, state, gl), "G1")
                    else:
                        if g < max_gaps:
                            score = prev_score + gap_open
                            if state != "S" and delta_cur != delta_prev:
                                score += switch_pen
                            key = (i, j + 1, g + 1, "G1", 1)
                            if key not in cur or score < cur[key]:
                                cur[key] = score
                                cur_back[key] = ((i, j, g, state, gl), "G1")

            # Gap in seq2 (advance i)
            if i < n1:
                delta_cur = j - (i + 1)
                if abs(delta_cur) <= band:
                    if not gap_in_seq1_only:
                        if state == "G2":
                            if gl < max_gap_len:
                                score = prev_score + gap_ext
                                if delta_cur != delta_prev:
                                    score += switch_pen
                                key = (i + 1, j, g, "G2", gl + 1)
                                if key not in cur or score < cur[key]:
                                    cur[key] = score
                                    cur_back[key] = ((i, j, g, state, gl), "G2")
                        else:
                            if g < max_gaps:
                                score = prev_score + gap_open
                                if state != "S" and delta_cur != delta_prev:
                                    score += switch_pen
                                key = (i + 1, j, g + 1, "G2", 1)
                                if key not in cur or score < cur[key]:
                                    cur[key] = score
                                    cur_back[key] = ((i, j, g, state, gl), "G2")

        dp.append(cur)
        back.append(cur_back)

        for key, score in cur.items():
            _, _, g, _, _ = key
            if g > max_gaps:
                continue
            if enforce_end_delta_zero:
                i, j, _, _, _ = key
                if (j - i) != 0:
                    continue
            if best_scores[t][g] is None or score < best_scores[t][g]:
                best_scores[t][g] = score
                best_keys[t][g] = key

    return best_scores, best_keys, back


def _stage2_backtrace(
    s1: str,
    s2: str,
    back: List[Dict[Tuple[int, int, int, str, int], Tuple[Tuple[int, int, int, str, int], str]]],
    key: Tuple[int, int, int, str, int],
    t: int,
) -> Tuple[str, str, Tuple[int, int]]:
    aln1 = []
    aln2 = []
    cur_key = key
    while t > 0:
        prev_key, move = back[t].get(cur_key, ((0, 0, 0, "S", 0), "S"))
        i, j, _, _, _ = cur_key
        if move == "M":
            aln1.append(s1[i - 1])
            aln2.append(s2[j - 1])
        elif move == "G1":
            aln1.append("-")
            aln2.append(s2[j - 1])
        elif move == "G2":
            aln1.append(s1[i - 1])
            aln2.append("-")
        else:
            break
        cur_key = prev_key
        t -= 1
    aln1 = "".join(reversed(aln1))
    aln2 = "".join(reversed(aln2))
    start_i, start_j, _, _, _ = cur_key
    return aln1, aln2, (start_i, start_j)


def stage2_extend_fixed_core(
    s1: str,
    s2: str,
    mj,
    *,
    anchor_i: int,
    anchor_j: int,
    seed_len: int,
    min_len: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_proline_force_runs_seq2: bool = False,
    gap_proline_force_post: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[Tuple[int, int]]]:
    if min_len <= 0 or max_len <= 0 or max_len < min_len:
        raise ValueError("Invalid length range for stage2 DP")
    n1 = len(s1)
    n2 = len(s2)
    if n1 == 0 or n2 == 0:
        return None, None, None, None
    if anchor_i < 0 or anchor_j < 0 or anchor_i > n1 or anchor_j > n2:
        return None, None, None, None

    if anchor_i + seed_len > n1 or anchor_j + seed_len > n2:
        return None, None, None, None
    if (gap_proline_force_runs_seq2 or gap_proline_force_post) and max_gaps <= 0:
        return None, None, None, None

    core_score = 0.0
    for k in range(seed_len):
        val = _mj_pair_score(
            s1[anchor_i + k], s2[anchor_j + k], mj, unknown_policy
        )
        if val is None:
            return None, None, None, None
        core_score += float(val)

    left1 = s1[:anchor_i][::-1]
    left2 = s2[:anchor_j][::-1]
    right1 = s1[anchor_i + seed_len :]
    right2 = s2[anchor_j + seed_len :]

    # Right DP (includes anchor at position 0)
    r_scores, r_keys, r_back = _stage2_dp_tables(
        right1,
        right2,
        mj,
        anchor_i=0,
        anchor_j=0,
        max_len=max_len,
        band=band,
        switch_pen=switch_pen,
        gap_open=gap_open,
        gap_ext=gap_ext,
        gap_in_seq1_only=gap_proline_force_runs_seq2,
        max_gaps=max_gaps,
        max_gap_len=max_gap_len,
        unknown_policy=unknown_policy,
    )

    # Left DP (reverse, anchor at 0)
    l_scores, l_keys, l_back = _stage2_dp_tables(
        left1,
        left2,
        mj,
        anchor_i=0,
        anchor_j=0,
        max_len=max_len,
        band=band,
        switch_pen=switch_pen,
        gap_open=gap_open,
        gap_ext=gap_ext,
        gap_in_seq1_only=gap_proline_force_runs_seq2,
        max_gaps=max_gaps,
        max_gap_len=max_gap_len,
        unknown_policy=unknown_policy,
        enforce_end_delta_zero=True,
    )

    # Include t_left=0 with score 0
    l_scores[0][0] = 0.0
    l_keys[0][0] = (0, 0, 0, "S", 0)

    best_score = None
    best_pair = None
    for t_left in range(0, max_len + 1):
        for g_left in range(0, max_gaps + 1):
            if l_scores[t_left][g_left] is None:
                continue
            if l_keys[t_left][g_left] is None:
                continue
            for t_right in range(0, max_len + 1):
                for g_right in range(0, max_gaps - g_left + 1):
                    if r_scores[t_right][g_right] is None:
                        continue
                    if r_keys[t_right][g_right] is None:
                        continue
                    total_len = seed_len + t_left + t_right
                    if total_len < min_len or total_len > max_len:
                        continue
                    score = (
                        core_score
                        + float(l_scores[t_left][g_left])
                        + float(r_scores[t_right][g_right])
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (t_left, g_left, t_right, g_right)

    if best_score is None or best_pair is None:
        return None, None, None, None

    t_left, g_left, t_right, g_right = best_pair
    aln1_r, aln2_r, start_r = _stage2_backtrace(
        right1, right2, r_back, r_keys[t_right][g_right], t_right
    )
    if t_left == 0:
        aln1_l = ""
        aln2_l = ""
        start_i = anchor_i
        start_j = anchor_j
    else:
        aln1_l_rev, aln2_l_rev, start_l_rev = _stage2_backtrace(
            left1, left2, l_back, l_keys[t_left][g_left], t_left
        )
        aln1_l = aln1_l_rev[::-1]
        aln2_l = aln2_l_rev[::-1]
        start_i = anchor_i - start_l_rev[0]
        start_j = anchor_j - start_l_rev[1]

    core_aln1 = s1[anchor_i : anchor_i + seed_len]
    core_aln2 = s2[anchor_j : anchor_j + seed_len]
    aln1 = aln1_l + core_aln1 + aln1_r
    aln2 = aln2_l + core_aln2 + aln2_r
    if gap_proline_force_runs_seq2:
        ok, gaps = alignment_gaps_cover_all_proline_runs_seq2(aln1, aln2)
        if not ok or gaps == 0:
            return None, None, None, None
    if gap_proline_force_post:
        forced1, forced2, changed = force_gap_per_proline_run_seq2(aln1, aln2)
        if not changed:
            return None, None, None, None
        forced_score = score_aligned_with_gaps(
            forced1,
            forced2,
            mj,
            gap_open=gap_open,
            gap_ext=gap_ext,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        return forced_score, forced1, forced2, (start_i, start_j)
    if context_bonus:
        best_score = score_aligned_with_gaps(
            aln1,
            aln2,
            mj,
            gap_open=gap_open,
            gap_ext=gap_ext,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
    return best_score, aln1, aln2, (start_i, start_j)


def stage2_best_from_seed(
    seq1: str,
    seq2: str,
    mj,
    *,
    seed_i: int,
    seed_j: int,
    seed_len: int,
    flank: int,
    min_len: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_proline_force_runs_seq2: bool = False,
    gap_proline_force_post: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str = "error",
    reanchor: bool = False,
    context_bonus: bool = False,
) -> Tuple[
    Optional[float],
    Optional[str],
    Optional[str],
    Optional[Tuple[int, int]],
    Optional[Tuple[int, int]],
]:
    s1_start = max(0, seed_i - flank)
    s1_end = min(len(seq1), seed_i + seed_len + flank)
    s2_start = max(0, seed_j - flank)
    s2_end = min(len(seq2), seed_j + seed_len + flank)

    sub1 = seq1[s1_start:s1_end]
    sub2 = seq2[s2_start:s2_end]

    if not reanchor:
        score, aln1, aln2, start = stage2_extend_fixed_core(
            sub1,
            sub2,
            mj,
            anchor_i=seed_i - s1_start,
            anchor_j=seed_j - s2_start,
            seed_len=seed_len,
            min_len=min_len,
            max_len=max_len,
            band=band,
            switch_pen=switch_pen,
            gap_open=gap_open,
            gap_ext=gap_ext,
            gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
            gap_proline_force_post=gap_proline_force_post,
            max_gaps=max_gaps,
            max_gap_len=max_gap_len,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        if score is None or start is None:
            return None, None, None, None, None

        start_i, start_j = start
        return (
            score,
            aln1,
            aln2,
            (s1_start + start_i, s2_start + start_j),
            (seed_i, seed_j),
        )

    best_score = None
    best = None
    best_anchor = None
    max_i = len(sub1) - seed_len
    max_j = len(sub2) - seed_len
    if max_i < 0 or max_j < 0:
        return None, None, None, None, None

    for ai in range(0, max_i + 1):
        for aj in range(0, max_j + 1):
            score, aln1, aln2, start = stage2_extend_fixed_core(
                sub1,
                sub2,
                mj,
                anchor_i=ai,
                anchor_j=aj,
                seed_len=seed_len,
                min_len=min_len,
                max_len=max_len,
                band=band,
                switch_pen=switch_pen,
                gap_open=gap_open,
                gap_ext=gap_ext,
                gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
                gap_proline_force_post=gap_proline_force_post,
                max_gaps=max_gaps,
                max_gap_len=max_gap_len,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
            if score is None or start is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best = (aln1, aln2, start)
                best_anchor = (ai, aj)

    if best_score is None or best is None or best_anchor is None:
        return None, None, None, None, None

    aln1, aln2, start = best
    start_i, start_j = start
    anchor_i, anchor_j = best_anchor
    return (
        best_score,
        aln1,
        aln2,
        (s1_start + start_i, s2_start + start_j),
        (s1_start + anchor_i, s2_start + anchor_j),
    )


def stage2_null_scores(
    seq1: str,
    seq2: str,
    mj,
    *,
    seed_i: int,
    seed_j: int,
    seed_len: int,
    flank: int,
    min_len: int,
    max_len: int,
    band: int,
    switch_pen: float,
    gap_open: float,
    gap_ext: float,
    gap_proline_force_runs_seq2: bool = False,
    gap_proline_force_post: bool = False,
    max_gaps: int,
    max_gap_len: int,
    unknown_policy: str,
    n: int,
    seed: Optional[int] = None,
    reanchor: bool = False,
    context_bonus: bool = False,
) -> List[float]:
    rng = random.Random(seed)
    s1_start = max(0, seed_i - flank)
    s1_end = min(len(seq1), seed_i + seed_len + flank)
    s2_start = max(0, seed_j - flank)
    s2_end = min(len(seq2), seed_j + seed_len + flank)
    sub1 = seq1[s1_start:s1_end]
    base2 = list(seq2[s2_start:s2_end])
    base1 = list(seq1[s1_start:s1_end])

    scores: List[float] = []
    for _ in range(n):
        rng.shuffle(base2)
        rng.shuffle(base1)
        sub1 = "".join(base1)
        sub2 = "".join(base2)
        if reanchor:
            max_i = len(sub1) - seed_len
            max_j = len(sub2) - seed_len
            best_score = None
            if max_i >= 0 and max_j >= 0:
                for ai in range(0, max_i + 1):
                    for aj in range(0, max_j + 1):
                        score, _, _, _ = stage2_extend_fixed_core(
                            sub1,
                            sub2,
                            mj,
                            anchor_i=ai,
                            anchor_j=aj,
                            seed_len=seed_len,
                            min_len=min_len,
                            max_len=max_len,
                            band=band,
                            switch_pen=switch_pen,
                            gap_open=gap_open,
                            gap_ext=gap_ext,
                            gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
                            gap_proline_force_post=gap_proline_force_post,
                            max_gaps=max_gaps,
                            max_gap_len=max_gap_len,
                            unknown_policy=unknown_policy,
                            context_bonus=context_bonus,
                        )
                        if score is None:
                            continue
                        if best_score is None or score < best_score:
                            best_score = score
            score = best_score
        else:
            score, _, _, _ = stage2_extend_fixed_core(
                sub1,
                sub2,
                mj,
                anchor_i=seed_i - s1_start,
                anchor_j=seed_j - s2_start,
                seed_len=seed_len,
                min_len=min_len,
                max_len=max_len,
                band=band,
                switch_pen=switch_pen,
                gap_open=gap_open,
                gap_ext=gap_ext,
                gap_proline_force_runs_seq2=gap_proline_force_runs_seq2,
                gap_proline_force_post=gap_proline_force_post,
                max_gaps=max_gaps,
                max_gap_len=max_gap_len,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
        if score is not None:
            scores.append(score)
    return scores


def fmt_prob(p: float) -> str:
    """Format a probability/p-value compactly.

    - Uses scientific notation for very small values.
    - Otherwise uses up to 6 decimals, trimmed.
    """
    if p != p:  # NaN
        return "nan"
    if p == 0.0:
        return "0"
    if p < 1e-4:
        return f"{p:.3e}"
    return fmt_pct(p, 6)


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard index for two collections of integers."""
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def hypergeom_p_at_least(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N, K, n).

    N: population size
    K: number of "success" states in population
    n: number of draws
    k: observed overlap (successes drawn)

    This is the appropriate exact probability when:
      - anchors in set A mark K "success" positions among N eligible positions
      - anchors in set B are modeled as n draws without replacement from those N positions
      - overlap count is the number of B anchors that land in A-success positions

    Returns 1.0 when inputs are degenerate or k <= 0.
    """
    if k <= 0:
        return 1.0
    if N <= 0 or K < 0 or n < 0 or K > N or n > N:
        return float("nan")
    max_x = min(K, n)
    if k > max_x:
        return 0.0

    denom = math.comb(N, n)
    num = 0
    for x in range(k, max_x + 1):
        num += math.comb(K, x) * math.comb(N - K, n - x)
    return num / denom


# ---------- MJ loading ----------

def load_mj_csv(path: str) -> Dict[Tuple[str, str], float]:
    """Load an MJ matrix from CSV.

    Expected format:
      - first row header: blank cell then amino-acid one-letter codes
      - first column: amino-acid one-letter codes
      - body: numeric values

    Returns dict keyed by (aa1, aa2) -> float.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.strip().upper() for h in header[1:]]
        mj: Dict[Tuple[str, str], float] = {}

        for row in reader:
            if not row or not row[0].strip():
                continue
            raa = row[0].strip().upper()
            for caa, v in zip(cols, row[1:]):
                v = v.strip()
                if v:
                    mj[(raa, caa)] = float(v)

    if not mj:
        raise ValueError(f"No MJ values loaded from {path!r}")
    return mj


def maybe_invert_mj(mj: Dict[Tuple[str, str], float]) -> Tuple[Dict[Tuple[str, str], float], bool, Tuple[float, float]]:
    """If matrix is non-negative, invert so 'lower is better' logic still works."""
    vals = list(mj.values())
    if not vals:
        return mj, False, (0.0, 0.0)
    vmin = min(vals)
    vmax = max(vals)
    if vmin >= 0.0:
        return {k: -v for k, v in mj.items()}, True, (vmin, vmax)
    return mj, False, (vmin, vmax)


# ---------- Core scoring ----------

def score_aligned(
    seq1_aln: str,
    seq2_aln: str,
    mj: Dict[Tuple[str, str], float],
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> Tuple[float, List[Optional[float]]]:
    """Score two *aligned* sequences position-by-position.

    - gap positions: ignored (per_pos=None) unless gap_penalty is provided
    - gap penalties are applied when gap_penalty is provided
    - unknown residues: controlled by unknown_policy: error|skip|zero

    Returns (total_score, per_pos_scores) where per_pos_scores is a list of:
      - float for scored positions
      - None for ignored positions (e.g., gaps when gap_penalty=None or unknown_policy=skip)
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    total = 0.0
    per_pos: List[Optional[float]] = []
    idx1 = 0
    idx2 = 0

    for i, (a, b) in enumerate(zip(s1, s2), start=1):
        idx1_cur = idx1 if a != gap_char else None
        idx2_cur = idx2 if b != gap_char else None
        if a == gap_char or b == gap_char:
            if gap_penalty is None:
                if a != gap_char:
                    idx1 += 1
                if b != gap_char:
                    idx2 += 1
                per_pos.append(None)
                continue
            total += gap_penalty
            per_pos.append(gap_penalty)
            if a != gap_char:
                idx1 += 1
            if b != gap_char:
                idx2 += 1
            continue

        val = mj.get((a, b), mj.get((b, a)))
        val = apply_mj_overrides(a, b, val)
        if val is None:
            if unknown_policy == "error":
                raise ValueError(f"Unknown pair {a},{b} at pos {i}")
            if unknown_policy == "skip":
                per_pos.append(None)
                continue
            # unknown_policy == "zero"
            val = 0.0

        total += val
        per_pos.append(val)
        idx1 += 1
        idx2 += 1

    if context_bonus:
        bonuses = context_bonus_aligned(
            s1, s2, mj, gap_char=gap_char, unknown_policy=unknown_policy
        )
        for i, bonus in enumerate(bonuses):
            if bonus == 0.0:
                continue
            if per_pos[i] is None:
                continue
            per_pos[i] = float(per_pos[i]) + bonus
            total += bonus

    return total, per_pos


def context_bonus_aligned(
    seq1_aln: str,
    seq2_aln: str,
    mj: Dict[Tuple[str, str], float],
    *,
    gap_char: str = "-",
    unknown_policy: str = "error",
) -> List[float]:
    """Compute context bonuses for ±1 interactions.

    Rules:
      - charge opposites (K/R vs D/E) within ±1: add 0.25 * MJ score
      - proline aligned to opposing ±1 aromatic: add 0.5 * MJ score
      - proline aligned to opposing ±1 proline: add 0.5 * MJ score
      - hydrophobe aligned to opposing ±1 hydrophobe: add 0.6 * MJ score
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")
    bonuses = [0.0] * len(s1)

    def pair_score(a: str, b: str) -> Optional[float]:
        if a == gap_char or b == gap_char:
            return None
        if a not in AA20 or b not in AA20:
            if unknown_policy == "error":
                raise ValueError(f"Unknown pair {a},{b}")
            if unknown_policy == "skip":
                return None
            return 0.0
        return _mj_pair_score(a, b, mj, unknown_policy)

    n = len(s1)
    for i in range(n):
        a = s1[i]
        b = s2[i]
        if a == gap_char or b == gap_char:
            continue
        if a not in AA20 or b not in AA20:
            continue

        # Charge opposites within ±1 (half weight per neighbor)
        if a in POS_CHARGES or a in NEG_CHARGES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                b2 = s2[j]
                if (a in POS_CHARGES and b2 in NEG_CHARGES) or (
                    a in NEG_CHARGES and b2 in POS_CHARGES
                ):
                    val = pair_score(a, b2)
                    if val is not None:
                        bonuses[i] += 0.25 * float(val)

        if b in POS_CHARGES or b in NEG_CHARGES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                a2 = s1[j]
                if (b in POS_CHARGES and a2 in NEG_CHARGES) or (
                    b in NEG_CHARGES and a2 in POS_CHARGES
                ):
                    val = pair_score(b, a2)
                    if val is not None:
                        bonuses[i] += 0.25 * float(val)

        # Proline aligned to opposing ±1 aromatic (full weight)
        if a == "P":
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                b2 = s2[j]
                if b2 in AROMATICS:
                    val = pair_score("P", b2)
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)
                if b2 == "P":
                    val = pair_score("P", "P")
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)
        if b == "P":
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                a2 = s1[j]
                if a2 in AROMATICS:
                    val = pair_score("P", a2)
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)
                if a2 == "P":
                    val = pair_score("P", "P")
                    if val is not None:
                        bonuses[i] += 0.5 * float(val)

        # Hydrophobe aligned to opposing ±1 hydrophobe
        if a in HYDROPHOBES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                b2 = s2[j]
                if b2 in HYDROPHOBES:
                    val = pair_score(a, b2)
                    if val is not None:
                        bonuses[i] += HYDROPHOBE_OFFSET_WEIGHT * float(val)
        if b in HYDROPHOBES:
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                a2 = s1[j]
                if a2 in HYDROPHOBES:
                    val = pair_score(b, a2)
                    if val is not None:
                        bonuses[i] += HYDROPHOBE_OFFSET_WEIGHT * float(val)

    return bonuses


# ---------- Top contributors ----------

def top_contributors(
    seq1_aln: str,
    seq2_aln: str,
    per_pos: List[Optional[float]],
    n: int = 10,
) -> List[Tuple[float, int, str, str]]:
    """Return the n most favorable (most negative) aligned positions."""
    s1 = seq1_aln
    s2 = seq2_aln
    hits: List[Tuple[float, int, str, str]] = []
    for i, v in enumerate(per_pos, start=1):
        if v is None:
            continue
        hits.append((float(v), i, s1[i - 1], s2[i - 1]))

    hits.sort(key=lambda x: x[0])  # most negative first
    return hits[: max(0, n)]


# ---------- Anchor-only scoring ----------

def anchors_by_threshold(per_pos: List[Optional[float]], thr: float = -25.0) -> List[int]:
    """Return 1-based indices where MJ <= thr (strong complements)."""
    return [i for i, v in enumerate(per_pos, start=1) if v is not None and v <= thr]


def anchor_score(
    per_pos: List[Optional[float]],
    anchors: Iterable[int],
) -> Tuple[float, List[Tuple[int, float]]]:
    """Sum MJ over provided 1-based anchor positions."""
    total = 0.0
    used: List[Tuple[int, float]] = []
    for i in anchors:
        if i < 1 or i > len(per_pos):
            continue
        v = per_pos[i - 1]
        if v is None:
            continue
        total += float(v)
        used.append((i, float(v)))
    return total, used


# ---------- Analytic anchor probability (Model C, uniform partner) ----------

def per_position_complement_prob_uniform(
    fixed_seq_aln: str,
    mj: Dict[Tuple[str, str], float],
    *,
    thr: float,
    gap_char: str = "-",
    unknown_policy: str = "error",
) -> List[Optional[float]]:
    """For each aligned position i, compute p_i = P(MJ(fixed[i], B) <= thr) with B ~ Uniform(20 aa).

    Returns a list of per-position probabilities (float in [0,1]) or None for positions that are not eligible
    (e.g., gaps in fixed_seq_aln).

    Notes:
      - This is a *taxon-agnostic* baseline: partner residues are uniform over the 20 standard amino acids.
      - unknown_policy matches score_aligned: error|skip|zero.
        * error: raises on unknown residue in fixed sequence
        * skip: returns None for that position
        * zero: treats unknown residue as having p_i = 0 (no complements)
    """
    s = fixed_seq_aln.strip().upper()
    out: List[Optional[float]] = []

    aa_list = sorted(AA20)
    for i, a in enumerate(s, start=1):
        if a == gap_char:
            out.append(None)
            continue
        if a not in AA20:
            if unknown_policy == 'error':
                raise ValueError(
                    f"Unknown residue {a!r} at pos {i} in fixed sequence")
            if unknown_policy == 'skip':
                out.append(None)
                continue
            # unknown_policy == 'zero'
            out.append(0.0)
            continue

        ok = 0
        for b in aa_list:
            val = mj.get((a, b), mj.get((b, a)))
            if val is None:
                # Should not happen for AA20 if matrix complete; treat per unknown_policy
                if unknown_policy == 'error':
                    raise ValueError(f"Missing MJ value for pair {a},{b}")
                if unknown_policy == 'skip':
                    continue
                val = 0.0
            if float(val) <= thr:
                ok += 1
        out.append(ok / 20.0)

    return out


def hmm_emission(state: str, s: float, tc: float, tp: float) -> float:
    """Emission score for HMM interface model."""
    score = 0.0
    if s >= 30:
        score -= 6.0
    elif s >= 20:
        score -= 3.0

    if state == "C":
        if s <= tc:
            score += 4.0 + 0.1 * (tc - s)
        else:
            score -= 8.0
    elif state == "P":
        if s <= tp:
            score += 2.0 + 0.05 * (tp - s)
        else:
            score -= 2.0
    else:  # B
        if s <= tp:
            score += 0.5
    return score


def hmm_interface_path(
    per_pos: List[Optional[float]],
    *,
    tc: float,
    tp: float,
) -> List[Optional[str]]:
    """Viterbi path for B/P/C states on per-position MJ scores.

    Gaps (None) are skipped and returned as None.
    """
    trans = {
        "B": {"B": 0.985, "P": 0.014, "C": 0.001},
        "P": {"B": 0.12, "P": 0.85, "C": 0.03},
        "C": {"B": 0.01, "P": 0.07, "C": 0.92},
    }
    logt = {a: {b: math.log(p) for b, p in trans[a].items()} for a in trans}
    states = ["B", "P", "C"]

    idxs = [i for i, v in enumerate(per_pos) if v is not None]
    if not idxs:
        return [None] * len(per_pos)

    dp = {s: float("-inf") for s in states}
    dp["B"] = 0.0
    back: List[Dict[str, str]] = []

    for i in idxs:
        s_val = float(per_pos[i])
        cur = {}
        back_i = {}
        for st in states:
            best_prev = None
            best_score = float("-inf")
            for prev in states:
                score = dp[prev] + logt[prev][st]
                if score > best_score:
                    best_score = score
                    best_prev = prev
            cur[st] = best_score + hmm_emission(st, s_val, tc, tp)
            back_i[st] = best_prev if best_prev is not None else "B"
        dp = cur
        back.append(back_i)

    # Backtrace
    last_state = max(dp.items(), key=lambda x: x[1])[0]
    path_states = [None] * len(idxs)
    for k in range(len(idxs) - 1, -1, -1):
        path_states[k] = last_state
        last_state = back[k][last_state]

    out: List[Optional[str]] = [None] * len(per_pos)
    for idx, st in zip(idxs, path_states):
        out[idx] = st
    return out


def hmm_segments(path: List[Optional[str]]) -> List[Tuple[str, int, int]]:
    """Return (state, start, end) segments using 1-based indices, skipping None."""
    segs = []
    cur_state = None
    cur_start = None
    for i, st in enumerate(path, start=1):
        if st is None:
            if cur_state is not None:
                segs.append((cur_state, cur_start, i - 1))
                cur_state = None
                cur_start = None
            continue
        if cur_state is None:
            cur_state = st
            cur_start = i
        elif st != cur_state:
            segs.append((cur_state, cur_start, i - 1))
            cur_state = st
            cur_start = i
    if cur_state is not None:
        segs.append((cur_state, cur_start, len(path)))
    return segs


def poisson_binomial_p_ge(ps: List[float], x: int) -> float:
    """Compute P(X >= x) for X = sum_i Bernoulli(p_i) via DP.

    ps: list of probabilities (each in [0,1])
    x: observed count threshold

    Returns a float in [0,1].
    """
    if x <= 0:
        return 1.0
    n = len(ps)
    if n == 0:
        return 0.0
    if x > n:
        return 0.0

    # dp[k] = P(sum == k) after processing i items
    dp = [0.0] * (n + 1)
    dp[0] = 1.0

    for p in ps:
        # update backwards to avoid overwriting
        for k in range(n, 0, -1):
            dp[k] = dp[k] * (1.0 - p) + dp[k - 1] * p
        dp[0] = dp[0] * (1.0 - p)

    return sum(dp[x:])


def modelc_uniform_anchor_stats(
    seq1_aln: str,
    seq2_aln: str,
    mj: Dict[Tuple[str, str], float],
    *,
    thr: float,
    gap_char: str = "-",
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> Tuple[int, float, float, float, float]:
    """Option 1 (Model C, uniform partner): analytic anchor-count tail probabilities in both directions.

    Returns (x_obs, ex_seq1_fixed, p_seq1_fixed, ex_seq2_fixed, p_seq2_fixed) where:
      - x_obs is the observed number of anchor positions (MJ <= thr) among eligible aligned positions
      - ex_seq1_fixed is E[X] when seq1 residues are fixed and partner residues are Uniform(AA20)
      - p_seq1_fixed = P(X >= x_obs | seq1 fixed, partner uniform)
      - ex_seq2_fixed is E[X] when seq2 residues are fixed and partner residues are Uniform(AA20)
      - p_seq2_fixed = P(X >= x_obs | seq2 fixed, partner uniform)

    Eligibility: positions where score_aligned returns a score (i.e., not gaps in either sequence and not
    excluded by unknown_policy='skip'). This matches how anchors are defined elsewhere in the script.
    """
    s1 = seq1_aln.strip().upper()
    s2 = seq2_aln.strip().upper()
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    # Observed anchors on eligible positions
    _total, per_pos = score_aligned(
        s1,
        s2,
        mj,
        gap_char=gap_char,
        gap_penalty=None,
        unknown_policy=unknown_policy,
        context_bonus=context_bonus,
    )
    anchors = [i for i, v in enumerate(
        per_pos, start=1) if v is not None and float(v) <= thr]
    x_obs = len(anchors)

    # Per-position probabilities for each conditioning (None for ineligible)
    p1 = per_position_complement_prob_uniform(
        s1, mj, thr=thr, gap_char=gap_char, unknown_policy=unknown_policy
    )
    p2 = per_position_complement_prob_uniform(
        s2, mj, thr=thr, gap_char=gap_char, unknown_policy=unknown_policy
    )

    # Base eligibility: positions scored in per_pos
    elig = [i for i, v in enumerate(per_pos, start=1) if v is not None]

    # Build ps lists on this eligibility, guarding against unknown_policy='skip'
    elig2 = [i for i in elig if p1[i - 1]
             is not None and p2[i - 1] is not None]
    ps1 = [float(p1[i - 1]) for i in elig2]  # type: ignore[arg-type]
    ps2 = [float(p2[i - 1]) for i in elig2]  # type: ignore[arg-type]

    # If eligibility was tightened (skip policy), recompute observed anchors accordingly
    if len(elig2) != len(elig):
        anchors_set = set(anchors)
        x_obs = sum(1 for i in elig2 if i in anchors_set)

    ex1 = float(sum(ps1))
    ex2 = float(sum(ps2))
    p_seq1_fixed = poisson_binomial_p_ge(ps1, x_obs)
    p_seq2_fixed = poisson_binomial_p_ge(ps2, x_obs)

    return x_obs, ex1, p_seq1_fixed, ex2, p_seq2_fixed


# ---------- Null distribution via shuffle ----------

def shuffle_preserve_gaps(seq: str, gap_char: str = "-") -> str:
    """Shuffle residues while keeping gap positions fixed."""
    residues = [c for c in seq if c != gap_char]
    random.shuffle(residues)

    out: List[str] = []
    r = 0
    for c in seq:
        if c == gap_char:
            out.append(gap_char)
        else:
            out.append(residues[r])
            r += 1
    return "".join(out)


def null_distribution(
    fixed_seq: str,
    shuffle_seq: str,
    mj: Dict[Tuple[str, str], float],
    n_iter: int = 1000,
    *,
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    context_bonus: bool = False,
) -> List[float]:
    """Shuffle shuffle_seq n_iter times (preserving gaps) and score each vs fixed_seq."""
    scores: List[float] = []
    for _ in range(n_iter):
        shuf = shuffle_preserve_gaps(shuffle_seq, gap_char=gap_char)
        total, _ = score_aligned(
            fixed_seq,
            shuf,
            mj,
            gap_char=gap_char,
            gap_penalty=gap_penalty,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        scores.append(total)
    return scores


def circular_shift(seq: str, k: int) -> str:
    """Circularly shift an aligned sequence by k positions (including gaps)."""
    if not seq:
        return seq
    k = k % len(seq)
    if k == 0:
        return seq
    return seq[-k:] + seq[:-k]


def circular_shift_null(
    fixed_seq: str,
    shift_seq: str,
    mj: Dict[Tuple[str, str], float],
    *,
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    context_bonus: bool = False,
) -> List[float]:
    """Null distribution by circularly shifting shift_seq relative to fixed_seq.

    Observed alignment is shift k=0. Null uses k in [1, L-1].

    If n_samples is provided and < (L-1), randomly sample shifts without replacement.
    """
    L = len(fixed_seq)
    if L != len(shift_seq):
        raise ValueError("Aligned sequences must be same length")
    if L < 2:
        return []

    shifts = list(range(1, L))
    if n_samples is not None and n_samples > 0 and n_samples < len(shifts):
        rng = random.Random(seed)
        shifts = rng.sample(shifts, n_samples)

    scores: List[float] = []
    for k in shifts:
        shifted = circular_shift(shift_seq, k)
        total, _ = score_aligned(
            fixed_seq,
            shifted,
            mj,
            gap_char=gap_char,
            gap_penalty=gap_penalty,
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        scores.append(total)
    return scores


# ---------- Scan-aware null (best window) ----------

def best_window_score(
    per_pos: List[Optional[float]],
    window: int,
    *,
    mode: str = "min",
    none_as: Optional[float] = 0.0,
) -> Tuple[Optional[float], Optional[int]]:
    """Compute best contiguous window score over per-position values.

    Windowing is performed over alignment indices.

    Handling of per_pos=None (typically gaps when gap_penalty is None, or unknown_policy=skip):
      - If none_as is not None (default 0.0), None values contribute none_as to the window sum.
      - If none_as is None, any window containing None is skipped.

    mode:
      - 'min': most negative window (strongest complement)
      - 'max': most positive window

    Returns (best_score, start_index_1based).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(per_pos):
        return None, None

    best: Optional[float] = None
    best_start: Optional[int] = None

    for start in range(0, len(per_pos) - window + 1):
        chunk = per_pos[start: start + window]
        if none_as is None:
            if any(v is None for v in chunk):
                continue
            s = float(sum(chunk))
        else:
            s = float(sum((none_as if v is None else float(v)) for v in chunk))

        if best is None:
            best, best_start = s, start + 1
        else:
            if mode == "min" and s < best:
                best, best_start = s, start + 1
            elif mode == "max" and s > best:
                best, best_start = s, start + 1

    return best, best_start


def quantile(values: List[float], q: float) -> float:
    """Deterministic quantile using linear interpolation between order statistics."""
    if not values:
        raise ValueError("quantile() requires a non-empty list")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    xs = sorted(values)
    n = len(xs)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def scan_null_best_window(
    fixed_seq: str,
    other_seq: str,
    mj: Dict[Tuple[str, str], float],
    *,
    window: int,
    null_method: str,
    n: int,
    shuffle_which: str,
    gap_char: str = "-",
    gap_penalty: Optional[float] = None,
    unknown_policy: str = "error",
    seed: Optional[int] = None,
    mode: str = "min",
    context_bonus: bool = False,
) -> List[float]:
    """Generate a scan-aware null distribution over best window scores.

    For each null draw, generate an aligned null pairing (shuffle or circular shift)
    then compute the best window score (min or max) over that null alignment.

    Returns a list of best-window scores (one per null draw).
    """
    if null_method not in {"shuffle", "shift"}:
        raise ValueError("null_method must be 'shuffle' or 'shift'")
    if shuffle_which not in {"seq1", "seq2"}:
        raise ValueError("shuffle_which must be 'seq1' or 'seq2'")

    s1 = fixed_seq
    s2 = other_seq
    if len(s1) != len(s2):
        raise ValueError("Aligned sequences must be same length")

    scores: List[float] = []

    if null_method == "shuffle":
        rng = random.Random(seed)

        if shuffle_which == "seq2":
            fixed = s1
            shuf_base = s2
        else:
            fixed = s2
            shuf_base = s1

        # Pre-split for gap-preserving shuffle
        gap_positions = [i for i, c in enumerate(shuf_base) if c == gap_char]
        residues = [c for c in shuf_base if c != gap_char]

        for _ in range(n):
            rng.shuffle(residues)
            out = []
            r = 0
            for i in range(len(shuf_base)):
                if i in gap_positions:
                    out.append(gap_char)
                else:
                    out.append(residues[r])
                    r += 1
            shuf = "".join(out)

            total, per_pos = score_aligned(
                fixed,
                shuf,
                mj,
                gap_char=gap_char,
                gap_penalty=gap_penalty,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
            best, _ = best_window_score(
                per_pos, window, mode=mode, none_as=0.0)
            if best is not None:
                scores.append(best)

    else:
        # Circular shifts: sample k in [1, L-1] without replacement if n < (L-1)
        L = len(s1)
        if L < 2:
            return []

        shifts = list(range(1, L))
        if n > 0 and n < len(shifts):
            rng = random.Random(seed)
            shifts = rng.sample(shifts, n)
        # If n >= L-1, we use all shifts.

        if shuffle_which == "seq2":
            fixed = s1
            shift_base = s2
        else:
            fixed = s2
            shift_base = s1

        for k in shifts:
            shifted = circular_shift(shift_base, k)
            total, per_pos = score_aligned(
                fixed,
                shifted,
                mj,
                gap_char=gap_char,
                gap_penalty=gap_penalty,
                unknown_policy=unknown_policy,
                context_bonus=context_bonus,
            )
            best, _ = best_window_score(
                per_pos, window, mode=mode, none_as=0.0)
            if best is not None:
                scores.append(best)

    return scores


def seed_null_best_scores(
    seq1: str,
    seq2: str,
    mj,
    *,
    window: int,
    n: int,
    unknown_policy: str = "error",
    seed: Optional[int] = None,
    context_bonus: bool = False,
) -> List[float]:
    """Null scores for the best ungapped window by shuffling seq2."""
    rng = random.Random(seed)
    scores: List[float] = []
    base = list(seq2)
    for _ in range(n):
        rng.shuffle(base)
        shuf = "".join(base)
        score, _, _, _ = best_ungapped_window_pair(
            seq1,
            shuf,
            mj,
            window=window,
            mode="min",
            unknown_policy=unknown_policy,
            context_bonus=context_bonus,
        )
        if score is not None:
            scores.append(float(score))
    return scores


# ---------- FASTA input ----------

def read_fasta_all(path):
    name = None
    seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_parts)
                name = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

        if name is not None:
            yield name, "".join(seq_parts)

    if name is None:
        raise ValueError(f"No FASTA header found in {path}")
    seq = "".join(seq_parts).strip()
    if not seq:
        raise ValueError(f"No sequence found in {path}")


def read_fasta_entry(path: str, name_filter: str) -> Tuple[str, str]:
    """Return the first FASTA record whose header contains name_filter."""
    matches = []
    for name, seq in read_fasta_all(path):
        if name_filter in name:
            matches.append((name, seq))
    if not matches:
        raise ValueError(f"No FASTA records match: {name_filter}")
    if len(matches) > 1:
        sample = "; ".join(n for n, _ in matches[:3])
        raise ValueError(
            f"Multiple FASTA records match: {name_filter} (n={len(matches)}). "
            f"Examples: {sample}"
        )
    return matches[0]




# ---------- CLI / main ----------

class TestMJScore(unittest.TestCase):
    def test_circular_shift(self):
        self.assertEqual(circular_shift("ABCD", 1), "DABC")
        self.assertEqual(circular_shift("ABCD", 4), "ABCD")

    def test_hypergeom_tail(self):
        # N=5 positions, A has K=2 anchors, B has n=2 anchors.
        # P(overlap >= 1) = 7/10 = 0.7
        p = hypergeom_p_at_least(k=1, N=5, K=2, n=2)
        self.assertAlmostEqual(p, 0.7, places=12)

    def test_shuffle_preserve_gaps(self):
        s = "A-BC--D"
        random.seed(0)
        sh = shuffle_preserve_gaps(s)
        # gaps preserved
        self.assertEqual([i for i, c in enumerate(s) if c == "-"],
                         [i for i, c in enumerate(sh) if c == "-"])
        # same multiset of residues
        self.assertEqual(sorted([c for c in s if c != "-"]),
                         sorted([c for c in sh if c != "-"]))

    def test_best_window_score_treats_none_as_zero(self):
        per = [1.0, None, -2.0, -3.0, 1.0]
        # With none_as=0, windows are always scorable; best 2-mer is [-2,-3]
        best, start = best_window_score(per, 2, mode="min", none_as=0.0)
        self.assertEqual(best, -5.0)
        self.assertEqual(start, 3)

        # If none_as=None, windows containing None are skipped; only windows without None are considered
        best2, start2 = best_window_score(per, 2, mode="min", none_as=None)
        self.assertEqual(best2, -5.0)
        self.assertEqual(start2, 3)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Score two aligned sequences using an MJ matrix (fixed register)."
    )
    p.add_argument(
        "--mj",
        default="mj_matrix.csv",
        help="Path to MJ matrix CSV (default: mj_matrix.csv)",
    )

    # Input methods: either direct strings, or FASTA files
    p.add_argument("--name1", default="seq1", help="Name for sequence 1")
    p.add_argument("--name2", default="seq2", help="Name for sequence 2")
    p.add_argument("--seq1", help="Aligned sequence 1 (may include '-')")
    p.add_argument("--seq2", help="Aligned sequence 2 (may include '-')")
    p.add_argument(
        "--fasta1", help="FASTA file for sequence 1 (first record used)")
    p.add_argument(
        "--fasta2", help="FASTA file for sequence 2 (first record used)")
    p.add_argument(
        "--fasta1-entry",
        help="Substring filter to select a specific entry from fasta1",
    )
    p.add_argument(
        "--fasta2-entry",
        help="Substring filter to select a specific entry from fasta2",
    )

    # Optional second pair for compare mode
    p.add_argument("--name1b", default="seq1b",
                   help="Name for sequence 1 (pair B)")
    p.add_argument("--name2b", default="seq2b",
                   help="Name for sequence 2 (pair B)")
    p.add_argument("--seq1b", help="Aligned sequence 1 for pair B")
    p.add_argument("--seq2b", help="Aligned sequence 2 for pair B")
    p.add_argument("--fasta1b", help="FASTA file for sequence 1 (pair B)")
    p.add_argument("--fasta2b", help="FASTA file for sequence 2 (pair B)")

    # Options
    p.add_argument(
        "--thr",
        type=float,
        default=-25.0,
        help="Anchor threshold (MJ <= thr). Default: -25",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top N contributing positions to print. Default: 10",
    )
    p.add_argument(
        "--null",
        type=int,
        default=1000,
        help="Null samples: shuffles (shuffle null) or sampled shifts (shift null). Default: 1000",
    )
    p.add_argument(
        "--null-method",
        choices=["shuffle", "shift"],
        default="shuffle",
        help="Null model: composition-preserving shuffle or circular shift. Default: shuffle",
    )
    p.add_argument(
        "--null-strict",
        action="store_true",
        help="Use a strict null (circular shift) and report a Bonferroni-corrected p-value",
    )
    p.add_argument(
        "--shuffle",
        choices=["seq1", "seq2"],
        default="seq2",
        help="Which sequence to shuffle for the null. Default: seq2",
    )
    p.add_argument(
        "--gap-penalty",
        type=float,
        default=None,
        help="Gap penalty. Omit to ignore gap positions (recommended)",
    )
    p.add_argument(
        "--unknown",
        choices=["error", "skip", "zero"],
        default="error",
        help="How to handle unknown residues. Default: error",
    )
    p.add_argument(
        "--clustal",
        action="store_true",
        help="Print Clustal-style similarity line and score",
    )
    p.add_argument(
        "--clustal-null",
        type=int,
        default=0,
        help="Null samples for Clustal-style similarity (default: 0)",
    )
    p.add_argument(
        "--clustal-null-method",
        choices=["shuffle", "shift"],
        default="shuffle",
        help="Null method for Clustal similarity: shuffle or shift (default: shuffle)",
    )
    p.add_argument(
        "--compare-clustal",
        action="store_true",
        help="Compare two aligned pairs using Clustal anchor Jaccard overlap",
    )
    p.add_argument(
        "--clustal-anchor-mode",
        choices=["identity", "strong", "strong+identity", "weak", "any"],
        default="strong+identity",
        help="Anchor definition for Clustal comparison (default: strong+identity)",
    )
    p.add_argument(
        "--clustal-search",
        action="store_true",
        help="Search for the best Clustal similarity windows",
    )
    p.add_argument(
        "--clustal-len",
        type=int,
        default=0,
        help="Window length for Clustal search",
    )
    p.add_argument(
        "--clustal-len-min",
        type=int,
        default=0,
        help="Minimum window length for Clustal range scan",
    )
    p.add_argument(
        "--clustal-len-max",
        type=int,
        default=0,
        help="Maximum window length for Clustal range scan",
    )
    p.add_argument(
        "--clustal-topk",
        type=int,
        default=5,
        help="Top K Clustal window hits to report (default: 5)",
    )
    p.add_argument(
        "--clustal-global-evals",
        type=int,
        default=0,
        help="Global sample budget for Clustal FASTA search (default: 0 = per-entry scan)",
    )
    p.add_argument(
        "--clustal-fasta-filter",
        help="Comma-separated substrings to restrict FASTA entries for Clustal search",
    )
    p.add_argument(
        "--clustal-exhaustive",
        action="store_true",
        help="Exhaustive per-entry Clustal scan across FASTA (no sampling)",
    )
    p.add_argument(
        "--clustal-collapse-species",
        action="store_true",
        help="Collapse FASTA hits by entry base (before _SPECIES) and count matches",
    )
    p.add_argument(
        "--clustal-kmer-len",
        type=int,
        default=0,
        help="K-mer length for Clustal prefilter (default: 0 = disabled)",
    )
    p.add_argument(
        "--clustal-kmer-min",
        type=int,
        default=1,
        help="Minimum matching k-mers within a window (default: 1)",
    )
    p.add_argument(
        "--clustal-require2",
        help="Require specific residues in seq2 window, e.g. '3=W,4=L' (1-based positions)",
    )
    p.add_argument(
        "--clustal-prefilter-strong",
        type=int,
        default=10,
        help="Minimum count of strong matches (* or :) to consider a window (default: 10)",
    )
    p.add_argument(
        "--clustal-prefilter-strong-frac",
        type=float,
        default=0.0,
        help="Minimum strong-match fraction for Clustal prefilter (0 = disabled)",
    )
    p.add_argument(
        "--clustal-prefilter-identity",
        type=int,
        default=0,
        help="Minimum count of identities (*) to consider a window (default: 0)",
    )
    p.add_argument(
        "--clustal-rank",
        choices=["score", "identity"],
        default="score",
        help="Rank Clustal windows by score or identity count (default: score)",
    )
    p.add_argument(
        "--filter-charge-runs",
        action="store_true",
        help="Filter out windows containing K/R or D/E runs of length >= 3",
    )
    p.add_argument(
        "--context-bonus",
        action="store_true",
        help="Include ±1 context bonuses in seed and stage2 scoring (default: off)",
    )
    p.add_argument(
        "--context-bonus-output",
        action="store_true",
        help="Include ±1 context bonuses in final scoring output (default: off)",
    )
    p.add_argument(
        "--combine-aligned",
        action="store_true",
        help="Print a combined ScanProsite-style regex from the aligned sequences",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling / shift sampling (optional)",
    )
    p.add_argument(
        "--scan-window",
        type=int,
        default=None,
        help="If set, compute scan-aware null for the best contiguous window of this length",
    )
    # Auto-align (ungapped): scan all window pairs of a fixed length and use the best-scoring pair as the effective alignment.
    p.add_argument(
        "--auto-align-len",
        type=int,
        default=0,
        help="If >0, auto-align by scanning all ungapped window pairs of this length and scoring the best (most favorable) window pair",
    )
    p.add_argument(
        "--auto-align-mode",
        choices=["min", "max"],
        default="min",
        help="Objective for auto-align window scanning: 'min' selects the most negative (most favorable) MJ sum; 'max' selects the most positive",
    )
    p.add_argument(
        "--auto-align-max-evals",
        type=int,
        default=0,
        help="Optional cap on evaluated window pairs (0 = no cap). If set and the search space exceeds this, random sampling is used",
    )

    # Auto-align (ungapped): scan all window pairs of a fixed length and use the best-scoring pair as the effective alignment.

    p.add_argument(
        "--scan-mode",
        choices=["min", "max"],
        default="min",
        help="Scan statistic: 'min' = most negative window (default), 'max' = most positive",
    )
    p.add_argument(
        "--quantiles",
        default="0.05,0.5,0.95",
        help="Comma-separated quantiles to report for null distributions (default: 0.05,0.5,0.95)",
    )
    p.add_argument(
        "--scanprosite",
        help="ScanProsite-style pattern to generate a complement motif (e.g., R-x(2)-K)",
    )
    p.add_argument(
        "--scanprosite-set-mode",
        choices=["all", "any"],
        default="all",
        help="How bracket sets are interpreted when generating complements (default: all)",
    )
    p.add_argument(
        "--scanprosite-context",
        action="store_true",
        help="Allow ±1 charge or proline/aromatic offsets when generating complements",
    )
    p.add_argument(
        "--scanprosite-topk",
        type=int,
        default=0,
        help="Limit complements to top K residues per position (default: 0 = all)",
    )
    p.add_argument(
        "--scanprosite-search",
        help="ScanProsite-style pattern to search in FASTA and list matching forms",
    )
    p.add_argument(
        "--scanprosite-search-complement",
        help="ScanProsite-style pattern; generate its complement and search it in FASTA",
    )
    p.add_argument(
        "--scanprosite-missing",
        action="store_true",
        help="When used with --scanprosite-search, also list any expected forms not present",
    )
    p.add_argument(
        "--scanprosite-score-avg",
        action="store_true",
        help="When searching complements, score matched forms by average MJ vs original pattern",
    )
    p.add_argument(
        "--scanprosite-score-top",
        type=int,
        default=0,
        help="Limit scored complement forms to top N (default: 0 = all)",
    )
    p.add_argument(
        "--fasta-filter",
        help="Only scan FASTA entries whose header contains this substring",
    )
    p.add_argument(
        "--two-stage",
        action="store_true",
        help="Enable two-stage seed+extension pipeline (seed scan + banded DP extension)",
    )
    p.add_argument(
        "--seed-len",
        type=int,
        default=12,
        help="Stage-1 seed window length (default: 12)",
    )
    p.add_argument(
        "--seed-rank",
        choices=["score", "anchors"],
        default="score",
        help="Stage-1 ranking: 'score' = most negative sum, 'anchors' = most anchors then best score",
    )
    p.add_argument(
        "--seed-score-max",
        type=float,
        default=-220.0,
        help="Stage-1 seed score threshold (keep scores <= this). Default: -220",
    )
    p.add_argument(
        "--seed-prefilter-len",
        type=int,
        default=0,
        help="Optional two-pass prefilter window length (< seed-len). 0 disables",
    )
    p.add_argument(
        "--seed-prefilter-score-max",
        type=float,
        default=-120.0,
        help="Prefilter score threshold (keep scores <= this). Default: -120",
    )
    p.add_argument(
        "--seed-prefilter-kmax",
        type=int,
        default=50000,
        help="Prefilter maximum candidate seeds to keep (default: 50000)",
    )
    p.add_argument(
        "--seed-prefilter-kmin",
        type=int,
        default=0,
        help="Prefilter minimum seeds to keep (default: 0)",
    )
    p.add_argument(
        "--seed-kmax",
        type=int,
        default=200,
        help="Stage-1 maximum seeds per target (default: 200)",
    )
    p.add_argument(
        "--seed-kmin",
        type=int,
        default=25,
        help="Stage-1 minimum seeds per target (default: 25)",
    )
    p.add_argument(
        "--seed-null",
        type=int,
        default=2000,
        help="Stage-1 null samples for best-seed score (default: 2000)",
    )
    p.add_argument(
        "--seed-topk",
        type=int,
        default=0,
        help="List top-K Stage-1 seeds and exit unless --seed-select is provided",
    )
    p.add_argument(
        "--seed-topk-global",
        type=int,
        default=0,
        help="List top-K Stage-1 seeds across all FASTA records and exit unless --seed-select is provided",
    )
    p.add_argument(
        "--seed-topk-global-offset",
        type=int,
        default=0,
        help="Offset into the global seed list (0-based). Use with --seed-topk-global for pagination",
    )
    p.add_argument(
        "--seed-select",
        type=int,
        default=0,
        help="1-based index to select from --seed-topk list for Stage-2 extension",
    )
    p.add_argument(
        "--stage2-minlen",
        type=int,
        default=20,
        help="Stage-2 minimum alignment length (default: 20)",
    )
    p.add_argument(
        "--stage2-maxlen",
        type=int,
        default=30,
        help="Stage-2 maximum alignment length (default: 30)",
    )
    p.add_argument(
        "--stage2-flank",
        type=int,
        default=10,
        help="Stage-2 flank size around seed (default: 10)",
    )
    p.add_argument(
        "--stage2-band",
        type=int,
        default=1,
        help="Stage-2 band (delta in [-band,band]) (default: 1)",
    )
    p.add_argument(
        "--stage2-switch",
        type=float,
        default=6.0,
        help="Stage-2 switching penalty for delta changes (default: 6)",
    )
    p.add_argument(
        "--stage2-gap-open",
        type=float,
        default=18.0,
        help="Stage-2 gap-open penalty (default: 18)",
    )
    p.add_argument(
        "--stage2-gap-ext",
        type=float,
        default=8.0,
        help="Stage-2 gap-extend penalty (default: 8)",
    )
    p.add_argument(
        "--stage2-max-gaps",
        type=int,
        default=2,
        help="Stage-2 maximum number of gaps (default: 2)",
    )
    p.add_argument(
        "--stage2-max-gap-len",
        type=int,
        default=1,
        help="Stage-2 maximum gap length (default: 1)",
    )
    p.add_argument(
        "--stage2-gap-proline-force-seq2",
        action="store_true",
        help="Stage-2: require at least one gap on every PP+ run in seq2 (gaps only in seq1)",
    )
    p.add_argument(
        "--stage2-gap-proline-force-seq2-post",
        action="store_true",
        help="Stage-2: after best alignment, force a gap on every PP+ run in seq2 and rescore",
    )
    p.add_argument(
        "--stage2-top",
        type=int,
        default=25,
        help="Stage-2: compute nulls for top N hits (default: 25)",
    )
    p.add_argument(
        "--stage2-reanchor",
        action="store_true",
        help="Stage-2: allow re-anchoring within the flank window (search best core position)",
    )
    p.add_argument(
        "--stage2-all-seeds",
        action="store_true",
        help="Stage-2: extend all Stage-1 seeds (default: only best seed per target)",
    )
    p.add_argument(
        "--stage2-null-min",
        type=int,
        default=500,
        help="Stage-2 null samples (initial). Default: 500",
    )
    p.add_argument(
        "--stage2-null-mid",
        type=int,
        default=2000,
        help="Stage-2 null samples (adaptive mid). Default: 2000",
    )
    p.add_argument(
        "--stage2-null-max",
        type=int,
        default=5000,
        help="Stage-2 null samples (adaptive max). Default: 5000",
    )
    p.add_argument(
        "--hmm-interface",
        action="store_true",
        help="Run HMM-style interface footprint on aligned positions",
    )
    p.add_argument(
        "--hmm-tc",
        type=float,
        default=-20.0,
        help="HMM core threshold Tc (default: -20)",
    )
    p.add_argument(
        "--hmm-tp",
        type=float,
        default=-15.0,
        help="HMM periphery threshold Tp (default: -15)",
    )
    p.add_argument(
        "--run-tests",
        action="store_true",
        help="Run internal unit tests and exit",
    )

    args = p.parse_args(argv)

    if args.run_tests:
        # Run tests and exit.
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMJScore)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.seed is not None:
        random.seed(args.seed)

    # Load MJ
    mj = load_mj_csv(args.mj)
    mj, mj_inverted, (mj_min, mj_max) = maybe_invert_mj(mj)
    if mj_inverted:
        global APPLY_MJ_OVERRIDES
        APPLY_MJ_OVERRIDES = False
        if args.thr == -25.0:
            args.thr = -0.5
            print(
                "[info] Auto-adjusted anchor threshold to -0.5 for nonnegative matrix.",
                file=sys.stderr,
            )
        if args.seed_score_max == -220.0:
            args.seed_score_max = -0.5 * float(args.seed_len)
            print(
                f"[info] Auto-adjusted seed score max to {args.seed_score_max:g} "
                f"based on seed length {args.seed_len}.",
                file=sys.stderr,
            )
        if args.seed_prefilter_len and args.seed_prefilter_score_max == -120.0:
            args.seed_prefilter_score_max = -0.5 * float(args.seed_prefilter_len)
            print(
                f"[info] Auto-adjusted prefilter score max to {args.seed_prefilter_score_max:g} "
                f"based on prefilter length {args.seed_prefilter_len}.",
                file=sys.stderr,
            )
        print(
            f"[info] MJ matrix has no negative values (min={mj_min:g}, max={mj_max:g}); "
            "auto-inverting values so lower scores are still more favorable. "
            "Adjust thresholds accordingly.",
            file=sys.stderr,
        )
        print(
            "[info] MJ overrides disabled for nonnegative matrix.",
            file=sys.stderr,
        )

    if args.scanprosite_search_complement:
        if not args.fasta2:
            print("ERROR: --scanprosite-search-complement requires --fasta2", file=sys.stderr)
            return 2
        try:
            comp = scanprosite_complement_motif(
                args.scanprosite_search_complement,
                mj,
                thr=args.thr,
                set_mode=args.scanprosite_set_mode,
                context_offset=args.scanprosite_context,
                top_k=args.scanprosite_topk,
            )
        except Exception as e:
            print(f"ERROR: scanprosite complement: {e}", file=sys.stderr)
            return 2
        try:
            counts = scanprosite_forms_in_fasta(
                comp,
                args.fasta2,
                unknown_policy=args.unknown,
                name_filter=args.fasta_filter,
            )
        except Exception as e:
            print(f"ERROR: scanprosite search: {e}", file=sys.stderr)
            return 2
        total = sum(counts.values())
        print(f"ScanProsite complement search: {args.scanprosite_search_complement}")
        print(f"Complement: {comp}")
        print(f"Total matches: {total}")
        if counts:
            print("Forms:")
            for form, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {n}  {form}")
        if args.scanprosite_score_avg and counts:
            scored = []
            for form in counts.keys():
                score = avg_mj_score_pattern_to_seq(
                    args.scanprosite_search_complement,
                    form,
                    mj,
                    unknown_policy=args.unknown,
                )
                if score is None:
                    continue
                scored.append((score, form))
            if scored:
                scored.sort(key=lambda x: x[0])
                if args.scanprosite_score_top and args.scanprosite_score_top > 0:
                    scored = scored[: args.scanprosite_score_top]
                print("Scored forms (avg MJ vs pattern):")
                for score, form in scored:
                    print(f"  {fmt_float(score, 2)}  {form}")
        return 0
    if args.scanprosite_search:
        if not args.fasta2:
            print("ERROR: --scanprosite-search requires --fasta2", file=sys.stderr)
            return 2
        try:
            counts = scanprosite_forms_in_fasta(
                args.scanprosite_search,
                args.fasta2,
                unknown_policy=args.unknown,
                name_filter=args.fasta_filter,
            )
        except Exception as e:
            print(f"ERROR: scanprosite search: {e}", file=sys.stderr)
            return 2
        total = sum(counts.values())
        print(f"ScanProsite search: {args.scanprosite_search}")
        print(f"Total matches: {total}")
        if counts:
            print("Forms:")
            for form, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {n}  {form}")
        if args.scanprosite_missing:
            expected = scanprosite_expected_forms(args.scanprosite_search)
            if expected is None:
                print("Missing forms: n/a (pattern contains x positions)")
            else:
                missing = sorted(expected.difference(counts.keys()))
                print(f"Missing forms: {len(missing)}")
                for form in missing:
                    print(f"  {form}")
        return 0
    if args.scanprosite:
        try:
            comp = scanprosite_complement_motif(
                args.scanprosite,
                mj,
                thr=args.thr,
                set_mode=args.scanprosite_set_mode,
                context_offset=args.scanprosite_context,
                top_k=args.scanprosite_topk,
            )
        except Exception as e:
            print(f"ERROR: scanprosite pattern: {e}", file=sys.stderr)
            return 2
        print("Complement:")
        print(comp)
        return 0

    if args.auto_align_len and not args.two_stage:
        # Consolidate auto-align into two-stage seed listing/selection.
        args.two_stage = True
        args.seed_len = args.auto_align_len
        if args.seed_topk == 0:
            args.seed_topk = 3
        if args.seed_select == 0:
            print(
                "NOTE: --auto-align-len is consolidated into --two-stage seed listing. "
                "Use --seed-select to extend a chosen seed."
            )
        args.auto_align_len = 0

    # Load sequences A
    if args.fasta1:
        if args.fasta1_entry:
            n1, s1 = read_fasta_entry(args.fasta1, args.fasta1_entry)
        else:
            n1, s1 = next(read_fasta_all(args.fasta1))
        args.name1 = n1
    else:
        s1 = args.seq1

    if args.fasta2:
        if args.fasta2_entry:
            n2, s2 = read_fasta_entry(args.fasta2, args.fasta2_entry)
        else:
            n2, s2 = next(read_fasta_all(args.fasta2))
        args.name2 = n2
    else:
        s2 = args.seq2

    if s1 is None or s2 is None:
        print(
            "Provide either --seq1/--seq2 or --fasta1/--fasta2 (or a mix)",
            file=sys.stderr,
        )
        return 2

    s1 = s1.strip().upper()
    s2 = s2.strip().upper()

    if args.clustal_search:
        require_positions2 = parse_clustal_require(args.clustal_require2)
        name_filters = None
        if args.clustal_fasta_filter:
            name_filters = [x.strip() for x in args.clustal_fasta_filter.split(",") if x.strip()]
        if args.fasta2_entry:
            if name_filters is None:
                name_filters = [args.fasta2_entry]
            else:
                name_filters.append(args.fasta2_entry)
        if args.clustal_len_min and args.clustal_len_max:
            if args.clustal_len_min <= 0 or args.clustal_len_max <= 0:
                print("ERROR: --clustal-len-min/max must be > 0", file=sys.stderr)
                return 2
            if args.clustal_len_min > args.clustal_len_max:
                print("ERROR: --clustal-len-min must be <= --clustal-len-max", file=sys.stderr)
                return 2
        elif not args.clustal_len or args.clustal_len <= 0:
            print("ERROR: --clustal-len must be set for --clustal-search", file=sys.stderr)
            return 2
        if require_positions2:
            for off, _allowed in require_positions2:
                if args.clustal_len and off >= args.clustal_len:
                    print("ERROR: --clustal-require2 position exceeds window length", file=sys.stderr)
                    return 2
        if args.clustal_len_min and args.clustal_len_max and args.fasta2 and not args.fasta2_entry:
            print(
                "ERROR: --clustal-len-min/max with FASTA search requires --fasta2-entry "
                "or a restricted FASTA filter",
                file=sys.stderr,
            )
            return 2
        if args.fasta2 and not args.fasta2_entry:
            if args.clustal_exhaustive:
                hits = []
                for name2, seq2 in read_fasta_all(args.fasta2):
                    if name_filters and not any(f in name2 for f in name_filters):
                        continue
                    if len(seq2) < args.clustal_len or len(s1) < args.clustal_len:
                        continue
                    strong_min = args.clustal_prefilter_strong
                    if args.clustal_prefilter_strong_frac and args.clustal_prefilter_strong_frac > 0:
                        strong_min = max(
                            strong_min,
                            math.ceil(args.clustal_prefilter_strong_frac * args.clustal_len),
                        )
                    score, i, j, ident_best = best_ungapped_window_pair_clustal(
                        s1,
                        seq2,
                        window=args.clustal_len,
                        max_evals=0,
                        rng_seed=args.seed or 1,
                        unknown_policy=args.unknown,
                        prefilter_min_strong=strong_min,
                        prefilter_min_identity=args.clustal_prefilter_identity,
                        kmer_len=args.clustal_kmer_len,
                        kmer_min=args.clustal_kmer_min,
                        require_positions2=require_positions2,
                        rank_by=args.clustal_rank,
                        filter_charge_runs=args.filter_charge_runs,
                    )
                    if score is None or i is None or j is None:
                        continue
                    win1 = s1[i : i + args.clustal_len]
                    win2 = seq2[j : j + args.clustal_len]
                    _sym, _s, _sn, _n = clustal_similarity(win1, win2)
                    hits.append((score, _sn, i, j, name2, win1, win2))

                hits.sort(key=lambda x: x[0], reverse=True)
                if args.clustal_collapse_species:
                    groups = {}
                    for score, sn, i, j, name2, win1, win2 in hits:
                        base, entry = clustal_entry_key(name2)
                        groups.setdefault(base, []).append((score, sn, i, j, name2, win1, win2))
                    collapsed = []
                    for base, items in groups.items():
                        best = max(items, key=lambda x: (x[0], x[1]))
                        collapsed.append((best[0], best[1], best[2], best[3], best[4], best[5], best[6], len(items)))
                    collapsed.sort(key=lambda x: x[0], reverse=True)
                    topk = collapsed[: max(1, args.clustal_topk)]
                    print(f"Clustal search top {len(topk)} across FASTA2 (collapsed)")
                    for idx, (score, sn, i, j, name2, win1, win2, count) in enumerate(topk, start=1):
                        print(
                            f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)} "
                            f"pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})  "
                            f"{name2}  {win1} ↔ {win2}  count={count}"
                        )
                else:
                    topk = hits[: max(1, args.clustal_topk)]
                    print(f"Clustal search top {len(topk)} across FASTA2 (exhaustive per-entry)")
                    for idx, (score, sn, i, j, name2, win1, win2) in enumerate(topk, start=1):
                        print(
                            f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)}  pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})  {name2}  {win1} ↔ {win2}"
                        )
                return 0
            if args.clustal_global_evals and args.clustal_global_evals > 0:
                if name_filters:
                    print("ERROR: --clustal-fasta-filter is not supported with global sampling", file=sys.stderr)
                    return 2
                strong_min = args.clustal_prefilter_strong
                if args.clustal_prefilter_strong_frac and args.clustal_prefilter_strong_frac > 0:
                    strong_min = max(
                        strong_min,
                        math.ceil(args.clustal_prefilter_strong_frac * args.clustal_len),
                    )
                hits = clustal_search_fasta_global(
                    s1,
                    args.fasta2,
                    window=args.clustal_len,
                    max_evals=args.clustal_global_evals,
                    rng_seed=args.seed or 1,
                    unknown_policy=args.unknown,
                    prefilter_min_strong=strong_min,
                    prefilter_min_identity=args.clustal_prefilter_identity,
                    kmer_len=args.clustal_kmer_len,
                    kmer_min=args.clustal_kmer_min,
                    require_positions2=require_positions2,
                    topk=args.clustal_topk,
                    filter_charge_runs=args.filter_charge_runs,
                )
                if args.clustal_collapse_species:
                    groups = {}
                    for score, sn, i, j, name2, win1, win2 in hits:
                        base, entry = clustal_entry_key(name2)
                        groups.setdefault(base, []).append((score, sn, i, j, name2, win1, win2))
                    collapsed = []
                    for base, items in groups.items():
                        best = max(items, key=lambda x: (x[0], x[1]))
                        collapsed.append((best[0], best[1], best[2], best[3], best[4], best[5], best[6], len(items)))
                    collapsed.sort(key=lambda x: x[0], reverse=True)
                    topk = collapsed[: max(1, args.clustal_topk)]
                    print(f"Clustal search top {len(topk)} across FASTA2 (collapsed)")
                    for idx, (score, sn, i, j, name2, win1, win2, count) in enumerate(topk, start=1):
                        print(
                            f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)} "
                            f"pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})  "
                            f"{name2}  {win1} ↔ {win2}  count={count}"
                        )
                else:
                    print(f"Clustal search top {len(hits)} across FASTA2 (global sampling)")
                    for idx, (score, sn, i, j, name2, win1, win2) in enumerate(hits, start=1):
                        print(
                            f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)}  pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})  {name2}  {win1} ↔ {win2}"
                        )
                return 0
            hits = []
            for name2, seq2 in read_fasta_all(args.fasta2):
                if name_filters and not any(f in name2 for f in name_filters):
                    continue
                if len(seq2) < args.clustal_len or len(s1) < args.clustal_len:
                    continue
                strong_min = args.clustal_prefilter_strong
                if args.clustal_prefilter_strong_frac and args.clustal_prefilter_strong_frac > 0:
                    strong_min = max(
                        strong_min,
                        math.ceil(args.clustal_prefilter_strong_frac * args.clustal_len),
                    )
                score, i, j, ident_best = best_ungapped_window_pair_clustal(
                    s1,
                    seq2,
                    window=args.clustal_len,
                    max_evals=args.auto_align_max_evals,
                    rng_seed=args.seed or 1,
                    unknown_policy=args.unknown,
                    prefilter_min_strong=strong_min,
                    prefilter_min_identity=args.clustal_prefilter_identity,
                    kmer_len=args.clustal_kmer_len,
                    kmer_min=args.clustal_kmer_min,
                    require_positions2=require_positions2,
                    rank_by=args.clustal_rank,
                    filter_charge_runs=args.filter_charge_runs,
                )
                if score is None or i is None or j is None:
                    continue
                win1 = s1[i : i + args.clustal_len]
                win2 = seq2[j : j + args.clustal_len]
                _sym, _s, _sn, _n = clustal_similarity(win1, win2)
                hits.append((score, _sn, i, j, name2, win1, win2))

            hits.sort(key=lambda x: x[0], reverse=True)
            if args.clustal_collapse_species:
                groups = {}
                for score, sn, i, j, name2, win1, win2 in hits:
                    base, entry = clustal_entry_key(name2)
                    groups.setdefault(base, []).append((score, sn, i, j, name2, win1, win2))
                collapsed = []
                for base, items in groups.items():
                    best = max(items, key=lambda x: (x[0], x[1]))
                    collapsed.append((best[0], best[1], best[2], best[3], best[4], best[5], best[6], len(items)))
                collapsed.sort(key=lambda x: x[0], reverse=True)
                topk = collapsed[: max(1, args.clustal_topk)]
                print(f"Clustal search top {len(topk)} across FASTA2 (collapsed)")
                for idx, (score, sn, i, j, name2, win1, win2, count) in enumerate(topk, start=1):
                    print(
                        f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)} "
                        f"pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})  "
                        f"{name2}  {win1} ↔ {win2}  count={count}"
                    )
            else:
                topk = hits[: max(1, args.clustal_topk)]
                print(f"Clustal search top {len(topk)} across FASTA2")
                for idx, (score, sn, i, j, name2, win1, win2) in enumerate(topk, start=1):
                    print(
                        f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)}  pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})  {name2}  {win1} ↔ {win2}"
                    )
            return 0

        if args.clustal_len_min and args.clustal_len_max:
            max_req = max((off + 1 for off, _ in (require_positions2 or [])), default=0)
            lengths = [L for L in range(args.clustal_len_min, args.clustal_len_max + 1) if L >= max_req]
            if not lengths:
                print("ERROR: no valid lengths for range scan", file=sys.stderr)
                return 2
            best_overall = None
            print(f"Clustal range scan ({lengths[0]}-{lengths[-1]})")
            for L in lengths:
                strong_min = args.clustal_prefilter_strong
                if args.clustal_prefilter_strong_frac and args.clustal_prefilter_strong_frac > 0:
                    strong_min = max(
                        strong_min,
                        math.ceil(args.clustal_prefilter_strong_frac * L),
                    )
                if args.clustal_rank == "identity" and args.clustal_topk > 1:
                    hits = best_ungapped_window_pair_clustal_topk(
                        s1,
                        s2,
                        window=L,
                        max_evals=args.auto_align_max_evals,
                        rng_seed=args.seed or 1,
                        unknown_policy=args.unknown,
                        prefilter_min_strong=strong_min,
                        prefilter_min_identity=args.clustal_prefilter_identity,
                        kmer_len=args.clustal_kmer_len,
                        kmer_min=args.clustal_kmer_min,
                        require_positions2=require_positions2,
                        rank_by=args.clustal_rank,
                        topk=args.clustal_topk,
                        filter_charge_runs=args.filter_charge_runs,
                    )
                    if not hits:
                        continue
                    seen = set()
                    kept = []
                    for idx, (score, ident, i, j) in enumerate(hits, start=1):
                        win1 = s1[i : i + L]
                        win2 = s2[j : j + L]
                        key_pair = (win1, win2)
                        if key_pair in seen:
                            continue
                        if any(
                            overlaps_fraction(i, i + L - 1, ki, ki + L - 1) >= 0.8
                            or overlaps_fraction(j, j + L - 1, kj, kj + L - 1) >= 0.8
                            for ki, kj in kept
                        ):
                            continue
                        seen.add(key_pair)
                        kept.append((i, j))
                        _sym, _s, sn, _n = clustal_similarity(win1, win2)
                        print(
                            f"  len={L} [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)}"
                            f" id={ident} pos=({i+1}-{i+L},{j+1}-{j+L})  {win1} ↔ {win2}"
                        )
                    score, ident_best, i, j = hits[0]
                    win1 = s1[i : i + L]
                    win2 = s2[j : j + L]
                    _sym, _s, sn, _n = clustal_similarity(win1, win2)
                    key = (ident_best, score)
                else:
                    score, i, j, ident_best = best_ungapped_window_pair_clustal(
                        s1,
                        s2,
                        window=L,
                        max_evals=args.auto_align_max_evals,
                        rng_seed=args.seed or 1,
                        unknown_policy=args.unknown,
                        prefilter_min_strong=strong_min,
                        prefilter_min_identity=args.clustal_prefilter_identity,
                        kmer_len=args.clustal_kmer_len,
                        kmer_min=args.clustal_kmer_min,
                        require_positions2=require_positions2,
                        rank_by=args.clustal_rank,
                        filter_charge_runs=args.filter_charge_runs,
                    )
                    if score is None or i is None or j is None:
                        continue
                    win1 = s1[i : i + L]
                    win2 = s2[j : j + L]
                    _sym, _s, sn, _n = clustal_similarity(win1, win2)
                    id_note = f" id={ident_best}" if ident_best is not None else ""
                    print(
                        f"  len={L} score={fmt_float(score, 2)} norm={fmt_float(sn, 3)}"
                        f"{id_note} pos=({i+1}-{i+L},{j+1}-{j+L})  {win1} ↔ {win2}"
                    )
                    key = (ident_best or 0, score) if args.clustal_rank == "identity" else (sn, score)
                if best_overall is None or key > best_overall[0]:
                    best_overall = (key, L, score, sn, ident_best, i, j, win1, win2)
            if best_overall is None:
                print("ERROR: no valid Clustal window found", file=sys.stderr)
                return 2
            _key, L, score, sn, ident_best, i, j, win1, win2 = best_overall
            label = "identity" if args.clustal_rank == "identity" else "normalized"
            print(f"Best overall ({label})")
            id_note = f" id={ident_best}" if ident_best is not None else ""
            print(
                f"  len={L} score={fmt_float(score, 2)} norm={fmt_float(sn, 3)}"
                f"{id_note} pos=({i+1}-{i+L},{j+1}-{j+L})  {win1} ↔ {win2}"
            )
            return 0

        strong_min = args.clustal_prefilter_strong
        if args.clustal_prefilter_strong_frac and args.clustal_prefilter_strong_frac > 0:
            strong_min = max(
                strong_min,
                math.ceil(args.clustal_prefilter_strong_frac * args.clustal_len),
            )
        if args.clustal_rank == "identity" and args.clustal_topk > 1:
            hits = best_ungapped_window_pair_clustal_topk(
                s1,
                s2,
                window=args.clustal_len,
                max_evals=args.auto_align_max_evals,
                rng_seed=args.seed or 1,
                unknown_policy=args.unknown,
                prefilter_min_strong=strong_min,
                prefilter_min_identity=args.clustal_prefilter_identity,
                kmer_len=args.clustal_kmer_len,
                kmer_min=args.clustal_kmer_min,
                require_positions2=require_positions2,
                rank_by=args.clustal_rank,
                topk=args.clustal_topk,
                filter_charge_runs=args.filter_charge_runs,
            )
            if not hits:
                print("ERROR: no valid Clustal window found", file=sys.stderr)
                return 2
            print(f"Clustal search top {len(hits)} (identity)")
            seen = set()
            kept = []
            for idx, (score, ident, i, j) in enumerate(hits, start=1):
                win1 = s1[i : i + args.clustal_len]
                win2 = s2[j : j + args.clustal_len]
                key = (win1, win2)
                if key in seen:
                    continue
                if any(
                    overlaps_fraction(i, i + args.clustal_len - 1, ki, ki + args.clustal_len - 1) >= 0.8
                    or overlaps_fraction(j, j + args.clustal_len - 1, kj, kj + args.clustal_len - 1) >= 0.8
                    for ki, kj in kept
                ):
                    continue
                seen.add(key)
                kept.append((i, j))
                _sym, _s, sn, _n = clustal_similarity(win1, win2)
                print(
                    f"  [{idx}] score={fmt_float(score, 2)} norm={fmt_float(sn, 3)} id={ident} "
                    f"pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})"
                )
                print(f"  {win1}")
                print(f"  {win2}")
            return 0
        score, i, j, ident_best = best_ungapped_window_pair_clustal(
            s1,
            s2,
            window=args.clustal_len,
            max_evals=args.auto_align_max_evals,
            rng_seed=args.seed or 1,
            unknown_policy=args.unknown,
            prefilter_min_strong=strong_min,
            prefilter_min_identity=args.clustal_prefilter_identity,
            kmer_len=args.clustal_kmer_len,
            kmer_min=args.clustal_kmer_min,
            require_positions2=require_positions2,
            rank_by=args.clustal_rank,
            filter_charge_runs=args.filter_charge_runs,
        )
        if score is None or i is None or j is None:
            print("ERROR: no valid Clustal window found", file=sys.stderr)
            return 2
        win1 = s1[i : i + args.clustal_len]
        win2 = s2[j : j + args.clustal_len]
        _sym, _s, sn, _n = clustal_similarity(win1, win2)
        print("Clustal search best window")
        print(
            f"  score={fmt_float(score, 2)} norm={fmt_float(sn, 3)} pos=({i+1}-{i+args.clustal_len},{j+1}-{j+args.clustal_len})"
        )
        print(f"  {win1}")
        print(f"  {win2}")
        return 0

    if args.two_stage:
        # Always run HMM after Stage-2 in two-stage mode.
        args.hmm_interface = True
        print("Two-stage seed + extension")
        targets: List[Tuple[str, str]] = []
        if args.fasta2:
            all_targets = [(n, s.upper()) for n, s in read_fasta_all(args.fasta2)]
            if args.fasta_filter:
                all_targets = [t for t in all_targets if args.fasta_filter in t[0]]
            targets = all_targets
        else:
            targets = [(args.name2, s2)]

        stage1_summary = []
        stage2_hits = []
        listed_only = False
        preselected_seed = None
        preselected_target = None

        if args.seed_topk and args.seed_topk_global:
            print(
                "ERROR: --seed-topk and --seed-topk-global are mutually exclusive.",
                file=sys.stderr,
            )
            return 2

        if args.seed_topk_global and args.seed_topk_global > 0:
            global_seeds = []
            for name2, seq2 in targets:
                seeds = seed_windows(
                    s1,
                    seq2,
                    mj,
                    window=args.seed_len,
                    score_max=args.seed_score_max,
                    kmax=args.seed_kmax,
                    kmin=args.seed_kmin,
                    rank_by_anchors=(args.seed_rank == "anchors"),
                    anchor_thr=args.thr,
                    prefilter_len=args.seed_prefilter_len,
                    prefilter_score_max=args.seed_prefilter_score_max,
                    prefilter_kmax=args.seed_prefilter_kmax,
                    prefilter_kmin=args.seed_prefilter_kmin,
                    unknown_policy=args.unknown,
                    context_bonus=args.context_bonus,
                )
                for score, i, j, anchors in seeds:
                    global_seeds.append((score, i, j, name2, seq2, anchors))

            if global_seeds:
                if args.seed_rank == "anchors":
                    global_seeds.sort(key=lambda x: (-x[5], x[0]))
                else:
                    global_seeds.sort(key=lambda x: x[0])
                offset = max(0, args.seed_topk_global_offset)
                if offset >= len(global_seeds):
                    print(
                        f"Stage-1 global seeds: offset {offset} exceeds available seeds ({len(global_seeds)})"
                    )
                    listed_only = True
                else:
                    end = min(offset + args.seed_topk_global, len(global_seeds))
                    topk = global_seeds[offset:end]
                    start_idx = offset + 1
                    end_idx = offset + len(topk)
                    print(
                        f"Stage-1 global seeds {start_idx}-{end_idx} across {len(targets)} targets"
                    )
                    for idx, (score, i, j, name2, seq2, anchors) in enumerate(topk, start=1):
                        s1_seed = s1[i : i + args.seed_len]
                        s2_seed = seq2[j : j + args.seed_len]
                        print(
                            f"  [{idx}] score={fmt_float(score, 2)}  anchors={anchors}  pos=({i+1}-{i+args.seed_len},{j+1}-{j+args.seed_len})  {name2}  {s1_seed} ↔ {s2_seed}"
                        )
                    if not args.seed_select:
                        listed_only = True
                    else:
                        if args.seed_select < 1 or args.seed_select > len(topk):
                            print(
                                f"ERROR: --seed-select must be in [1,{len(topk)}] for the listed seeds.",
                                file=sys.stderr,
                            )
                            return 2
                        score, i, j, name2, seq2, _anchors = topk[args.seed_select - 1]
                        preselected_seed = (score, i, j)
                        preselected_target = (name2, seq2)
            else:
                listed_only = True

        if listed_only and args.seed_topk_global and preselected_target is None:
            return 0

        if preselected_target is not None:
            targets = [preselected_target]

        for name2, seq2 in targets:
            if preselected_seed is not None:
                seeds = [preselected_seed]
            else:
                seeds = seed_windows(
                    s1,
                    seq2,
                    mj,
                    window=args.seed_len,
                    score_max=args.seed_score_max,
                    kmax=args.seed_kmax,
                    kmin=args.seed_kmin,
                    rank_by_anchors=(args.seed_rank == "anchors"),
                    anchor_thr=args.thr,
                    prefilter_len=args.seed_prefilter_len,
                    prefilter_score_max=args.seed_prefilter_score_max,
                    prefilter_kmax=args.seed_prefilter_kmax,
                    prefilter_kmin=args.seed_prefilter_kmin,
                    unknown_policy=args.unknown,
                    context_bonus=args.context_bonus,
                )
                if not seeds:
                    continue

                if args.seed_topk and args.seed_topk > 0:
                    topk = seeds[: min(args.seed_topk, len(seeds))]
                    print(f"Stage-1 top {len(topk)} seeds for {name2}")
                    for idx, seed in enumerate(topk, start=1):
                        score, i, j, anchors = _seed_unpack(seed)
                        s1_seed = s1[i : i + args.seed_len]
                        s2_seed = seq2[j : j + args.seed_len]
                        print(
                            f"  [{idx}] score={fmt_float(score, 2)}  anchors={anchors}  pos=({i+1}-{i+args.seed_len},{j+1}-{j+args.seed_len})  {s1_seed} ↔ {s2_seed}"
                        )
                    if not args.seed_select:
                        listed_only = True
                        continue
                    if len(targets) > 1:
                        print(
                            "ERROR: --seed-select requires a single target when --fasta2 has multiple records.",
                            file=sys.stderr,
                        )
                        return 2
                    if args.seed_select < 1 or args.seed_select > len(topk):
                        print(
                            f"ERROR: --seed-select must be in [1,{len(topk)}] for the listed seeds.",
                            file=sys.stderr,
                        )
                        return 2
                    seeds = [topk[args.seed_select - 1]]

            best_seed_score, best_seed_i, best_seed_j, best_seed_anchors = _seed_unpack(seeds[0])
            seed_pct = None
            if args.seed_null and args.seed_null > 0:
                null_scores = seed_null_best_scores(
                    s1,
                    seq2,
                    mj,
                    window=args.seed_len,
                    n=args.seed_null,
                    unknown_policy=args.unknown,
                    seed=args.seed,
                    context_bonus=args.context_bonus,
                )
                if null_scores:
                    seed_pct = sum(1 for x in null_scores if x <=
                                   best_seed_score) / len(null_scores)
                    seed_null_stats = {
                        "min": min(null_scores),
                        "median": quantile(null_scores, 0.5),
                        "max": max(null_scores),
                        "count_le": sum(1 for x in null_scores if x <= best_seed_score),
                        "n": len(null_scores),
                    }
                else:
                    seed_null_stats = None
            else:
                seed_null_stats = None

            stage1_summary.append(
                {
                    "name2": name2,
                    "seed_count": len(seeds),
                    "best_score": best_seed_score,
                    "best_i": best_seed_i,
                    "best_j": best_seed_j,
                    "best_s1": s1[best_seed_i : best_seed_i + args.seed_len],
                    "best_s2": seq2[best_seed_j : best_seed_j + args.seed_len],
                    "best_anchors": best_seed_anchors,
                    "seed_pct": seed_pct,
                    "seed_null_stats": seed_null_stats,
                }
            )

            seeds_to_extend = seeds if args.stage2_all_seeds else seeds[:1]
            for seed in seeds_to_extend:
                score, i, j, _anchors = _seed_unpack(seed)
                st2_score, aln1, aln2, start, anchor = stage2_best_from_seed(
                    s1,
                    seq2,
                    mj,
                    seed_i=i,
                    seed_j=j,
                    seed_len=args.seed_len,
                    flank=args.stage2_flank,
                    min_len=args.stage2_minlen,
                    max_len=args.stage2_maxlen,
                    band=args.stage2_band,
                    switch_pen=args.stage2_switch,
                    gap_open=args.stage2_gap_open,
                    gap_ext=args.stage2_gap_ext,
                    gap_proline_force_runs_seq2=args.stage2_gap_proline_force_seq2,
                    gap_proline_force_post=args.stage2_gap_proline_force_seq2_post,
                    max_gaps=args.stage2_max_gaps,
                    max_gap_len=args.stage2_max_gap_len,
                    unknown_policy=args.unknown,
                    reanchor=args.stage2_reanchor,
                    context_bonus=args.context_bonus,
                )
                if (
                    st2_score is None
                    or aln1 is None
                    or aln2 is None
                    or start is None
                    or anchor is None
                ):
                    continue
                stage2_hits.append(
                    {
                        "name2": name2,
                        "seq2": seq2,
                        "seed_score": score,
                        "seed_i": i,
                        "seed_j": j,
                        "stage2_score": st2_score,
                        "aln1": aln1,
                        "aln2": aln2,
                        "start_i": start[0],
                        "start_j": start[1],
                        "anchor_i": anchor[0],
                        "anchor_j": anchor[1],
                    }
                )

        if listed_only:
            return 0
        if not stage1_summary:
            print("Stage-1: no seeds found.")
            return 2

        print("Stage-1 seeds")
        for s in stage1_summary:
            pct_str = fmt_pct(
                s["seed_pct"], 4) if s["seed_pct"] is not None else "n/a"
            anchors_str = f"  anchors={s['best_anchors']}" if s.get("best_anchors") is not None else ""
            print(
                f"  {s['name2']}  seeds={s['seed_count']}  best={fmt_float(s['best_score'], 2)}"
                f"{anchors_str}  @ ({s['best_i']+1},{s['best_j']+1})  null_pct={pct_str}"
            )
            print(f"    seed1: {s['best_s1']}")
            print(f"    seed2: {s['best_s2']}")
            if s["seed_null_stats"] is not None:
                st = s["seed_null_stats"]
                print(
                    "    null_best:",
                    f"min={fmt_float(st['min'], 2)}",
                    f"median={fmt_float(st['median'], 2)}",
                    f"max={fmt_float(st['max'], 2)}",
                    f"count_le={st['count_le']}/{st['n']}",
                )

        if not stage2_hits:
            print("Stage-2: no DP hits found.")
            return 2

        stage2_hits.sort(key=lambda x: x["stage2_score"])
        top_hits = stage2_hits[: max(1, args.stage2_top)]

        print("Stage-2 hits")
        for idx, h in enumerate(top_hits, start=1):
            score = h["stage2_score"]
            # Adaptive nulls
            null_n = args.stage2_null_min
            null_scores: List[float] = []
            if null_n > 0:
                null_scores = stage2_null_scores(
                    s1,
                    h["seq2"],
                    mj,
                    seed_i=h["seed_i"],
                    seed_j=h["seed_j"],
                    seed_len=args.seed_len,
                    flank=args.stage2_flank,
                    min_len=args.stage2_minlen,
                    max_len=args.stage2_maxlen,
                    band=args.stage2_band,
                    switch_pen=args.stage2_switch,
                    gap_open=args.stage2_gap_open,
                    gap_ext=args.stage2_gap_ext,
                    gap_proline_force_runs_seq2=args.stage2_gap_proline_force_seq2,
                    gap_proline_force_post=args.stage2_gap_proline_force_seq2_post,
                    max_gaps=args.stage2_max_gaps,
                    max_gap_len=args.stage2_max_gap_len,
                    unknown_policy=args.unknown,
                    n=null_n,
                    seed=args.seed,
                    reanchor=args.stage2_reanchor,
                    context_bonus=args.context_bonus,
                )
            pct = None
            if null_scores:
                pct = sum(1 for x in null_scores if x <=
                          score) / len(null_scores)
            if pct is not None and pct <= 0.02 and args.stage2_null_mid > null_n:
                extra = args.stage2_null_mid - null_n
                null_scores += stage2_null_scores(
                    s1,
                    h["seq2"],
                    mj,
                    seed_i=h["seed_i"],
                    seed_j=h["seed_j"],
                    seed_len=args.seed_len,
                    flank=args.stage2_flank,
                    min_len=args.stage2_minlen,
                    max_len=args.stage2_maxlen,
                    band=args.stage2_band,
                    switch_pen=args.stage2_switch,
                    gap_open=args.stage2_gap_open,
                    gap_ext=args.stage2_gap_ext,
                    gap_proline_force_runs_seq2=args.stage2_gap_proline_force_seq2,
                    gap_proline_force_post=args.stage2_gap_proline_force_seq2_post,
                    max_gaps=args.stage2_max_gaps,
                    max_gap_len=args.stage2_max_gap_len,
                    unknown_policy=args.unknown,
                    n=extra,
                    seed=args.seed,
                    reanchor=args.stage2_reanchor,
                    context_bonus=args.context_bonus,
                )
                pct = sum(1 for x in null_scores if x <=
                          score) / len(null_scores)
                null_n = args.stage2_null_mid
            if pct is not None and pct <= 0.005 and args.stage2_null_max > null_n:
                extra = args.stage2_null_max - null_n
                null_scores += stage2_null_scores(
                    s1,
                    h["seq2"],
                    mj,
                    seed_i=h["seed_i"],
                    seed_j=h["seed_j"],
                    seed_len=args.seed_len,
                    flank=args.stage2_flank,
                    min_len=args.stage2_minlen,
                    max_len=args.stage2_maxlen,
                    band=args.stage2_band,
                    switch_pen=args.stage2_switch,
                    gap_open=args.stage2_gap_open,
                    gap_ext=args.stage2_gap_ext,
                    gap_proline_force_runs_seq2=args.stage2_gap_proline_force_seq2,
                    gap_proline_force_post=args.stage2_gap_proline_force_seq2_post,
                    max_gaps=args.stage2_max_gaps,
                    max_gap_len=args.stage2_max_gap_len,
                    unknown_policy=args.unknown,
                    n=extra,
                    seed=args.seed,
                    reanchor=args.stage2_reanchor,
                    context_bonus=args.context_bonus,
                )
                pct = sum(1 for x in null_scores if x <=
                          score) / len(null_scores)
                null_n = args.stage2_null_max

            null_mean = None
            if null_scores:
                null_mean = sum(null_scores) / len(null_scores)
            aln_len = len(h["aln1"])
            print(
                f"[{idx}] {h['name2']}  stage2={fmt_float(score, 2)}  aln_len={aln_len}  null_pct={fmt_pct(pct, 4) if pct is not None else 'n/a'}  null_n={null_n}"
            )
            print(
                f"  seed @ ({h['seed_i']+1},{h['seed_j']+1}) score={fmt_float(h['seed_score'], 2)}"
            )
            if (
                h.get("anchor_i") is not None
                and (h["anchor_i"], h["anchor_j"]) != (h["seed_i"], h["seed_j"])
            ):
                print(
                    f"  reanchor @ ({h['anchor_i']+1},{h['anchor_j']+1})"
                )
            print(
                f"  start @ ({h['start_i']+1},{h['start_j']+1})  null_mean={fmt_float(null_mean, 2) if null_mean is not None else 'n/a'}"
            )
            print(f"  {h['aln1']}")
            print(f"  {h['aln2']}")
            _st2_total, _st2_per = score_aligned(
                h["aln1"],
                h["aln2"],
                mj,
                gap_penalty=None,
                unknown_policy=args.unknown,
                context_bonus=args.context_bonus_output,
            )
            contrib = []
            for i, v in enumerate(_st2_per, start=1):
                if v is None:
                    continue
                a = h["aln1"][i - 1]
                b = h["aln2"][i - 1]
                contrib.append(f"{i}:{a}-{b}:{fmt_float(float(v), 2)}")
            print("  MJ per-pos:", " ".join(contrib))
            if args.hmm_interface:
                path = hmm_interface_path(
                    _st2_per, tc=args.hmm_tc, tp=args.hmm_tp)
                segs = hmm_segments(path)
                path_str = "".join(st if st is not None else "-" for st in path)
                print("  HMM path:", path_str)
                print(
                    "  HMM segments:",
                    ", ".join(f"{s}:{a}-{b}" for s, a, b in segs),
                )

        return 0

    # Score A

    auto_align_pairs = None
    # Auto-align: scan ungapped window pairs of a fixed length and use the best as the effective alignment.
    if args.auto_align_len and args.auto_align_len > 0:

        # If fasta2 was provided, treat it as a multi-record database and find the single best match globally.
        if args.fasta2:
            best = None
            best_name2 = None
            best_full2 = None
            best_per = None
            scanned = 0
            skipped = 0
            total_pairs_global = 0

            for name2, full2 in read_fasta_all(args.fasta2):
                scanned += 1

                # Skip sequences too short for the requested window
                if len(full2) < args.auto_align_len or len(s1) < args.auto_align_len:
                    skipped += 1
                    continue
                total_pairs_global += (len(s1) - args.auto_align_len + 1) * (
                    len(full2) - args.auto_align_len + 1
                )

                score, i, j, per = best_ungapped_window_pair(
                    s1,
                    full2,
                    mj,
                    window=args.auto_align_len,
                    mode=args.auto_align_mode,
                    max_evals=args.auto_align_max_evals,
                    unknown_policy=args.unknown,
                    context_bonus=args.context_bonus,
                )

                if score is None:
                    skipped += 1
                    continue

                if best is None or score < best["score"]:
                    best = {"score": score, "i": i, "j": j}
                    best_name2 = name2
                    best_full2 = full2
                    best_per = per

            if best is None:
                print(
                    f"ERROR: --auto-align-len={args.auto_align_len} not feasible for sequence lengths {len(s1)} and FASTA2 records",
                    file=sys.stderr,
                )
                return 2

            best_score = best["score"]
            best_i = best["i"]
            best_j = best["j"]
            _best_per = best_per

            orig_s1 = s1
            orig_s2 = best_full2  # <-- crucial: the winning protein from the FASTA

            # Update displayed partner name to the winning record
            args.name2 = best_name2
            auto_align_pairs = total_pairs_global if total_pairs_global > 0 else None

        else:
            # Original behavior: seq2 provided directly (single sequence)
            best_score, best_i, best_j, _best_per = best_ungapped_window_pair(
                s1,
                s2,
                mj,
                window=args.auto_align_len,
                mode=args.auto_align_mode,
                max_evals=args.auto_align_max_evals,
                unknown_policy=args.unknown,
                context_bonus=args.context_bonus,
            )
            if best_score is None:
                print(
                    f"ERROR: --auto-align-len={args.auto_align_len} is not feasible for sequence lengths {len(s1)} and {len(s2)}",
                    file=sys.stderr,
                )
                return 2

            orig_s1 = s1
            orig_s2 = s2

        # Slice the best windows
        s1 = orig_s1[best_i: best_i + args.auto_align_len]
        s2 = orig_s2[best_j: best_j + args.auto_align_len]

        total_pairs = (len(orig_s1) - args.auto_align_len + 1) * \
            (len(orig_s2) - args.auto_align_len + 1)
        if auto_align_pairs is None:
            auto_align_pairs = total_pairs
        sampled = args.auto_align_max_evals and args.auto_align_max_evals > 0 and total_pairs > args.auto_align_max_evals

        print("Auto-align (ungapped best window pair)")
        print(
            f"  window length: {args.auto_align_len}  objective: {args.auto_align_mode}"
            + (f"  (sampled {args.auto_align_max_evals} pairs)" if sampled else "")
        )
        print(f"  best window score: {fmt_float(best_score, 2)}")
        print(f"  seq1 window: pos {best_i+1}-{best_i+args.auto_align_len}  {s1}")
        print(f"  seq2 window: pos {best_j+1}-{best_j+args.auto_align_len}  {s2}")

        # Optional (but very useful) debug line:
        # if args.fasta2:
        # print(f"[DEBUG] scanned {scanned} FASTA2 records (skipped {skipped})", file=sys.stderr)

    total, per_pos = score_aligned(
        s1,
        s2,
        mj,
        gap_penalty=args.gap_penalty,
        unknown_policy=args.unknown,
        context_bonus=args.context_bonus_output,
    )

    print(f"{args.name1} ↔ {args.name2}")
    print(f"Total MJ score: {fmt_float(total, 2)}")

    print("Top contributors:")
    for v, i, a, b in top_contributors(s1, s2, per_pos, n=args.top):
        print(f"pos {i:3d}: {a}-{b}  MJ={fmt_float(v, 2)}")

    anchors = anchors_by_threshold(per_pos, thr=args.thr)
    a_total, a_used = anchor_score(per_pos, anchors)
    print(f"Anchors (MJ <= {args.thr:g}): {anchors}")
    print("Number of anchors:", len(anchors))
    print(f"Anchor-only MJ score: {fmt_float(a_total, 2)}")
    print("Anchor details:",
          "[" + ", ".join(f"({i}, {fmt_float(v, 2)})" for i, v in a_used) + "]")
    # ScanProsite-style anchor motifs (anchors only; non-anchors as x, compressed as x(n))
    motif1 = scanprosite_motif_from_anchors(s1, anchors)
    motif2 = scanprosite_motif_from_anchors(s2, anchors)
    print("Anchor motifs (ScanProsite regex style):")
    print(f"  {args.name1}: {motif1}")
    print(f"  {args.name2}: {motif2}")
    if args.combine_aligned:
        combined = combined_aligned_regex(s1, s2)
        strong = combined_aligned_strong_regex(s1, s2)
        print("Combined regex:")
        print(f"  {combined}")
        print("Combined regex (any sim):")
        print(f"  {strong}")

    if args.clustal or (args.clustal_null and args.clustal_null > 0):
        symbols, c_score, c_norm, c_n = clustal_similarity(s1, s2)
        print("Clustal similarity:")
        print(f"  symbols: {symbols}")
        print(f"  score: {fmt_float(c_score, 3)}")
        print(f"  score_norm: {fmt_float(c_norm, 3)}  (n={c_n})")
        if args.clustal_null and args.clustal_null > 0:
            null_scores = []
            if args.clustal_null_method == "shuffle":
                rng = random.Random(args.seed)
                if args.shuffle == "seq2":
                    fixed = s1
                    shuf_base = s2
                else:
                    fixed = s2
                    shuf_base = s1
                for _ in range(args.clustal_null):
                    shuf = shuffle_preserve_gaps(shuf_base)
                    _sym, _s, _n, _nn = clustal_similarity(fixed, shuf)
                    null_scores.append(_s)
            else:
                shifts = list(range(1, len(s1)))
                if args.clustal_null > 0 and args.clustal_null < len(shifts):
                    rng = random.Random(args.seed)
                    shifts = rng.sample(shifts, args.clustal_null)
                for k in shifts:
                    if args.shuffle == "seq2":
                        shifted = circular_shift(s2, k)
                        fixed = s1
                    else:
                        shifted = circular_shift(s1, k)
                        fixed = s2
                    _sym, _s, _n, _nn = clustal_similarity(fixed, shifted)
                    null_scores.append(_s)
            if null_scores:
                count_ge = sum(1 for x in null_scores if x >= c_score)
                p_clustal = (count_ge + 1) / (len(null_scores) + 1)
                null_mean = sum(null_scores) / len(null_scores)
                print(
                    f"  clustal null: mean={fmt_float(null_mean, 3)}  p={fmt_prob(p_clustal)}"
                )
                if auto_align_pairs is not None and auto_align_pairs > 1:
                    p_bonf = min(1.0, p_clustal * float(auto_align_pairs))
                    print(
                        f"  clustal Bonferroni p (M={auto_align_pairs}): {fmt_prob(p_bonf)}"
                    )

    # Analytic baseline: Model C (uniform partner), report both conditionings
    try:
        x_obs_c, ex1, p_c1, ex2, p_c2 = modelc_uniform_anchor_stats(
            s1,
            s2,
            mj,
            thr=args.thr,
            unknown_policy=args.unknown,
            context_bonus=args.context_bonus_output,
        )
        print("Model C (uniform partner) anchor-count p-values")
        print(f"Observed anchors (eligible positions): {x_obs_c}")

        print("Seq1 fixed:")
        print(f"  Expected anchors E[X]: {fmt_float(ex1, 3)}")
        print(
            f"  P(X >= x | seq1 fixed, partner uniform, thr={args.thr:g}): {fmt_prob(p_c1)}"
        )

        print("Seq2 fixed:")
        print(f"  Expected anchors E[X]: {fmt_float(ex2, 3)}")
        print(
            f"  P(X >= x | seq2 fixed, partner uniform, thr={args.thr:g}): {fmt_prob(p_c2)}"
        )
    except Exception as e:
        print(
            f"Model C (uniform partner) p-values: ERROR: {e}", file=sys.stderr)

    if args.hmm_interface:
        path = hmm_interface_path(per_pos, tc=args.hmm_tc, tp=args.hmm_tp)
        segs = hmm_segments(path)
        path_str = "".join(st if st is not None else "-" for st in path)
        print("HMM interface footprint")
        print(f"Path: {path_str}")
        print("Segments:", ", ".join(f"{s}:{a}-{b}" for s, a, b in segs))

    # Null
    if args.null and args.null > 0:
        use_strict = bool(args.null_strict)
        if (not use_strict) and args.null_method == "shuffle":
            if args.shuffle == "seq2":
                null_scores = null_distribution(
                    s1,
                    s2,
                    mj,
                    n_iter=args.null,
                    gap_penalty=args.gap_penalty,
                    unknown_policy=args.unknown,
                    context_bonus=args.context_bonus_output,
                )
            else:
                null_scores = null_distribution(
                    s2,
                    s1,
                    mj,
                    n_iter=args.null,
                    gap_penalty=args.gap_penalty,
                    unknown_policy=args.unknown,
                    context_bonus=args.context_bonus_output,
                )
            null_label = "shuffle"
        else:
            # Circular shift preserves internal sequence structure exactly; randomizes registration.
            if args.shuffle == "seq2":
                null_scores = circular_shift_null(
                    s1,
                    s2,
                    mj,
                    gap_penalty=args.gap_penalty,
                    unknown_policy=args.unknown,
                    n_samples=args.null,
                    seed=args.seed,
                    context_bonus=args.context_bonus_output,
                )
            else:
                null_scores = circular_shift_null(
                    s2,
                    s1,
                    mj,
                    gap_penalty=args.gap_penalty,
                    unknown_policy=args.unknown,
                    n_samples=args.null,
                    seed=args.seed,
                    context_bonus=args.context_bonus_output,
                )
            null_label = "strict-circular-shift" if use_strict else "circular-shift"

        if not null_scores:
            print("Null:", null_label)
            print("Null distribution is empty (sequence too short).")
        else:
            n_null = len(null_scores)
            null_mean = sum(null_scores) / n_null
            count_le = sum(1 for x in null_scores if x <= total)
            cdf = count_le / n_null
            p_raw = (count_le + 1) / (n_null + 1)

            # Quantiles
            try:
                qs = [float(x.strip())
                      for x in args.quantiles.split(",") if x.strip()]
            except ValueError:
                print(
                    "ERROR: --quantiles must be comma-separated floats, e.g. 0.05,0.5,0.95", file=sys.stderr)
                return 2

            q_report = {q: quantile(null_scores, q) for q in qs}

            print("Null:", null_label)
            print(f"Null mean: {fmt_float(null_mean, 2)}")
            print(f"Null exceedances (<= observed): {count_le}/{n_null}")
            if count_le == 0:
                print(
                    "Empirical CDF at observed: 0 (no nulls <= observed)")
                print(
                    f"Monte Carlo p (lower tail): <= {fmt_prob(1.0 / (n_null + 1))}"
                )
            else:
                print(f"Empirical CDF at observed: {fmt_pct(cdf, 4)}")
                print(f"Monte Carlo p (lower tail): {fmt_prob(p_raw)}")
            print("Null quantiles:", ", ".join(
                f"q{q:g}={fmt_float(q_report[q], 2)}" for q in qs))
            if use_strict:
                if auto_align_pairs is not None and auto_align_pairs > 1:
                    p_bonf = min(1.0, p_raw * float(auto_align_pairs))
                    if count_le == 0:
                        print(
                            f"Bonferroni p (M={auto_align_pairs}): <= {fmt_prob(p_bonf)}"
                        )
                    else:
                        print(
                            f"Bonferroni p (M={auto_align_pairs}): {fmt_prob(p_bonf)}"
                        )
                else:
                    print("Bonferroni p: n/a (M=1)")

            # Optional scan-aware reporting (best window) using the same null method
            if args.scan_window is not None:
                if args.scan_window <= 0:
                    print("ERROR: --scan-window must be positive", file=sys.stderr)
                    return 2

                obs_best, obs_start = best_window_score(
                    per_pos, args.scan_window, mode=args.scan_mode, none_as=0.0)
                if obs_best is None or obs_start is None:
                    print(
                        f"Scan-aware: no scorable windows of length {args.scan_window}.")
                else:
                    scan_null = scan_null_best_window(
                        s1,
                        s2,
                        mj,
                        window=args.scan_window,
                        null_method="shift" if use_strict else args.null_method,
                        n=args.null,
                        shuffle_which=args.shuffle,
                        gap_penalty=args.gap_penalty,
                        unknown_policy=args.unknown,
                        seed=args.seed,
                        mode=args.scan_mode,
                        context_bonus=args.context_bonus_output,
                    )

                    if not scan_null:
                        print(
                            "Scan-aware null: empty (sequence too short or windows unscorable)")
                    else:
                        scan_n = len(scan_null)
                        scan_count_le = sum(
                            1 for x in scan_null if x <= obs_best)
                        scan_cdf = scan_count_le / scan_n
                        scan_mean = sum(scan_null) / scan_n
                        scan_q = {q: quantile(scan_null, q) for q in qs}

                        end = obs_start + args.scan_window - 1
                        w1 = s1[obs_start - 1: end]
                        w2 = s2[obs_start - 1: end]
                        print("Scan-aware null (best window)")
                        print(
                            f"Window length: {args.scan_window}  mode: {args.scan_mode}")
                        print(
                            f"Observed best window score: {fmt_float(obs_best, 2)}  at pos {obs_start}-{end}")
                        print(f"Observed best window: {w1} ↔ {w2}")
                        print(f"Scan-null mean: {fmt_float(scan_mean, 2)}")
                        print(
                            f"Scan-null exceedances (<= observed): {scan_count_le}/{scan_n}")
                        if scan_count_le == 0:
                            print(
                                "Empirical CDF at observed: 0 (no nulls <= observed)"
                            )
                        else:
                            print(
                                f"Empirical CDF at observed: {fmt_pct(scan_cdf, 4)}")
                        print("Scan-null quantiles:",
                              ", ".join(f"q{q:g}={fmt_float(scan_q[q], 2)}" for q in qs))

    # ---- Compare mode: pair B ----
    # Load sequences B (optional)
    if args.fasta1b:
        n1b, s1b = next(read_fasta_all(args.fasta1b))
        args.name1b = n1b
    else:
        s1b = args.seq1b

    if args.fasta2b:
        n2b, s2b = next(read_fasta_all(args.fasta2b))
        args.name2b = n2b
    else:
        s2b = args.seq2b

    if s1b is not None and s2b is not None:
        s1b = s1b.strip().upper()
        s2b = s2b.strip().upper()

        if len(s1b) != len(s2b):
            print(
                f"Pair B aligned lengths differ: {len(s1b)} vs {len(s2b)}",
                file=sys.stderr,)
            return 2

        total_b, per_pos_b = score_aligned(
            s1b,
            s2b,
            mj,
            gap_penalty=args.gap_penalty,
            unknown_policy=args.unknown,
            context_bonus=args.context_bonus_output,
        )

        print(f"{args.name1b} ↔ {args.name2b}")
        print(f"Total MJ score: {fmt_float(total_b, 2)}")

        anchors_b = anchors_by_threshold(per_pos_b, thr=args.thr)
        a_total_b, _ = anchor_score(per_pos_b, anchors_b)
        print(f"Anchors (MJ <= {args.thr:g}): {anchors_b}")
        print("Number of anchors:", len(anchors_b))
        print("Anchor-only MJ score:", a_total_b)

        # Jaccard overlap requires same alignment length between A and B
        if len(per_pos_b) != len(per_pos):
            print(
                "[Compare] Cannot compute anchor Jaccard: alignments A and B have different lengths."
            )
        else:
            jac = jaccard(anchors, anchors_b)
            inter = sorted(set(anchors) & set(anchors_b))
            only_a = sorted(set(anchors) - set(anchors_b))
            only_b = sorted(set(anchors_b) - set(anchors))

            def _safe_sum(pp: List[Optional[float]], idxs: List[int]) -> float:
                s = 0.0
                for i in idxs:
                    v = pp[i - 1]
                    if v is None:
                        continue
                    s += float(v)
                return s

            a_only_sum = _safe_sum(per_pos, only_a)
            b_only_sum = _safe_sum(per_pos_b, only_b)

            print("[Compare] A-only anchor MJ sum:", a_only_sum)
            print("[Compare] B-only anchor MJ sum:", b_only_sum)

            # Universe for overlap probability: positions where BOTH alignments are scorable (non-None)
            eligible = [i for i, (va, vb) in enumerate(
                zip(per_pos, per_pos_b), start=1) if va is not None and vb is not None]
            elig_set = set(eligible)
            A = set(anchors) & elig_set
            B = set(anchors_b) & elig_set
            inter_elig = sorted(A & B)

            N = len(eligible)
            K = len(A)
            n = len(B)
            k = len(inter_elig)

            p_at_least = hypergeom_p_at_least(
                k, N, K, n) if N > 0 else float("nan")
            exp_overlap = (n * K / N) if N > 0 else float("nan")

            print(f"[Compare] Eligible positions (both scorable): N={N}")
            print(
                f"[Compare] Overlap k={k} with |A|={K}, |B|={n}; E[k]={fmt_float(exp_overlap, 2)}")
            print(
                f"[Compare] P(overlap >= k) under random placement: {fmt_prob(p_at_least)}")

            print(
                f"[Compare] Anchor-restricted Jaccard overlap (MJ <= {args.thr:g}): {jac:.3f}"
            )
            print(
                "[Compare] |A| =",
                len(anchors),
                "|B| =",
                len(anchors_b),
                "|A∩B| =",
                len(inter),
                "|A∪B| =",
                len(set(anchors) | set(anchors_b)),
            )
            print("[Compare] Shared anchors (A∩B):", inter)
            print("[Compare] A-only anchors:", only_a)
            for i in only_a:
                v = per_pos[i - 1]
                if v is None:
                    continue
                print(
                    f"[Compare] A-only anchor detail @pos {i}:",
                    s1[i - 1],
                    "↔",
                    s2[i - 1],
                    "MJ =",
                    v,
                )
            print("[Compare] B-only anchors:", only_b)
            for i in only_b:
                v = per_pos_b[i - 1]
                if v is None:
                    continue
                print(
                    f"[Compare] B-only anchor detail @pos {i}:",
                    s1b[i - 1],
                    "↔",
                    s2b[i - 1],
                    "MJ =",
                    v,
                )

        if args.compare_clustal:
            if len(s1) != len(s2) or len(s1b) != len(s2b):
                print(
                    "[Compare] Cannot compute Clustal anchor Jaccard: pair A or B is not aligned.",
                )
            elif len(s1) != len(s1b):
                print(
                    "[Compare] Cannot compute Clustal anchor Jaccard: alignments A and B have different lengths."
                )
            else:
                a_anchors, a_eligible = clustal_anchor_positions(
                    s1, s2, mode=args.clustal_anchor_mode
                )
                b_anchors, b_eligible = clustal_anchor_positions(
                    s1b, s2b, mode=args.clustal_anchor_mode
                )
                jac_c = jaccard(a_anchors, b_anchors)
                inter_c = sorted(set(a_anchors) & set(b_anchors))
                only_a_c = sorted(set(a_anchors) - set(b_anchors))
                only_b_c = sorted(set(b_anchors) - set(a_anchors))

                elig = sorted(set(a_eligible) & set(b_eligible))
                elig_set = set(elig)
                A = set(a_anchors) & elig_set
                B = set(b_anchors) & elig_set
                k = len(A & B)
                N = len(elig)
                K = len(A)
                n = len(B)
                p_at_least = hypergeom_p_at_least(
                    k, N, K, n) if N > 0 else float("nan")
                exp_overlap = (n * K / N) if N > 0 else float("nan")

                print(
                    f"[Compare][Clustal] Anchor mode: {args.clustal_anchor_mode}"
                )
                print(
                    f"[Compare][Clustal] Eligible positions (both scorable): N={N}"
                )
                print(
                    f"[Compare][Clustal] Overlap k={k} with |A|={K}, |B|={n}; E[k]={fmt_float(exp_overlap, 2)}"
                )
                print(
                    f"[Compare][Clustal] P(overlap >= k) under random placement: {fmt_prob(p_at_least)}"
                )
                print(
                    f"[Compare][Clustal] Jaccard overlap: {jac_c:.3f}"
                )
                print(
                    "[Compare][Clustal] |A| =",
                    len(a_anchors),
                    "|B| =",
                    len(b_anchors),
                    "|A∩B| =",
                    len(inter_c),
                    "|A∪B| =",
                    len(set(a_anchors) | set(b_anchors)),
                )
                print("[Compare][Clustal] Shared anchors (A∩B):", inter_c)
                print("[Compare][Clustal] A-only anchors:", only_a_c)
                print("[Compare][Clustal] B-only anchors:", only_b_c)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
