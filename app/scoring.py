"""Robust score calibration with median and MAD (median absolute deviation).

Each score is turned into a robust z-score within its calibration group
(either per judge, or per judge x item):

    z = (score - group_median) / (group_MAD * 1.4826)

and then mapped back onto the global score scale:

    calibrated = global_median + z * global_MAD_scaled

1.4826 makes MAD a consistent estimator of the standard deviation for
normally distributed data. Groups without spread (MAD == 0) get z = 0,
i.e. every score maps to the global median.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

MAD_K = 1.4826

SCOPE_JUDGE = "judge"
SCOPE_JUDGE_ITEM = "judge_item"
SCOPES = (SCOPE_JUDGE, SCOPE_JUDGE_ITEM)
SCOPE_LABELS = {
    SCOPE_JUDGE: "按评委（跨项目汇总）",
    SCOPE_JUDGE_ITEM: "按评委×项目",
}

TIE_EPS = 1e-9

# 评委排序分歧的四类选手对
DISAGREEMENT_CONCORDANT = "concordant"  # 同向：双方均非同分且偏好一致
DISAGREEMENT_REVERSED = "reversed"      # 逆向：双方均非同分但偏好相反
DISAGREEMENT_ONE_TIE = "one_tie"        # 单方同分：仅一方给两名选手同分
DISAGREEMENT_BOTH_TIE = "both_tie"      # 双方同分：两方都给两名选手同分
DISAGREEMENT_KIND_LABELS = {
    DISAGREEMENT_CONCORDANT: "同向",
    DISAGREEMENT_REVERSED: "逆向",
    DISAGREEMENT_ONE_TIE: "单方同分",
    DISAGREEMENT_BOTH_TIE: "双方同分",
}
# 共评选手少于该数时无法枚举任何选手对，标记「样本不足」且不计算分歧率。
DISAGREEMENT_MIN_SHARED = 2
DISAGREEMENT_KINDS = (
    DISAGREEMENT_CONCORDANT,
    DISAGREEMENT_REVERSED,
    DISAGREEMENT_ONE_TIE,
    DISAGREEMENT_BOTH_TIE,
)


def robust_params(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (median, raw MAD, MAD scaled by 1.4826)."""
    med = median(values)
    raw_mad = median(abs(v - med) for v in values)
    return med, raw_mad, raw_mad * MAD_K


def competition_rank(pairs: Iterable[tuple[Any, float]]) -> dict[Any, int]:
    """Rank (key, total) pairs highest-first with competition ranking (1224)."""
    ordered = sorted(pairs, key=lambda p: (-p[1], str(p[0])))
    ranks: dict[Any, int] = {}
    for i, (key, total) in enumerate(ordered):
        if i > 0 and abs(total - ordered[i - 1][1]) <= TIE_EPS:
            ranks[key] = ranks[ordered[i - 1][0]]
        else:
            ranks[key] = i + 1
    return ranks


def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def judge_disagreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """评委排序分歧：按「项目 × 评委对」枚举共评选手对并依原始分归类。

    对每个项目，取一对评委都在该项目打过分的选手（共评选手），枚举其中全部
    选手对，按两位评委各自给出的**原始分**（与校准范围、人工裁决无关）归入四类：

    - 同向（concordant）：双方都非同分，且对哪名选手更强的判断一致；
    - 逆向（reversed）：双方都非同分，但偏好相反；
    - 单方同分（one_tie）：仅一方给两名选手同分；
    - 双方同分（both_tie）：两方都给两名选手同分。

    某评委对在某项目的共评选手少于 2 名时标为「样本不足」，不给分歧率。
    分歧率（逆向占比）= 逆向对数 / 可比较对数，其中可比较对仅统计双方均
    非同分的选手对（同向 + 逆向）；同分对不计入分母。
    """
    # 保留首次出现顺序：评委、项目、选手。
    judges: list[str] = list(dict.fromkeys(r["judge"] for r in records))
    items: list[str] = list(dict.fromkeys(r["item"] for r in records))
    contestants_order: dict[str, int] = {}
    for r in records:
        if r["contestant"] not in contestants_order:
            contestants_order[r["contestant"]] = len(contestants_order)

    # (judge_a, judge_b, item) -> {contestant: (score_a, score_b)}
    scores: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    # (item, judge) -> contestant -> raw score
    raw_by_ij: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in records:
        raw_by_ij[(r["item"], r["judge"])][r["contestant"]] = r["score"]
    for item in items:
        for ai in range(len(judges)):
            a = judges[ai]
            sa = raw_by_ij.get((item, a))
            if not sa:
                continue
            for b in judges[ai + 1:]:
                sb = raw_by_ij.get((item, b))
                if not sb:
                    continue
                for contestant in sorted(
                    sa.keys() & sb.keys(), key=lambda c: contestants_order[c]
                ):
                    scores[(a, b, item)][contestant] = (sa[contestant], sb[contestant])

    def classify(va: float, vb: float, wa: float, wb: float) -> str:
        a_tie = abs(va - wa) <= TIE_EPS
        b_tie = abs(vb - wb) <= TIE_EPS
        if a_tie and b_tie:
            return DISAGREEMENT_BOTH_TIE
        if a_tie or b_tie:
            return DISAGREEMENT_ONE_TIE
        a_prefers_v = va > wa
        b_prefers_v = vb > wb
        return (
            DISAGREEMENT_CONCORDANT
            if a_prefers_v == b_prefers_v
            else DISAGREEMENT_REVERSED
        )

    pairs: list[dict[str, Any]] = []
    for item in items:
        for ai in range(len(judges)):
            a = judges[ai]
            for b in judges[ai + 1:]:
                shared = scores.get((a, b, item))
                if not shared:
                    continue  # 两位评委在该项目没有任何共评选手
                names = list(shared.keys())
                counts = {kind: 0 for kind in DISAGREEMENT_KINDS}
                reversed_pairs: list[dict[str, Any]] = []
                sufficient = len(names) >= DISAGREEMENT_MIN_SHARED
                for pi in range(len(names)):
                    for pj in range(pi + 1, len(names)):
                        cn, dn = names[pi], names[pj]
                        ca, cb = shared[cn]
                        da, db = shared[dn]
                        kind = classify(ca, cb, da, db)
                        counts[kind] += 1
                        if kind == DISAGREEMENT_REVERSED:
                            # 归一化方向：contestant_a 为 judge 更偏好的选手，
                            # contestant_b 为 other_judge 更偏好的选手。
                            if ca < da:  # judge a 更偏好 dn
                                cn, dn = dn, cn
                                ca, cb, da, db = da, db, ca, cb
                            reversed_pairs.append(
                                {
                                    "contestant_a": cn,
                                    "contestant_b": dn,
                                    "judge": a,
                                    "other_judge": b,
                                    "judge_a_score": ca,
                                    "judge_b_score": da,
                                    "other_a_score": cb,
                                    "other_b_score": db,
                                }
                            )
                comparable = counts[DISAGREEMENT_CONCORDANT] + counts[DISAGREEMENT_REVERSED]
                pairs.append(
                    {
                        "item": item,
                        "judge": a,
                        "other_judge": b,
                        "shared_count": len(names),
                        "sufficient_sample": sufficient,
                        "counts": counts,
                        "comparable_pairs": comparable if sufficient else None,
                        "reversal_rate": (
                            counts[DISAGREEMENT_REVERSED] / comparable
                            if sufficient and comparable > 0
                            else None
                        ),
                        "reversed_pairs": reversed_pairs if sufficient else [],
                    }
                )

    total_pairs = len(pairs)
    summary_counts = {
        kind: sum(p["counts"][kind] for p in pairs) for kind in DISAGREEMENT_KINDS
    }
    rated = [p for p in pairs if p["sufficient_sample"]]
    total_comparable = sum(p["comparable_pairs"] for p in rated)
    total_reversed = summary_counts[DISAGREEMENT_REVERSED]
    return {
        "kind_labels": dict(DISAGREEMENT_KIND_LABELS),
        "min_shared": DISAGREEMENT_MIN_SHARED,
        "summary": {
            "pair_rows": total_pairs,
            "rated_pairs": len(rated),
            "insufficient_pairs": total_pairs - len(rated),
            "counts": summary_counts,
            "comparable_pairs": total_comparable,
            "reversed_pairs": total_reversed,
            "reversal_rate": (
                total_reversed / total_comparable if total_comparable > 0 else None
            ),
        },
        "pairs": pairs,
    }


def analyze(
    records: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, tuple[bool, str]],
    scope: str = SCOPE_JUDGE,
) -> dict[str, Any]:
    """Run the full calibration pipeline.

    records: dicts with id, contestant, judge, item, score.
    decisions: score_id -> (use_raw, reason). Missing decisions default to
    the calibrated score.
    """
    if scope not in SCOPES:
        raise ValueError(f"未知校准范围: {scope}")

    def group_key(rec: Mapping[str, Any]) -> tuple:
        return (rec["judge"],) if scope == SCOPE_JUDGE else (rec["judge"], rec["item"])

    grouped: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[group_key(rec)].append(rec)

    group_params: dict[tuple, tuple[float, float, float]] = {
        key: robust_params([r["score"] for r in rs]) for key, rs in grouped.items()
    }

    if records:
        global_med, global_raw_mad, global_mad = robust_params(
            [r["score"] for r in records]
        )
    else:
        global_med = global_raw_mad = global_mad = 0.0

    # Per-record calibrated values.
    enriched: dict[int, dict[str, Any]] = {}
    for rec in records:
        med, raw_mad, mad = group_params[group_key(rec)]
        z = (rec["score"] - med) / mad if mad > 0 else 0.0
        calibrated = global_med + z * global_mad
        use_raw, reason = decisions.get(rec["id"], (False, ""))
        effective = rec["score"] if use_raw else calibrated
        enriched[rec["id"]] = {
            "id": rec["id"],
            "judge": rec["judge"],
            "raw": rec["score"],
            "group_median": med,
            "group_raw_mad": raw_mad,
            "group_mad": mad,
            "z": z,
            "calibrated": calibrated,
            "use_raw": bool(use_raw),
            "reason": reason or "",
            "effective": effective,
        }

    # Preserve first-appearance order for judges / items / contestants.
    judges: list[str] = list(dict.fromkeys(r["judge"] for r in records))
    items: list[str] = list(dict.fromkeys(r["item"] for r in records))
    contestants: list[str] = list(dict.fromkeys(r["contestant"] for r in records))

    by_ci: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        by_ci[(rec["contestant"], rec["item"])].append(rec)

    raw_totals: dict[str, float] = {}
    cal_totals: dict[str, float] = {}
    eff_totals: dict[str, float] = {}
    raw_override_count: dict[str, int] = defaultdict(int)

    contestant_blocks: list[dict[str, Any]] = []
    for contestant in contestants:
        item_blocks = []
        r_total = c_total = e_total = 0.0
        item_count = 0
        for item in items:
            rows = by_ci.get((contestant, item))
            if not rows:
                continue
            item_count += 1
            score_rows = [enriched[r["id"]] for r in rows]
            score_rows.sort(key=lambda s: judges.index(s["judge"]) if s["judge"] in judges else 0)
            raw_avg = _avg([s["raw"] for s in score_rows])
            cal_avg = _avg([s["calibrated"] for s in score_rows])
            eff_avg = _avg([s["effective"] for s in score_rows])
            r_total += raw_avg
            c_total += cal_avg
            e_total += eff_avg
            for s in score_rows:
                if s["use_raw"]:
                    raw_override_count[contestant] += 1
            item_blocks.append(
                {
                    "item": item,
                    "raw_avg": raw_avg,
                    "cal_avg": cal_avg,
                    "eff_avg": eff_avg,
                    "scores": score_rows,
                }
            )
        raw_totals[contestant] = r_total
        cal_totals[contestant] = c_total
        eff_totals[contestant] = e_total
        contestant_blocks.append(
            {
                "contestant": contestant,
                "item_count": item_count,
                "raw_total": r_total,
                "cal_total": c_total,
                "eff_total": e_total,
                "raw_overrides": raw_override_count[contestant],
                "items": item_blocks,
            }
        )

    raw_rank = competition_rank(raw_totals.items())
    cal_rank = competition_rank(cal_totals.items())
    eff_rank = competition_rank(eff_totals.items())

    for block in contestant_blocks:
        name = block["contestant"]
        block["raw_rank"] = raw_rank[name]
        block["cal_rank"] = cal_rank[name]
        block["eff_rank"] = eff_rank[name]
        block["rank_delta"] = raw_rank[name] - cal_rank[name]  # >0: moved up

    # Pairwise rank reversals between raw ranking and calibrated ranking.
    reversals: list[dict[str, Any]] = []
    for i, a in enumerate(contestants):
        for b in contestants[i + 1:]:
            ra, rb = raw_totals[a], raw_totals[b]
            ca, cb = cal_totals[a], cal_totals[b]
            if ra > rb + TIE_EPS and cb > ca + TIE_EPS:
                reversals.append(
                    {
                        "winner_before": a,
                        "winner_after": b,
                        "raw_rank_before": raw_rank[a],
                        "raw_rank_loser": raw_rank[b],
                        "cal_rank_winner": cal_rank[b],
                        "cal_rank_loser": cal_rank[a],
                        "raw_gap": ra - rb,
                        "cal_gap": cb - ca,
                    }
                )
            elif rb > ra + TIE_EPS and ca > cb + TIE_EPS:
                reversals.append(
                    {
                        "winner_before": b,
                        "winner_after": a,
                        "raw_rank_before": raw_rank[b],
                        "raw_rank_loser": raw_rank[a],
                        "cal_rank_winner": cal_rank[a],
                        "cal_rank_loser": cal_rank[b],
                        "raw_gap": rb - ra,
                        "cal_gap": ca - cb,
                    }
                )
    reversals.sort(key=lambda x: -(x["raw_gap"] + x["cal_gap"]))

    contestant_blocks.sort(key=lambda blk: (blk["eff_rank"], blk["contestant"]))

    groups = []
    for key, rs in sorted(grouped.items(), key=lambda kv: kv[0]):
        med, raw_mad, mad = group_params[key]
        entry = {
            "judge": key[0],
            "n": len(rs),
            "median": med,
            "raw_mad": raw_mad,
            "mad": mad,
        }
        if scope == SCOPE_JUDGE_ITEM:
            entry["item"] = key[1]
        groups.append(entry)

    return {
        "scope": scope,
        "scope_label": SCOPE_LABELS[scope],
        "reference": {
            "global_median": global_med,
            "global_raw_mad": global_raw_mad,
            "global_mad": global_mad,
            "mad_k": MAD_K,
        },
        "judges": judges,
        "items": items,
        "groups": groups,
        "contestants": contestant_blocks,
        "reversals": reversals,
        # 评委排序分歧只基于原始分，与校准范围、人工裁决无关，但随导入/合并实时刷新。
        "judge_disagreement": judge_disagreement(records),
        "counts": {
            "scores": len(records),
            "contestants": len(contestants),
            "judges": len(judges),
            "items": len(items),
            "decisions": sum(1 for v in decisions.values() if v[0]),
        },
    }


def compare_contestants(
    analysis: Mapping[str, Any],
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    """Head-to-head comparison of two contestants from a finished analysis.

    Pure read-only transform: totals and per-item judge averages are reused
    from ``analysis`` as-is, so the result always matches the current
    calibration scope and any manual decisions — nothing is recomputed or
    persisted. Items a contestant was never scored in are reported as
    ``None`` (无数据) and excluded from every delta and from lead/tie
    counts; they are never treated as zero.
    """
    if name_a == name_b:
        raise ValueError("请选择两名不同的选手进行对比。")
    blocks = {b["contestant"]: b for b in analysis["contestants"]}
    unknown = [n for n in (name_a, name_b) if n not in blocks]
    if unknown:
        raise LookupError("选手不存在: " + "、".join(unknown))
    block_a, block_b = blocks[name_a], blocks[name_b]
    map_a = {ib["item"]: ib for ib in block_a["items"]}
    map_b = {ib["item"]: ib for ib in block_b["items"]}

    def snapshot(ib: Mapping[str, Any] | None) -> dict[str, float] | None:
        if ib is None:
            return None
        return {"raw": ib["raw_avg"], "cal": ib["cal_avg"], "eff": ib["eff_avg"]}

    kinds = ("raw", "cal", "eff")
    rows: list[dict[str, Any]] = []
    lead_counts = {k: {"a_leads": 0, "b_leads": 0, "ties": 0} for k in kinds}
    compared = a_only = b_only = 0
    for item in analysis["items"]:
        snap_a = snapshot(map_a.get(item))
        snap_b = snapshot(map_b.get(item))
        delta: dict[str, float] | None = None
        lead: str | None = None
        if snap_a is not None and snap_b is not None:
            compared += 1
            delta = {k: snap_a[k] - snap_b[k] for k in kinds}
            for k in kinds:
                if delta[k] > TIE_EPS:
                    lead_counts[k]["a_leads"] += 1
                elif delta[k] < -TIE_EPS:
                    lead_counts[k]["b_leads"] += 1
                else:
                    lead_counts[k]["ties"] += 1
            d = delta["eff"]
            lead = "a" if d > TIE_EPS else ("b" if d < -TIE_EPS else "tie")
        elif snap_a is not None:
            a_only += 1
        elif snap_b is not None:
            b_only += 1
        rows.append(
            {"item": item, "a": snap_a, "b": snap_b, "delta": delta, "lead": lead}
        )

    totals_a = {
        "raw": block_a["raw_total"],
        "cal": block_a["cal_total"],
        "eff": block_a["eff_total"],
    }
    totals_b = {
        "raw": block_b["raw_total"],
        "cal": block_b["cal_total"],
        "eff": block_b["eff_total"],
    }
    return {
        "scope": analysis["scope"],
        "scope_label": analysis["scope_label"],
        "a": name_a,
        "b": name_b,
        "totals": {
            "a": totals_a,
            "b": totals_b,
            "delta": {k: totals_a[k] - totals_b[k] for k in kinds},
        },
        "ranks": {
            "a": {
                "raw": block_a["raw_rank"],
                "cal": block_a["cal_rank"],
                "eff": block_a["eff_rank"],
            },
            "b": {
                "raw": block_b["raw_rank"],
                "cal": block_b["cal_rank"],
                "eff": block_b["eff_rank"],
            },
        },
        "items": rows,
        "summary": {
            "items_total": len(analysis["items"]),
            "compared": compared,
            "a_only": a_only,
            "b_only": b_only,
            "raw": lead_counts["raw"],
            "cal": lead_counts["cal"],
            "eff": lead_counts["eff"],
        },
    }


EXCLUDE_JUDGE = "judge"
EXCLUDE_ITEM = "item"
EXCLUDE_TYPES = (EXCLUDE_JUDGE, EXCLUDE_ITEM)
EXCLUDE_LABELS = {
    EXCLUDE_JUDGE: "评委",
    EXCLUDE_ITEM: "项目",
}


def _rank_block_snapshot(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_total": block["raw_total"],
        "raw_rank": block["raw_rank"],
        "cal_total": block["cal_total"],
        "cal_rank": block["cal_rank"],
        "eff_total": block["eff_total"],
        "eff_rank": block["eff_rank"],
    }


def stability_experiment(
    records: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, tuple[bool, str]],
    scope: str,
    exclude_type: str,
    exclude_name: str,
) -> dict[str, Any]:
    """What-if analysis that recomputes everything with one judge/item removed.

    Baseline is computed from the full record set; the scenario drops every
    score attributed to ``exclude_name`` (a judge or an item) and re-runs the
    same calibration / aggregation / ranking pipeline under ``scope``.
    Decisions attached to removed scores drop with them. Inputs are never
    mutated and nothing here touches persistence.
    """
    if exclude_type not in EXCLUDE_TYPES:
        raise ValueError(f"未知排除类型: {exclude_type}")
    if scope not in SCOPES:
        raise ValueError(f"未知校准范围: {scope}")

    available = sorted({r[exclude_type] for r in records})
    if exclude_name not in available:
        raise LookupError(
            f"找不到要排除的{EXCLUDE_LABELS[exclude_type]}: {exclude_name}"
        )

    removed = [r for r in records if r[exclude_type] == exclude_name]
    removed_ids = {r["id"] for r in removed}
    kept = [r for r in records if r[exclude_type] != exclude_name]
    kept_decisions = {
        sid: verdict for sid, verdict in decisions.items() if sid not in removed_ids
    }

    baseline = analyze(records, decisions, scope=scope)
    scenario = analyze(kept, kept_decisions, scope=scope)

    affected_per_contestant: dict[str, int] = defaultdict(int)
    for r in removed:
        affected_per_contestant[r["contestant"]] += 1

    scenario_blocks = {b["contestant"]: b for b in scenario["contestants"]}

    rows: list[dict[str, Any]] = []
    for block in baseline["contestants"]:  # already ordered by baseline eff_rank
        name = block["contestant"]
        sblock = scenario_blocks.get(name)
        present = sblock is not None
        row: dict[str, Any] = {
            "contestant": name,
            "present": present,
            "affected_scores": affected_per_contestant.get(name, 0),
            "baseline": _rank_block_snapshot(block),
            "scenario": _rank_block_snapshot(sblock) if present else None,
        }
        if present:
            for kind in ("raw", "cal", "eff"):
                # Rank delta: baseline rank - scenario rank (positive = moved up).
                row[f"{kind}_rank_delta"] = (
                    block[f"{kind}_rank"] - sblock[f"{kind}_rank"]
                )
                # Total delta: scenario total - baseline total.
                row[f"{kind}_total_delta"] = (
                    sblock[f"{kind}_total"] - block[f"{kind}_total"]
                )
        else:
            for kind in ("raw", "cal", "eff"):
                row[f"{kind}_rank_delta"] = None
                row[f"{kind}_total_delta"] = None
        rows.append(row)

    present_rows = [r for r in rows if r["present"]]

    def _changed(kind: str) -> int:
        return sum(1 for r in present_rows if r[f"{kind}_rank_delta"] not in (0, None))

    eff_deltas = [r["eff_rank_delta"] for r in present_rows]
    eff_total_deltas = [abs(r["eff_total_delta"]) for r in present_rows]
    summary = {
        "raw_rank_changed": _changed("raw"),
        "cal_rank_changed": _changed("cal"),
        "eff_rank_changed": _changed("eff"),
        "eff_moved_up": sum(1 for d in eff_deltas if d and d > 0),
        "eff_moved_down": sum(1 for d in eff_deltas if d and d < 0),
        "max_eff_rank_gain": max((d for d in eff_deltas if d), default=0),
        "max_eff_rank_loss": max((-d for d in eff_deltas if d and d < 0), default=0),
        "max_eff_total_delta_abs": max(eff_total_deltas, default=0.0),
    }

    return {
        "scope": scope,
        "scope_label": SCOPE_LABELS[scope],
        "exclusion": {
            "type": exclude_type,
            "type_label": EXCLUDE_LABELS[exclude_type],
            "name": exclude_name,
        },
        "available": available,
        "removed_scores": len(removed),
        "baseline_counts": dict(baseline["counts"]),
        "scenario_counts": dict(scenario["counts"]),
        "summary": summary,
        "contestants": rows,
    }
