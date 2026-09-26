"""交叉点水深闭合测试：示例数据、包围样本选择、插值、Huber 平差与跳过原因。"""
import json

import pytest

from app.closure import check_closure
from app.sample import CLOSURE_SAMPLE
from app.schemas import ClosureRequest

# 两条十字交叉计划线：H: y=50 (0→100)；V: x=50 (0→100)，交点 (50,50)
H = {"id": "H", "start": [0, 50], "end": [100, 50], "swath": 20}
V = {"id": "V", "start": [50, 0], "end": [50, 100], "swath": 20}


def _run(payload):
    return check_closure(ClosureRequest.model_validate(payload))


def _sample_payload(samples, lines=None, options=None):
    return {"lines": lines or [H, V], "samples": samples,
            "options": options or {}}


def _s(lid, x, y, d):
    return {"line_id": lid, "x": x, "y": y, "depth": d}


# ----------------------------- 示例数据冒烟 -----------------------------
def test_sample_counts_and_status():
    res = _run({
        "lines": CLOSURE_SAMPLE["lines"],
        "samples": CLOSURE_SAMPLE["samples"],
        "options": CLOSURE_SAMPLE["options"],
    })
    s = res["summary"]
    # 网格 3×3=9 个 H-V 交点；H2D×V* 三个交点因 H2D 无样本跳过；H3×V3 缺包围跳过
    assert s["point_intersection_count"] == 12
    assert s["intersection_count"] == 8
    assert (s["ok_count"], s["anomaly_count"], s["indeterminate_count"]) == (7, 1, 0)
    assert s["component_count"] == 1
    assert s["unknown_line_ids"] == ["LX"]


def test_sample_skips_cover_three_reasons():
    res = _run({
        "lines": CLOSURE_SAMPLE["lines"],
        "samples": CLOSURE_SAMPLE["samples"],
        "options": CLOSURE_SAMPLE["options"],
    })
    kinds = [sk["kind"] for sk in res["skips"]]
    assert kinds.count("overlapping") == 1          # H2/H2D 共线重叠
    assert kinds.count("no_bracketing_samples") == 1  # H3×V3 横向偏离超限
    assert kinds.count("unknown_line") == 4          # LX + H2D 三个点交点
    ovl = next(sk for sk in res["skips"] if sk["kind"] == "overlapping")
    assert ovl["lines"] == ["H2", "H2D"]
    assert ovl["overlap_length"] == 800.0
    # 跳过记录编号连续且按类型排序
    assert [sk["id"] for sk in res["skips"]] == [f"SKP-{i:03d}" for i in range(1, 7)]


def test_sample_huber_downweights_gross_error():
    res = _run({
        "lines": CLOSURE_SAMPLE["lines"],
        "samples": CLOSURE_SAMPLE["samples"],
        "options": CLOSURE_SAMPLE["options"],
    })
    bad = next(x for x in res["intersections"] if x["status"] == "anomaly")
    assert bad["lines"] == ["H2", "V2"]
    # 粗差 +5 只污染该交点：权重被显著压低，校正残差仍超阈值
    assert bad["final_weight"] < 0.1
    assert abs(bad["corrected_residual"]) > 4.0
    # 其余交点权重为 1，偏置估计接近注入值（锚线 H1=0）
    for x in res["intersections"]:
        if x["id"] != bad["id"]:
            assert x["final_weight"] == 1.0
            assert abs(x["corrected_residual"]) <= 0.2
    biases = {l["line_id"]: l["bias"] for l in res["line_biases"]}
    assert biases["H1"] == 0.0
    # 注入 offset：H2 +0.4、V1 -0.2，偏置 b = -offset（粗差拉动在 0.25 以内）
    assert biases["H2"] == pytest.approx(-0.4, abs=0.25)
    assert biases["V1"] == pytest.approx(0.2, abs=0.1)


def test_interpolation_endpoints_and_ratio():
    res = _run({
        "lines": CLOSURE_SAMPLE["lines"],
        "samples": CLOSURE_SAMPLE["samples"],
        "options": CLOSURE_SAMPLE["options"],
    })
    xpt = next(x for x in res["intersections"] if x["lines"] == ["H1", "V1"])
    h1 = xpt["interpolations"]["H1"]
    v1 = xpt["interpolations"]["V1"]
    # 交点 (150,120)：H1 里程 150 落在 100/200 之间 t=0.5；V1 里程 120 落在 100/200 间 t=0.2
    assert h1["ratio_t"] == 0.5
    assert v1["ratio_t"] == 0.2
    assert [e["index"] for e in h1["endpoints"]] == [1, 2]
    assert [e["index"] for e in v1["endpoints"]] == [34, 35]  # H1/H2/H3 各 11 个后 V1 起 33


# ----------------------------- 交点与重叠 -----------------------------
def test_single_crossing_basic_closure():
    samples = [
        _s("H", 0, 50, 10.0), _s("H", 50, 50, 10.0), _s("H", 100, 50, 10.0),
        _s("V", 50, 0, 10.5), _s("V", 50, 50, 10.5), _s("V", 50, 100, 10.5),
    ]
    res = _run(_sample_payload(samples))
    assert res["summary"]["intersection_count"] == 1
    xpt = res["intersections"][0]
    assert xpt["id"] == "XPT-001"
    assert xpt["point"] == [50.0, 50.0]
    assert xpt["lines"] == ["H", "V"]
    assert xpt["raw_misclosure"] == -0.5
    # 两节点树（观测 1 = 节点 2-1）：无冗余不可判，但偏置仍给出
    assert xpt["status"] == "indeterminate"
    assert xpt["corrected_residual"] == 0.0  # 单观测可被偏置完全吸收
    comp = res["components"][0]
    assert comp["anchor_line"] == "H"
    assert comp["redundant"] is False
    assert comp["redundancy"] == 0
    biases = {l["line_id"]: l for l in res["line_biases"]}
    assert biases["H"]["bias"] == 0.0 and biases["H"]["is_anchor"] is True
    assert biases["V"]["bias"] == -0.5


def test_overlapping_collinear_lines_skipped():
    h2 = {"id": "H2", "start": [10, 50], "end": [90, 50], "swath": 10}  # 与 H 共线重叠
    samples = [_s("H", 0, 50, 10), _s("H", 100, 50, 10),
               _s("H2", 10, 50, 10), _s("H2", 90, 50, 10),
               _s("V", 50, 0, 10), _s("V", 50, 100, 10)]
    res = _run(_sample_payload(samples, lines=[H, h2, V]))
    kinds = {(tuple(sk["lines"]), sk["kind"]) for sk in res["skips"]}
    assert (("H", "H2"), "overlapping") in kinds
    # H×V、H2×V 仍为点状交点且有效
    assert res["summary"]["point_intersection_count"] == 2


def test_parallel_disjoint_lines_not_reported():
    h2 = {"id": "H2", "start": [0, 80], "end": [100, 80], "swath": 10}  # 平行不相交
    samples = [_s("H", 0, 50, 10), _s("H", 100, 50, 10),
               _s("H2", 0, 80, 10), _s("H2", 100, 80, 10),
               _s("V", 50, 0, 10), _s("V", 50, 100, 10)]
    res = _run(_sample_payload(samples, lines=[H, h2, V]))
    assert res["skips"] == []
    assert res["summary"]["point_intersection_count"] == 2  # H×V 与 H2×V


# ----------------------------- 包围样本选择 -----------------------------
def test_missing_enclosing_pair_skip():
    # H 在交点两侧没有样本（只有 0 和 40，里程未跨过 50）
    samples = [
        _s("H", 0, 50, 10.0), _s("H", 40, 50, 10.0),
        _s("V", 50, 0, 10.0), _s("V", 50, 50, 10.0), _s("V", 50, 100, 10.0),
    ]
    res = _run(_sample_payload(samples))
    assert res["intersections"] == []
    sk = res["skips"][0]
    assert sk["kind"] == "no_bracketing_samples"
    assert sk["line"] == "H"
    assert res["summary"]["skip_counts"]["no_bracketing_samples"] == 1


def test_projection_distance_limit_filters_pair():
    # H 的 40/60 样本横向偏离 40，超出包围距离 15，不能包围交点
    samples = [
        _s("H", 0, 50, 10.0), _s("H", 40, 90, 10.0),
        _s("H", 60, 90, 10.0), _s("H", 100, 50, 10.0),
        _s("V", 50, 0, 10.0), _s("V", 50, 100, 10.0),
    ]
    res = _run(_sample_payload(samples, options={"max_enclosure_distance": 15,
                                                 "huber_delta": 0.3, "residual_threshold": 0.5}))
    assert res["intersections"] == []
    assert res["skips"][0]["kind"] == "no_bracketing_samples"
    # 放宽包围距离后即可包围（投影距离 40）
    res2 = _run(_sample_payload(samples, options={"max_enclosure_distance": 45,
                                                  "huber_delta": 0.3, "residual_threshold": 0.5}))
    assert res2["summary"]["intersection_count"] == 1


def test_candidate_pair_min_distance_sum_then_original_index():
    # 两组候选均跨过里程 50：(0,100) 与 (49,51)；后者两端到交点距离之和更小
    samples = [
        _s("H", 0, 50, 10.0), _s("H", 49, 50, 10.0), _s("H", 51, 50, 10.0),
        _s("H", 100, 50, 10.0),
        _s("V", 50, 0, 10.0), _s("V", 50, 100, 10.0),
    ]
    res = _run(_sample_payload(samples))
    xpt = res["intersections"][0]
    assert [e["index"] for e in xpt["interpolations"]["H"]["endpoints"]] == [1, 2]


def test_reverse_acquisition_direction_supported():
    # H 样本按里程 100→0 采集（与计划线 start→end 反向），仍应跨过交点插值
    samples = [
        _s("H", 100, 50, 12.0), _s("H", 0, 50, 10.0),
        _s("V", 50, 0, 11.0), _s("V", 50, 100, 11.0),
    ]
    res = _run(_sample_payload(samples))
    xpt = res["intersections"][0]
    h = xpt["interpolations"]["H"]
    # t 始终按 (s_x-s_a)/(s_b-s_a) 计算，a 为原序在前的样本（里程 100）
    assert h["ratio_t"] == 0.5
    assert h["interpolated_depth"] == 11.0


def test_linear_depth_interpolation_value():
    # 水深从 10 线性变到 20，交点 t=0.5 -> 15
    samples = [
        _s("H", 0, 50, 10.0), _s("H", 100, 50, 20.0),
        _s("V", 50, 0, 15.0), _s("V", 50, 100, 15.0),
    ]
    res = _run(_sample_payload(samples))
    h = res["intersections"][0]["interpolations"]["H"]
    assert h["ratio_t"] == 0.5
    assert h["interpolated_depth"] == 15.0
    assert h["endpoints"][0]["depth"] == 10.0 and h["endpoints"][1]["depth"] == 20.0


# ----------------------------- 未知测线 / 空输入 -----------------------------
def test_unknown_sample_line_and_uncollected_planned_line():
    samples = [
        _s("H", 0, 50, 10.0), _s("H", 100, 50, 10.0),
        _s("V", 50, 0, 10.0), _s("V", 50, 100, 10.0),
        _s("ZZ", 0, 0, 10.0), _s("ZZ", 1, 1, 10.0),
    ]
    # 去掉 V 的样本：V 成为“该计划线无采集样本”的 unknown_line 交点级跳过
    res = _run(_sample_payload(samples[:2] + samples[4:]))
    assert res["summary"]["intersection_count"] == 0
    kinds = {sk["kind"]: sk for sk in res["skips"]}
    assert kinds["unknown_line"]["line"] == "V"
    # ZZ 是任务输入未定义的未知测线样本
    zz = [sk for sk in res["skips"] if sk.get("line") == "ZZ"]
    assert zz and zz[0]["kind"] == "unknown_line" and zz[0]["sample_count"] == 2
    assert res["summary"]["unknown_line_ids"] == ["ZZ"]


def test_empty_inputs():
    res = _run({"lines": [], "samples": [], "options": {}})
    assert res["summary"] == {
        **res["summary"],
        "planned_line_count": 0, "sample_count": 0,
        "point_intersection_count": 0, "intersection_count": 0,
        "ok_count": 0, "anomaly_count": 0, "indeterminate_count": 0,
        "skip_count": 0, "component_count": 0,
    }
    assert res["intersections"] == [] and res["skips"] == []
    assert res["components"] == [] and res["line_biases"] == []


# ----------------------------- 偏置网 / Huber -----------------------------
def _grid_lines():
    return [
        {"id": "H1", "start": [0, 0], "end": [100, 0], "swath": 10},
        {"id": "H2", "start": [0, 100], "end": [100, 100], "swath": 10},
        {"id": "V1", "start": [0, 0], "end": [0, 100], "swath": 10},
        {"id": "V2", "start": [100, 0], "end": [100, 100], "swath": 10},
    ]  # 矩形 4 交点；注意交点位于样本端点


def test_square_network_redundant_component_and_biases():
    # 每条线在 0,50,100 三个里程采样；偏置（实测 = 真值 + offset）：
    # H1 offset 0、H2 +0.4、V1 -0.2、V2 +0.25，基准水深 20
    # 方程 b_li - b_lj = d_j - d_i，估计出的偏置 b = -offset（校正量加在实测上）
    offset = {"H1": 0.0, "H2": 0.4, "V1": -0.2, "V2": 0.25}

    samples = []
    for x in (0, 50, 100):
        samples.append(_s("H1", x, 0, 20.0 + offset["H1"]))
    for x in (0, 50, 100):
        samples.append(_s("H2", x, 100, 20.0 + offset["H2"]))
    for y in (0, 50, 100):
        samples.append(_s("V1", 0, y, 20.0 + offset["V1"]))
    for y in (0, 50, 100):
        samples.append(_s("V2", 100, y, 20.0 + offset["V2"]))
    res = _run(_sample_payload(samples, lines=_grid_lines(),
                               options={"max_enclosure_distance": 5, "huber_delta": 0.3,
                                        "residual_threshold": 0.5}))
    assert res["summary"]["intersection_count"] == 4
    comp = res["components"][0]
    assert comp["lines"] == ["H1", "H2", "V1", "V2"]
    assert comp["anchor_line"] == "H1"
    assert comp["redundancy"] == 1 and comp["redundant"] is True
    biases = {l["line_id"]: l["bias"] for l in res["line_biases"]}
    assert biases["H1"] == 0.0
    assert biases["H2"] == pytest.approx(-0.4, abs=1e-6)
    assert biases["V1"] == pytest.approx(0.2, abs=1e-6)
    assert biases["V2"] == pytest.approx(-0.25, abs=1e-6)
    assert all(x["status"] == "ok" for x in res["intersections"])


def _ladder_lines():
    # 2 横 × 3 竖 梯子网：6 交点，5 节点，冗余度 2（单粗差可被 Huber 定位到两邻边）
    return [
        {"id": "H1", "start": [0, 0], "end": [100, 0], "swath": 10},
        {"id": "H2", "start": [0, 100], "end": [100, 100], "swath": 10},
        {"id": "V1", "start": [0, 0], "end": [0, 100], "swath": 10},
        {"id": "V2", "start": [50, 0], "end": [50, 100], "swath": 10},
        {"id": "V3", "start": [100, 0], "end": [100, 100], "swath": 10},
    ]


def test_anomaly_threshold_respected():
    # V2 在里程 100（H2×V2 交点）处粗差 +5：冗余度 2 时 Huber 把粗差压在 V2 两邻边
    samples = []
    for x in (0, 50, 100):
        samples.append(_s("H1", x, 0, 20.0))
    for x in (0, 50, 100):
        samples.append(_s("H2", x, 100, 20.0))
    for y in (0, 50, 100):
        samples.append(_s("V1", 0, y, 20.0))
    for k, y in enumerate((0, 50, 100)):
        samples.append(_s("V2", 50, y, 20.0 + (5.0 if k == 2 else 0.0)))
    for y in (0, 50, 100):
        samples.append(_s("V3", 100, y, 20.0))
    res = _run(_sample_payload(samples, lines=_ladder_lines(),
                               options={"max_enclosure_distance": 5, "huber_delta": 0.3,
                                        "residual_threshold": 0.5}))
    bad = [x for x in res["intersections"] if x["status"] == "anomaly"]
    assert {tuple(x["lines"]) for x in bad} == {("H1", "V2"), ("H2", "V2")}
    assert all(x["final_weight"] < 0.2 for x in bad)
    clean = [x for x in res["intersections"] if x["status"] == "ok"]
    assert len(clean) == 4 and all(x["final_weight"] == 1.0 for x in clean)
    # 放宽残差阈值后不再判异常
    res2 = _run(_sample_payload(samples, lines=_ladder_lines(),
                                options={"max_enclosure_distance": 5, "huber_delta": 0.3,
                                         "residual_threshold": 5.0}))
    assert all(x["status"] == "ok" for x in res2["intersections"])


def test_two_components_anchored_independently():
    # 两对互不相交的十字线，构成两个独立分量，各自字典序最小线为零
    lines = [
        {"id": "A", "start": [0, 0], "end": [100, 0], "swath": 10},
        {"id": "B", "start": [50, -50], "end": [50, 50], "swath": 10},
        {"id": "C", "start": [0, 200], "end": [100, 200], "swath": 10},
        {"id": "D", "start": [50, 150], "end": [50, 250], "swath": 10},
    ]
    samples = [
        _s("A", 0, 0, 20.0), _s("A", 100, 0, 20.0),
        _s("B", 50, -50, 20.3), _s("B", 50, 50, 20.3),
        _s("C", 0, 200, 20.0), _s("C", 100, 200, 20.0),
        _s("D", 50, 150, 19.6), _s("D", 50, 250, 19.6),
    ]
    res = _run(_sample_payload(samples, lines=lines))
    assert res["summary"]["component_count"] == 2
    c1, c2 = res["components"]
    assert c1["anchor_line"] == "A" and c2["anchor_line"] == "C"
    assert c1["lines"] == ["A", "B"] and c2["lines"] == ["C", "D"]
    biases = {l["line_id"]: l["bias"] for l in res["line_biases"]}
    assert biases["A"] == 0.0 and biases["C"] == 0.0


# ----------------------------- 确定性 -----------------------------
def test_deterministic_output():
    payload = {"lines": CLOSURE_SAMPLE["lines"], "samples": CLOSURE_SAMPLE["samples"],
               "options": CLOSURE_SAMPLE["options"]}
    a = json.dumps(_run(payload), sort_keys=True, ensure_ascii=False)
    b = json.dumps(_run(payload), sort_keys=True, ensure_ascii=False)
    assert a == b


def test_intersection_ids_sorted_by_point():
    payload = {"lines": CLOSURE_SAMPLE["lines"], "samples": CLOSURE_SAMPLE["samples"],
               "options": CLOSURE_SAMPLE["options"]}
    res = _run(payload)
    pts = [tuple(x["point"]) for x in res["intersections"]]
    assert pts == sorted(pts)
