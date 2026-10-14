"""Paper Galaxy projection tests."""

import numpy as np

from ui import galaxy


def _data():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (6, 384)) + np.eye(384)[0]       # paper A clusters near axis 0
    b = rng.normal(0, 0.05, (5, 384)) + np.eye(384)[1]       # paper B near axis 1
    chunks = {name: [{"chunk_id": f"{name}_{i}", "page_number": 1, "section": "Results", "text": f"text {i}"}
                     for i in range(len(v))] for name, v in (("a.pdf", a), ("b.pdf", b))}
    return {"a.pdf": a, "b.pdf": b}, chunks


def test_projection_has_one_point_per_chunk_and_separates_papers():
    vectors, chunks = _data()
    proj = galaxy.project(vectors, chunks)
    pts = proj["points"]
    assert len(pts) == 11 and set(pts["paper"]) == {"a.pdf", "b.pdf"}
    centre_a = pts[pts.paper == "a.pdf"][["x", "y"]].mean().values
    centre_b = pts[pts.paper == "b.pdf"][["x", "y"]].mean().values
    assert np.linalg.norm(centre_a - centre_b) > 0.5          # distinct papers -> distinct regions
    assert 0 < sum(proj["explained"]) <= 1


def test_question_lands_near_the_matching_paper():
    vectors, chunks = _data()
    proj = galaxy.project(vectors, chunks)
    qx, qy = galaxy.place(proj, np.eye(384)[0])               # a question "about" paper A
    pts = proj["points"]
    dist = lambda name: np.linalg.norm(pts[pts.paper == name][["x", "y"]].mean().values - [qx, qy])
    assert dist("a.pdf") < dist("b.pdf")


def test_chart_builds_with_question_and_retrieval_highlights():
    vectors, chunks = _data()
    proj = galaxy.project(vectors, chunks)
    spec = galaxy.chart(proj, "section", "q?", galaxy.place(proj, np.eye(384)[0]), ["a.pdf_0", "a.pdf_1"]).to_dict()
    assert len(spec["layer"]) == 4                             # dots, links, ringed hits, star


def test_mismatched_or_empty_inputs_are_skipped():
    assert galaxy.project({}, {}) == {}
    assert galaxy.project({"x.pdf": np.ones((2, 384))}, {"x.pdf": [{"chunk_id": "1"}]}) == {}
