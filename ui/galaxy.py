"""
Paper Galaxy
------------
A 2-D "semantic map" of every chunk in the knowledge base, used to make
retrieval visible in the UI and the viva:

  - each dot is one chunk (passage); dots close together have similar
    meaning, because they come from the same all-MiniLM embedding space the
    retriever searches
  - the 384-dimensional embeddings are projected to 2-D with PCA (plain
    numpy SVD - no extra dependency); the same projection places the
    user's question as a star
  - the chunks the agent would retrieve are ringed and linked to the star

PCA is linear, so the map shows the main directions of variation; distances
are approximate - the retriever itself always works in the full 384-D space.
"""

import numpy as np
import pandas as pd


def project(vectors_by_paper: dict, chunks_by_paper: dict) -> dict:
    """
    Args:
        vectors_by_paper: {paper: (n_chunks, dim) array}
        chunks_by_paper:  {paper: [chunk dicts]} in the same order as the vectors

    Returns:
        {"points": DataFrame(x, y, paper, section, page, chunk_id, snippet),
         "mean": array, "components": (2, dim) array, "explained": [float, float]}
        or {} when there is nothing to plot.
    """
    rows, matrices = [], []
    for paper, vectors in vectors_by_paper.items():
        chunks = chunks_by_paper.get(paper, [])
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not len(chunks):
            continue
        matrices.append(vectors)
        for chunk in chunks:
            text = " ".join(chunk["text"].split())
            rows.append({"paper": paper, "section": chunk.get("section", "Unknown"),
                         "page": chunk["page_number"], "chunk_id": chunk["chunk_id"],
                         "snippet": text[:160] + ("…" if len(text) > 160 else "")})
    if not matrices:
        return {}

    X = np.vstack(matrices)
    if len(X) < 3:
        return {}
    mean = X.mean(axis=0)
    _, singular, vt = np.linalg.svd(X - mean, full_matrices=False)
    components = vt[:2]
    coords = (X - mean) @ components.T
    variance = singular ** 2
    explained = (variance[:2] / variance.sum()).tolist() if variance.sum() else [0.0, 0.0]

    points = pd.DataFrame(rows)
    points["x"], points["y"] = coords[:, 0], coords[:, 1]
    return {"points": points, "mean": mean, "components": components, "explained": explained}


def place(projection: dict, vector) -> tuple[float, float]:
    """Projects one more vector (e.g. the question) onto the same 2-D map."""
    xy = (np.asarray(vector, dtype="float32") - projection["mean"]) @ projection["components"].T
    return float(xy[0]), float(xy[1])


def chart(projection: dict, color_by: str = "paper", question: str = None, question_xy=None,
          retrieved_ids=(), dark: bool = False):
    """Builds the interactive Altair chart (zoom/pan, hover tooltips)."""
    import altair as alt

    palette = ["#007AFF", "#FF9500", "#AF52DE", "#34C759", "#FF2D55", "#5AC8FA", "#FFCC00", "#5856D6"]
    text_color = "#F2F2F7" if dark else "#1C1C1E"
    grid = "#2C2C2E" if dark else "#E5E5EA"
    points = projection["points"].copy()
    points["retrieved"] = points["chunk_id"].isin(set(retrieved_ids))

    field = "paper" if color_by == "paper" else "section"
    tooltip = [alt.Tooltip("paper:N", title="Paper"), alt.Tooltip("section:N", title="Section"),
               alt.Tooltip("page:Q", title="Page"), alt.Tooltip("snippet:N", title="Passage")]
    base = alt.Chart(points).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        color=alt.Color(f"{field}:N", scale=alt.Scale(range=palette),
                        legend=alt.Legend(orient="bottom", title=None, labelColor=text_color, labelLimit=260)),
        tooltip=tooltip,
    )
    layers = [base.mark_circle(size=70, opacity=0.55 if question_xy else 0.8, stroke="white", strokeWidth=0.4)]

    if question_xy is not None:
        hits = points[points["retrieved"]]
        if len(hits):
            links = pd.DataFrame({"x": [question_xy[0]] * len(hits), "y": [question_xy[1]] * len(hits),
                                  "x2": hits["x"].values, "y2": hits["y"].values})
            layers.append(alt.Chart(links).mark_rule(color="#FF9500", strokeWidth=1.4, opacity=0.7,
                                                      strokeDash=[4, 3])
                          .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q"))
            layers.append(alt.Chart(hits).mark_circle(size=260, opacity=1, stroke="#FF9500", strokeWidth=2.5)
                          .encode(x="x:Q", y="y:Q", color=alt.Color(f"{field}:N", scale=alt.Scale(range=palette),
                                                                    legend=None), tooltip=tooltip))
        star = pd.DataFrame({"x": [question_xy[0]], "y": [question_xy[1]], "label": [question or "Your question"]})
        star_path = ("M0,-1L0.29,-0.4L0.95,-0.31L0.47,0.15L0.59,0.81L0,0.5L-0.59,0.81L-0.47,0.15L-0.95,-0.31"
                     "L-0.29,-0.4Z")
        layers.append(alt.Chart(star).mark_point(shape=star_path, size=900, filled=True, color="#FFCC00",
                                                 stroke="#1C1C1E" if not dark else "#FFFFFF", strokeWidth=1.2)
                      .encode(x="x:Q", y="y:Q", tooltip=[alt.Tooltip("label:N", title="Question")]))

    return (alt.layer(*layers)
            .properties(height=520, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False, domainColor=grid)
            .interactive())
