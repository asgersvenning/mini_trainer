"""Focused numerical checks for the research transforms, not package APIs."""

import numpy as np
from advanced import PNS
from benchmark import Coordinates, unit
from numpy.testing import assert_allclose


def test_chart_roundtrip_and_batch_insertion():
    rng = np.random.default_rng(7)
    train, query = unit(rng.normal(size=(100, 9))), unit(rng.normal(size=(12, 9)))
    for kind in ["ambient", "log_mean", "log_intrinsic", "stereographic", "equal_area"]:
        model = Coordinates(kind).fit(train)
        z = model.transform(query)
        assert_allclose(model.inverse(z), query, atol=2e-12)
        assert_allclose(model.transform(query[:1]), z[:1], atol=2e-12)
        assert_allclose(model.transform(model.mu[None]), 0 if kind != "ambient" else model.mu[None], atol=2e-12)


def test_pns_known_small_circle_and_heldout_roundtrip():
    angle = np.linspace(-np.pi, np.pi, 80, endpoint=False)
    radius = 0.4
    x = np.column_stack((np.full(len(angle), np.cos(radius)), np.sin(radius) * np.cos(angle), np.sin(radius) * np.sin(angle)))
    model = PNS().fit(x)
    assert_allclose(model.stages[0][1], radius, atol=1e-5)
    q = unit(np.random.default_rng(4).normal(size=(20, 3)))
    assert_allclose(model.inverse(model.transform(q)), q, atol=1e-10)


def test_truncation_is_lossy_and_radial_endpoints_are_not_invertible():
    rng = np.random.default_rng(4)
    x = unit(rng.normal(size=(100, 9)))
    model = Coordinates("log_radial").fit(x)
    q = model.mu[None]
    # The clamped empirical radial mapping cannot encode unseen endpoint radii.
    assert np.linalg.norm(model.inverse(model.transform(q)) - q) > 1e-3
