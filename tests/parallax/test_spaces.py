"""Tests for action/observation space definitions."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from parallax.spaces import Box, Discrete, MultiBinary, MultiDiscrete, PyTreeSpace


class TestDiscrete:
    def test_sample_shape_scalar(self):
        sample = Discrete(5).sample(key=jax.random.key(0))
        assert sample.shape == ()

    def test_sample_shape_batched(self):
        sample = Discrete(3, shape=(4,)).sample(key=jax.random.key(0))
        assert sample.shape == (4,)

    def test_sample_in_range(self):
        space = Discrete(5)
        keys = jax.random.split(jax.random.key(0), 200)
        samples = jax.vmap(lambda k: space.sample(key=k))(keys)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < 5)

    def test_sample_dtype(self):
        sample = Discrete(5).sample(key=jax.random.key(0))
        assert jnp.issubdtype(sample.dtype, jnp.integer)

    def test_determinism(self):
        space = Discrete(10)
        s1 = space.sample(key=jax.random.key(42))
        s2 = space.sample(key=jax.random.key(42))
        npt.assert_array_equal(s1, s2)

    def test_different_keys_vary(self):
        space = Discrete(100)
        keys = jax.random.split(jax.random.key(0), 50)
        samples = jax.vmap(lambda k: space.sample(key=k))(keys)
        assert len(jnp.unique(samples)) > 1


class TestBox:
    def test_sample_shape(self):
        sample = Box(low=-1.0, high=1.0, shape=(8,)).sample(key=jax.random.key(0))
        assert sample.shape == (8,)

    def test_sample_shape_scalar(self):
        sample = Box(low=0.0, high=1.0, shape=()).sample(key=jax.random.key(0))
        assert sample.shape == ()

    def test_sample_in_bounds(self):
        space = Box(low=-2.0, high=3.0, shape=(100,))
        sample = space.sample(key=jax.random.key(0))
        assert jnp.all(sample >= -2.0)
        assert jnp.all(sample <= 3.0)

    def test_sample_dtype(self):
        sample = Box(low=0.0, high=1.0, shape=(4,)).sample(key=jax.random.key(0))
        assert jnp.issubdtype(sample.dtype, jnp.floating)

    def test_determinism(self):
        space = Box(low=0.0, high=1.0, shape=(4,))
        s1 = space.sample(key=jax.random.key(0))
        s2 = space.sample(key=jax.random.key(0))
        npt.assert_array_equal(s1, s2)

    def test_array_bounds(self):
        """Per-dimension low/high bounds are respected."""
        low = jnp.array([0.0, -1.0])
        high = jnp.array([1.0, 2.0])
        space = Box(low=low, high=high, shape=(2,))
        keys = jax.random.split(jax.random.key(0), 100)
        samples = jax.vmap(lambda k: space.sample(key=k))(keys)
        assert jnp.all(samples[:, 0] >= 0.0) and jnp.all(samples[:, 0] <= 1.0)
        assert jnp.all(samples[:, 1] >= -1.0) and jnp.all(samples[:, 1] <= 2.0)

    def test_multidim_shape(self):
        sample = Box(low=0.0, high=1.0, shape=(3, 4)).sample(key=jax.random.key(0))
        assert sample.shape == (3, 4)


class TestMultiDiscrete:
    def test_sample_shape(self):
        space = MultiDiscrete(actions_per_dim=jnp.array([3, 5, 2]))
        sample = space.sample(key=jax.random.key(0))
        assert sample.shape == (3,)

    def test_shape_property(self):
        space = MultiDiscrete(actions_per_dim=jnp.array([3, 5, 2]))
        assert space.shape == (3,)

    def test_sample_per_dim_range(self):
        actions = jnp.array([3, 5, 2])
        space = MultiDiscrete(actions_per_dim=actions)
        keys = jax.random.split(jax.random.key(0), 100)
        samples = jax.vmap(lambda k: space.sample(key=k))(keys)
        for i, n in enumerate([3, 5, 2]):
            assert jnp.all(samples[:, i] >= 0)
            assert jnp.all(samples[:, i] < n)

    def test_sample_dtype(self):
        space = MultiDiscrete(actions_per_dim=jnp.array([3, 5]))
        sample = space.sample(key=jax.random.key(0))
        assert jnp.issubdtype(sample.dtype, jnp.integer)

    def test_determinism(self):
        space = MultiDiscrete(actions_per_dim=jnp.array([4, 6, 3]))
        s1 = space.sample(key=jax.random.key(0))
        s2 = space.sample(key=jax.random.key(0))
        npt.assert_array_equal(s1, s2)


class TestMultiBinary:
    def test_sample_shape(self):
        sample = MultiBinary(n=6).sample(key=jax.random.key(0))
        assert sample.shape == (6,)

    def test_shape_property(self):
        assert MultiBinary(n=4).shape == (4,)

    def test_sample_values_binary(self):
        sample = MultiBinary(n=100).sample(key=jax.random.key(0))
        assert jnp.all((sample == 0) | (sample == 1))

    def test_sample_dtype(self):
        sample = MultiBinary(n=4).sample(key=jax.random.key(0))
        assert sample.dtype == jnp.int8

    def test_determinism(self):
        space = MultiBinary(n=8)
        s1 = space.sample(key=jax.random.key(0))
        s2 = space.sample(key=jax.random.key(0))
        npt.assert_array_equal(s1, s2)


class TestPyTreeSpace:
    def test_dict_space(self):
        space = PyTreeSpace(
            spaces={
                "pos": Box(low=-1.0, high=1.0, shape=(2,)),
                "vel": Box(low=-5.0, high=5.0, shape=(2,)),
            }
        )
        sample = space.sample(key=jax.random.key(0))
        assert isinstance(sample, dict)
        assert sample["pos"].shape == (2,)
        assert sample["vel"].shape == (2,)

    def test_mixed_space_types(self):
        space = PyTreeSpace(
            spaces={
                "continuous": Box(low=0.0, high=1.0, shape=(3,)),
                "discrete": Discrete(5),
            }
        )
        sample = space.sample(key=jax.random.key(0))
        assert sample["continuous"].shape == (3,)
        assert sample["discrete"].shape == ()

    def test_nested_space(self):
        space = PyTreeSpace(
            spaces={
                "obs": {
                    "visual": Box(low=0.0, high=1.0, shape=(4,)),
                    "scalar": Box(low=-1.0, high=1.0, shape=()),
                },
            }
        )
        sample = space.sample(key=jax.random.key(0))
        assert "obs" in sample
        assert sample["obs"]["visual"].shape == (4,)
        assert sample["obs"]["scalar"].shape == ()

    def test_determinism(self):
        space = PyTreeSpace(
            spaces={
                "a": Box(low=0.0, high=1.0, shape=(2,)),
                "b": Discrete(5),
            }
        )
        s1 = space.sample(key=jax.random.key(0))
        s2 = space.sample(key=jax.random.key(0))
        npt.assert_array_equal(s1["a"], s2["a"])
        npt.assert_array_equal(s1["b"], s2["b"])

    def test_bounds_respected(self):
        space = PyTreeSpace(
            spaces={
                "bounded": Box(low=-1.0, high=1.0, shape=(10,)),
            }
        )
        sample = space.sample(key=jax.random.key(0))
        assert jnp.all(sample["bounded"] >= -1.0)
        assert jnp.all(sample["bounded"] <= 1.0)
