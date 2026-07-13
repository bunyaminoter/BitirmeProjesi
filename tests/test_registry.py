"""Tests for the Registry pattern."""

from __future__ import annotations

import pytest

from src.core.registry import Registry


class TestRegistry:
    """Test suite for the Registry class."""

    def test_register_and_build(self):
        """Test basic registration and building."""
        registry = Registry("test")

        @registry.register("my_component")
        class MyComponent:
            def __init__(self, value: int = 0):
                self.value = value

        instance = registry.build("my_component", value=42)
        assert instance.value == 42

    def test_register_duplicate_raises(self):
        """Test that duplicate registration raises ValueError."""
        registry = Registry("test")

        @registry.register("duplicate")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @registry.register("duplicate")
            class Second:
                pass

    def test_build_unknown_raises(self):
        """Test that building an unknown name raises KeyError."""
        registry = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            registry.build("nonexistent")

    def test_list_registered(self):
        """Test listing registered components."""
        registry = Registry("test")

        @registry.register("b_component")
        class B:
            pass

        @registry.register("a_component")
        class A:
            pass

        assert registry.list_registered() == ["a_component", "b_component"]

    def test_contains(self):
        """Test __contains__ check."""
        registry = Registry("test")

        @registry.register("exists")
        class Exists:
            pass

        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self):
        """Test __len__."""
        registry = Registry("test")
        assert len(registry) == 0

        @registry.register("one")
        class One:
            pass

        assert len(registry) == 1

    def test_get_class(self):
        """Test getting a class without instantiation."""
        registry = Registry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        cls = registry.get("my_class")
        assert cls is MyClass
