"""
Unit tests for 3-class label generation.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dl.data.labels import create_3class_labels, LstmClassIndex, get_class_name


def test_create_3class_labels_basic():
    """Test basic 3-class label generation."""
    returns = np.array([0.002, 0.0005, -0.0005, -0.002, 0.0])
    labels = create_3class_labels(returns, pos_threshold=0.001, neg_threshold=0.001)
    
    # Expected: [LONG, FLAT, FLAT, SHORT, FLAT]
    expected = np.array([
        LstmClassIndex.LONG,   # 0.002 > 0.001
        LstmClassIndex.FLAT,   # 0.0005 <= 0.001 and >= -0.001
        LstmClassIndex.FLAT,   # -0.0005 <= 0.001 and >= -0.001
        LstmClassIndex.SHORT,  # -0.002 < -0.001
        LstmClassIndex.FLAT,   # 0.0 <= 0.001 and >= -0.001
    ])
    
    np.testing.assert_array_equal(labels, expected)


def test_create_3class_labels_edge_cases():
    """Test edge cases for 3-class label generation."""
    # Exactly at thresholds
    returns = np.array([0.001, -0.001, 0.0011, -0.0011])
    labels = create_3class_labels(returns, pos_threshold=0.001, neg_threshold=0.001)
    
    # Expected: [FLAT, FLAT, LONG, SHORT]
    # Note: > pos_threshold and < -neg_threshold (strict inequalities)
    expected = np.array([
        LstmClassIndex.FLAT,   # 0.001 == 0.001 (not >)
        LstmClassIndex.FLAT,   # -0.001 == -0.001 (not <)
        LstmClassIndex.LONG,   # 0.0011 > 0.001
        LstmClassIndex.SHORT,  # -0.0011 < -0.001
    ])
    
    np.testing.assert_array_equal(labels, expected)


def test_create_3class_labels_custom_thresholds():
    """Test 3-class label generation with custom thresholds."""
    returns = np.array([0.005, 0.001, -0.001, -0.005])
    labels = create_3class_labels(returns, pos_threshold=0.003, neg_threshold=0.003)
    
    # Expected: [LONG, FLAT, FLAT, SHORT]
    expected = np.array([
        LstmClassIndex.LONG,   # 0.005 > 0.003
        LstmClassIndex.FLAT,   # 0.001 <= 0.003 and >= -0.003
        LstmClassIndex.FLAT,   # -0.001 <= 0.003 and >= -0.003
        LstmClassIndex.SHORT,  # -0.005 < -0.003
    ])
    
    np.testing.assert_array_equal(labels, expected)


def test_get_class_name():
    """Test class name retrieval."""
    assert get_class_name(0) == "FLAT"
    assert get_class_name(1) == "LONG"
    assert get_class_name(2) == "SHORT"
    
    with pytest.raises(ValueError):
        get_class_name(3)


def test_class_index_enum():
    """Test LstmClassIndex enum values."""
    assert LstmClassIndex.FLAT == 0
    assert LstmClassIndex.LONG == 1
    assert LstmClassIndex.SHORT == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

