"""
Color management utilities for visualizations.

Handles color mapping for categories and provides consistent styling.
"""

from typing import Dict
from src.core.config import Config


def get_color_map(config: Config) -> Dict[str, str]:
    """
    Get color mapping for categories.

    Returns:
        Dictionary mapping category names to hex color codes
    """
    return config.colors


def get_category_color(category: str, config: Config) -> str:
    """
    Get color for specific category.

    Args:
        category: Category name ('animals', 'music', or 'food')
        config: Configuration object

    Returns:
        Hex color code

    Raises:
        KeyError: If category not found
    """
    return config.colors[category]


def category_to_label_color(label: int, categories: tuple, config: Config) -> str:
    """
    Convert numeric label to category color.

    Args:
        label: Numeric label (0, 1, or 2)
        categories: Tuple of category names
        config: Configuration object

    Returns:
        Hex color code for the category
    """
    category = categories[label]
    return config.colors[category]
