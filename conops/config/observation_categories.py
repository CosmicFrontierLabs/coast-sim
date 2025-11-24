"""Configuration for observation categories based on target ID ranges."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ObservationCategory(BaseModel):
    """Configuration for a single observation category.

    Attributes
    ----------
    name : str
        Name of the category (e.g., "GO", "Survey", "GRB").
    obsid_min : int
        Minimum obsid value (inclusive) for this category.
    obsid_max : int
        Maximum obsid value (exclusive) for this category.
    color : str
        Matplotlib color specification for visualization.
    """

    name: str
    obsid_min: int
    obsid_max: int
    color: str = "tab:blue"


class ObservationCategories(BaseModel):
    """Configuration for categorizing observations by target ID ranges.

    This class defines how target IDs (obsid) map to observation categories
    for visualization and analysis. Categories are matched in order, so more
    specific ranges should be defined first.

    Attributes
    ----------
    categories : list[ObservationCategory]
        List of observation categories with their ID ranges and colors.
    default_name : str
        Default category name for obsids that don't match any range.
    default_color : str
        Default color for obsids that don't match any range.

    Examples
    --------
    >>> categories = ObservationCategories.default_categories()
    >>> category = categories.get_category(15000)
    >>> print(category.name)  # "Survey"

    >>> # Custom configuration
    >>> custom = ObservationCategories(
    ...     categories=[
    ...         ObservationCategory(name="Science", obsid_min=10000, obsid_max=20000, color="blue"),
    ...         ObservationCategory(name="Calibration", obsid_min=90000, obsid_max=91000, color="gray"),
    ...     ]
    ... )
    """

    categories: list[ObservationCategory] = Field(default_factory=list)
    default_name: str = "Observation"
    default_color: str = "tab:blue"

    def get_category(self, obsid: int) -> ObservationCategory:
        """Get the category for a given observation ID.

        Parameters
        ----------
        obsid : int
            Target observation ID.

        Returns
        -------
        ObservationCategory
            The matching category, or a default category if no match found.
        """
        for category in self.categories:
            if category.obsid_min <= obsid < category.obsid_max:
                return category

        # Return default category
        return ObservationCategory(
            name=self.default_name,
            obsid_min=0,
            obsid_max=10000000,
            color=self.default_color,
        )

    def get_all_category_names(self) -> list[str]:
        """Get list of all defined category names including default."""
        names = [cat.name for cat in self.categories]
        if self.default_name not in names:
            names.append(self.default_name)
        return names

    def get_category_color(self, category_name: str) -> str:
        """Get the color for a given category name.

        Parameters
        ----------
        category_name : str
            Name of the category.

        Returns
        -------
        str
            Matplotlib color specification.
        """
        for category in self.categories:
            if category.name == category_name:
                return category.color
        return self.default_color

    @classmethod
    def default_categories(cls) -> ObservationCategories:
        """Create default observation categories.

        Returns
        -------
        ObservationCategories
            Configuration with default categories:
            - GO programs: 20000-30000 (blue)
            - Survey: 10000-20000 (green)
            - GRB: 1000000-2000000 (orange)
            - TOO: 30000-40000 (red)
            - Charging: 999000-1000000 (gold)
            - Default: Other (purple)
        """
        return cls(
            categories=[
                ObservationCategory(
                    name="GO", obsid_min=20000, obsid_max=30000, color="tab:blue"
                ),
                ObservationCategory(
                    name="Survey", obsid_min=10000, obsid_max=20000, color="tab:green"
                ),
                ObservationCategory(
                    name="GRB", obsid_min=1000000, obsid_max=2000000, color="tab:orange"
                ),
                ObservationCategory(
                    name="TOO", obsid_min=30000, obsid_max=40000, color="tab:red"
                ),
                ObservationCategory(
                    name="Charging", obsid_min=999000, obsid_max=1000000, color="gold"
                ),
            ],
            default_name="Other",
            default_color="tab:purple",
        )
