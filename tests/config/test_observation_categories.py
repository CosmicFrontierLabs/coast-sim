"""Unit tests for conops.config.observation_categories module."""

from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)


class TestObservationCategory:
    """Test ObservationCategory class."""

    def test_initialization_basic(self):
        """Test basic initialization of ObservationCategory."""
        category = ObservationCategory(
            name="Test Category", obsid_min=1000, obsid_max=2000, color="red"
        )
        assert category.name == "Test Category"
        assert category.obsid_min == 1000
        assert category.obsid_max == 2000
        assert category.color == "red"

    def test_initialization_default_color(self):
        """Test initialization with default color."""
        category = ObservationCategory(
            name="Test Category", obsid_min=1000, obsid_max=2000
        )
        assert category.name == "Test Category"
        assert category.obsid_min == 1000
        assert category.obsid_max == 2000
        assert category.color == "tab:blue"

    def test_initialization_custom_color(self):
        """Test initialization with custom color."""
        category = ObservationCategory(
            name="Test Category", obsid_min=1000, obsid_max=2000, color="green"
        )
        assert category.color == "green"


class TestObservationCategories:
    """Test ObservationCategories class."""

    def test_initialization_empty(self):
        """Test initialization with empty categories."""
        categories = ObservationCategories()
        assert categories.categories == []
        assert categories.default_name == "Observation"
        assert categories.default_color == "tab:blue"

    def test_initialization_custom_defaults(self):
        """Test initialization with custom defaults."""
        categories = ObservationCategories(
            default_name="Custom Default", default_color="purple"
        )
        assert categories.categories == []
        assert categories.default_name == "Custom Default"
        assert categories.default_color == "purple"

    def test_initialization_with_categories(self):
        """Test initialization with categories."""
        cat1 = ObservationCategory(name="Cat1", obsid_min=1000, obsid_max=2000)
        cat2 = ObservationCategory(name="Cat2", obsid_min=2000, obsid_max=3000)
        categories = ObservationCategories(categories=[cat1, cat2])
        assert len(categories.categories) == 2
        assert categories.categories[0].name == "Cat1"
        assert categories.categories[1].name == "Cat2"

    def test_get_category_matching_first_category(self):
        """Test get_category returns first matching category."""
        cat1 = ObservationCategory(
            name="Cat1", obsid_min=1000, obsid_max=2000, color="red"
        )
        cat2 = ObservationCategory(
            name="Cat2", obsid_min=2000, obsid_max=3000, color="blue"
        )
        categories = ObservationCategories(categories=[cat1, cat2])

        result = categories.get_category(1500)
        assert result.name == "Cat1"
        assert result.obsid_min == 1000
        assert result.obsid_max == 2000
        assert result.color == "red"

    def test_get_category_matching_second_category(self):
        """Test get_category returns second matching category."""
        cat1 = ObservationCategory(
            name="Cat1", obsid_min=1000, obsid_max=2000, color="red"
        )
        cat2 = ObservationCategory(
            name="Cat2", obsid_min=2000, obsid_max=3000, color="blue"
        )
        categories = ObservationCategories(categories=[cat1, cat2])

        result = categories.get_category(2500)
        assert result.name == "Cat2"
        assert result.obsid_min == 2000
        assert result.obsid_max == 3000
        assert result.color == "blue"

    def test_get_category_no_match_returns_default(self):
        """Test get_category returns default category when no match."""
        cat1 = ObservationCategory(
            name="Cat1", obsid_min=1000, obsid_max=2000, color="red"
        )
        categories = ObservationCategories(
            categories=[cat1], default_name="Default", default_color="green"
        )

        result = categories.get_category(5000)
        assert result.name == "Default"
        assert result.obsid_min == 0
        assert result.obsid_max == 10000000
        assert result.color == "green"

    def test_get_category_boundary_min_inclusive(self):
        """Test get_category includes minimum boundary."""
        cat = ObservationCategory(name="Cat", obsid_min=1000, obsid_max=2000)
        categories = ObservationCategories(categories=[cat])

        result = categories.get_category(1000)
        assert result.name == "Cat"

    def test_get_category_boundary_max_exclusive(self):
        """Test get_category excludes maximum boundary."""
        cat = ObservationCategory(name="Cat", obsid_min=1000, obsid_max=2000)
        categories = ObservationCategories(categories=[cat])

        result = categories.get_category(2000)
        assert result.name == "Observation"  # default

    def test_get_all_category_names_includes_categories_and_default(self):
        """Test get_all_category_names includes all category names."""
        cat1 = ObservationCategory(name="Cat1", obsid_min=1000, obsid_max=2000)
        cat2 = ObservationCategory(name="Cat2", obsid_min=2000, obsid_max=3000)
        categories = ObservationCategories(
            categories=[cat1, cat2], default_name="Default"
        )

        names = categories.get_all_category_names()
        assert set(names) == {"Cat1", "Cat2", "Default"}

    def test_get_all_category_names_default_already_in_categories(self):
        """Test get_all_category_names when default name is already in categories."""
        cat1 = ObservationCategory(name="Cat1", obsid_min=1000, obsid_max=2000)
        cat2 = ObservationCategory(name="Default", obsid_min=2000, obsid_max=3000)
        categories = ObservationCategories(
            categories=[cat1, cat2], default_name="Default"
        )

        names = categories.get_all_category_names()
        assert set(names) == {"Cat1", "Default"}

    def test_get_category_color_existing_category(self):
        """Test get_category_color returns color for existing category."""
        cat1 = ObservationCategory(
            name="Cat1", obsid_min=1000, obsid_max=2000, color="red"
        )
        cat2 = ObservationCategory(
            name="Cat2", obsid_min=2000, obsid_max=3000, color="blue"
        )
        categories = ObservationCategories(categories=[cat1, cat2])

        assert categories.get_category_color("Cat1") == "red"
        assert categories.get_category_color("Cat2") == "blue"

    def test_get_category_color_nonexistent_category(self):
        """Test get_category_color returns default color for nonexistent category."""
        cat = ObservationCategory(
            name="Cat1", obsid_min=1000, obsid_max=2000, color="red"
        )
        categories = ObservationCategories(categories=[cat], default_color="green")

        assert categories.get_category_color("Nonexistent") == "green"

    def test_default_categories_structure(self):
        """Test default_categories creates expected structure."""
        categories = ObservationCategories.default_categories()

        assert len(categories.categories) == 5
        assert categories.default_name == "Other"
        assert categories.default_color == "tab:purple"

        # Check specific categories
        category_names = [cat.name for cat in categories.categories]
        assert "GO" in category_names
        assert "Survey" in category_names
        assert "GRB" in category_names
        assert "TOO" in category_names
        assert "Charging" in category_names

    def test_default_categories_go_range(self):
        """Test default GO category range."""
        categories = ObservationCategories.default_categories()
        go_cat = next(cat for cat in categories.categories if cat.name == "GO")
        assert go_cat.obsid_min == 20000
        assert go_cat.obsid_max == 30000
        assert go_cat.color == "tab:blue"

    def test_default_categories_survey_range(self):
        """Test default Survey category range."""
        categories = ObservationCategories.default_categories()
        survey_cat = next(cat for cat in categories.categories if cat.name == "Survey")
        assert survey_cat.obsid_min == 10000
        assert survey_cat.obsid_max == 20000
        assert survey_cat.color == "tab:green"

    def test_default_categories_grb_range(self):
        """Test default GRB category range."""
        categories = ObservationCategories.default_categories()
        grb_cat = next(cat for cat in categories.categories if cat.name == "GRB")
        assert grb_cat.obsid_min == 1000000
        assert grb_cat.obsid_max == 2000000
        assert grb_cat.color == "tab:orange"

    def test_default_categories_too_range(self):
        """Test default TOO category range."""
        categories = ObservationCategories.default_categories()
        too_cat = next(cat for cat in categories.categories if cat.name == "TOO")
        assert too_cat.obsid_min == 30000
        assert too_cat.obsid_max == 40000
        assert too_cat.color == "tab:red"

    def test_default_categories_charging_range(self):
        """Test default Charging category range."""
        categories = ObservationCategories.default_categories()
        charging_cat = next(
            cat for cat in categories.categories if cat.name == "Charging"
        )
        assert charging_cat.obsid_min == 999000
        assert charging_cat.obsid_max == 1000000
        assert charging_cat.color == "gold"

    def test_default_categories_obsid_matching(self):
        """Test obsid matching with default categories."""
        categories = ObservationCategories.default_categories()

        # Test various obsids
        assert categories.get_category(15000).name == "Survey"
        assert categories.get_category(25000).name == "GO"
        assert categories.get_category(35000).name == "TOO"
        assert categories.get_category(1500000).name == "GRB"
        assert categories.get_category(999500).name == "Charging"
        assert categories.get_category(5000).name == "Other"  # default
        assert categories.get_category(50000).name == "Other"  # default
