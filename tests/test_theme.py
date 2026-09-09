"""Phase 6b tests: styling is shared, tables are display-safe, captions are complete.

Presentation bugs are quiet by nature — an off-brand PNG or an unrounded float renders
fine and simply looks wrong in the memoria. These tests make the two guarantees the theme
module exists to provide (one palette, one template) checkable rather than assumed.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.evaluation import dashboard_figures, export_figures, theme
from src.evaluation.dashboard_figures import (
    FIGURE_NUMBERS,
    caption,
    comparison_figure,
    day_type_figure,
    demand_figure,
    importance_group_figure,
    importance_top_figure,
    residual_figure,
)
from src.evaluation.report_tables import (
    comparison_table,
    day_type_table,
    feature_importance_table,
    group_importance_table,
    hyperparameters_table,
)
from src.evaluation.theme import (
    FAMILY_COLOR,
    GROUP_COLOR,
    VIU_COLORS,
    VIU_TEMPLATE,
    theme_fingerprint,
)


# --------------------------------------------------------------------------------------
# One theme, shared by both rendering paths
# --------------------------------------------------------------------------------------


def test_interactive_and_static_paths_share_one_template_object():
    """Not merely equal values — the same object, so they cannot drift independently."""
    assert dashboard_figures.VIU_TEMPLATE is theme.VIU_TEMPLATE
    assert export_figures.VIU_TEMPLATE is theme.VIU_TEMPLATE


def test_figure_modules_define_no_colours_of_their_own():
    """Colour constants must live only in theme.py."""
    assert dashboard_figures.FAMILY_COLOR is theme.FAMILY_COLOR
    assert dashboard_figures.GROUP_COLOR is theme.GROUP_COLOR


def test_no_hardcoded_hex_colours_in_figure_builders():
    """A literal hex in a builder would bypass the theme and go unnoticed."""
    import re
    from pathlib import Path

    source = Path(dashboard_figures.__file__).read_text(encoding="utf-8")
    # Strip comments before scanning, so explanatory text mentioning a hex does not trip.
    code = "\n".join(
        line.split("#")[0] for line in source.splitlines()
    )
    assert not re.search(r"#[0-9a-fA-F]{6}\b", code)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: demand_figure("ensemble_equal", "test"),
        lambda: comparison_figure("test"),
        lambda: residual_figure("xgboost_alone", "test"),
        lambda: day_type_figure("test"),
        lambda: importance_group_figure("xgboost_alone"),
        lambda: importance_top_figure("xgboost_alone"),
    ],
)
def test_every_builder_returns_a_themed_figure(builder):
    fig = builder()
    assert isinstance(fig, go.Figure)
    assert fig.layout.template.layout.colorway == theme_fingerprint()


def test_export_theme_check_accepts_a_themed_figure():
    export_figures.assert_theme_applied(comparison_figure("test"), "probe.png")


def test_export_theme_check_rejects_an_unthemed_figure():
    """The guard must be capable of firing."""
    rogue = go.Figure(go.Bar(x=["a"], y=[1]))
    rogue.update_layout(template="plotly_dark")
    with pytest.raises(ValueError, match="does not carry the shared VIU template"):
        export_figures.assert_theme_applied(rogue, "rogue.png")


# --------------------------------------------------------------------------------------
# Palette properties
# --------------------------------------------------------------------------------------


def test_primary_is_the_documented_approximate_orange():
    """Pinned so replacing it with the official VIU hex is a deliberate, visible change."""
    assert VIU_COLORS["primary"] == "#E8590C"


def test_required_palette_roles_exist():
    for role in ("primary", "secondary", "white", "gray", "near_black"):
        assert role in VIU_COLORS
        assert VIU_COLORS[role].startswith("#") and len(VIU_COLORS[role]) == 7


def test_family_colours_are_all_distinct():
    assert len(set(FAMILY_COLOR.values())) == len(FAMILY_COLOR)


def test_group_colours_are_all_distinct():
    assert len(set(GROUP_COLOR.values())) == len(GROUP_COLOR)


def test_family_palette_avoids_the_red_green_cvd_trap():
    """The old palette separated LSTM (red) from ensemble (green) by hue alone.

    Under deuteranopia/protanopia those two were near-identical, and they are exactly the
    pair a reader must distinguish. Assert neither is a saturated red or green.
    """

    def rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    for color in FAMILY_COLOR.values():
        r, g, b = rgb(color)
        is_saturated_green = g > 120 and g > r + 40 and g > b + 40
        assert not is_saturated_green, f"{color} is a saturated green"


def test_family_colours_differ_in_luminance_for_greyscale_printing():
    """A memoria may be printed in black and white; hue alone is then unavailable."""

    def luminance(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    values = sorted(luminance(c) for c in FAMILY_COLOR.values())
    gaps = [b - a for a, b in zip(values, values[1:])]
    assert min(gaps) > 15, f"family luminances too close: {values}"


def test_backgrounds_are_white_for_google_docs_paste():
    assert VIU_TEMPLATE.layout.paper_bgcolor == VIU_COLORS["white"]
    assert VIU_TEMPLATE.layout.plot_bgcolor == VIU_COLORS["white"]


def test_template_uses_a_system_font_stack():
    """Proprietary VIU typefaces are not assumed installed — see theme.py."""
    family = VIU_TEMPLATE.layout.font.family
    assert "sans-serif" in family
    assert "Periodico" not in family and "Visuelt" not in family


def test_gridlines_are_light_neutral_not_default_plotly_gray():
    assert VIU_TEMPLATE.layout.xaxis.gridcolor == VIU_COLORS["gray_pale"]
    assert VIU_TEMPLATE.layout.yaxis.gridcolor == VIU_COLORS["gray_pale"]


# --------------------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------------------


def test_caption_exists_and_is_numbered_for_every_figure_type():
    kwargs = {"model": "ensemble_equal", "split": "test", "top_n": 15}
    for fig_id, number in FIGURE_NUMBERS.items():
        text = caption(fig_id, **kwargs)
        assert text.startswith(f"Figura {number}. ")
        assert len(text) > len(f"Figura {number}. ") + 20


def test_caption_numbers_are_unique_and_contiguous():
    numbers = sorted(FIGURE_NUMBERS.values())
    assert numbers == list(range(1, len(numbers) + 1))


def test_caption_leaves_no_unfilled_placeholders():
    for fig_id in FIGURE_NUMBERS:
        text = caption(fig_id, model="hybrid", split="val")
        assert "{" not in text and "}" not in text


def test_caption_reuses_notebook_wording():
    """Captions must echo the notebook prose, not introduce new phrasing."""
    assert "Serie observada y predicha superpuestas" in caption(
        "demand", model="hybrid", split="test"
    )
    assert "ordenadas de mejor a peor por MAE" in caption("comparison", split="test")
    assert "son indicativos, no concluyentes" in caption("day_type", split="test")


def test_caption_rejects_unknown_figure_id():
    with pytest.raises(ValueError, match="unknown fig_id"):
        caption("scatter_matrix")


def test_caption_reports_a_missing_placeholder_argument():
    with pytest.raises(ValueError, match="needs"):
        caption("demand")


# --------------------------------------------------------------------------------------
# Report tables are display-safe
# --------------------------------------------------------------------------------------


def _has_no_long_floats(df: pd.DataFrame, max_decimals: int = 4) -> bool:
    """No float column may carry more precision than the documented rounding."""
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            rounded = df[col].round(max_decimals)
            if not np.allclose(df[col].to_numpy(), rounded.to_numpy(), equal_nan=True):
                return False
    return True


@pytest.mark.parametrize("split", ["val", "test"])
def test_comparison_table_is_rounded_for_display(split):
    table = comparison_table(split)
    assert _has_no_long_floats(table)
    assert pd.api.types.is_integer_dtype(table["MAE"])
    assert pd.api.types.is_integer_dtype(table["RMSE"])


def test_comparison_table_has_spanish_headers():
    table = comparison_table("test")
    assert list(table.columns) == ["Modelo", "Familia", "n", "MAE", "RMSE", "MAPE (%)", "R²"]


def test_comparison_table_is_sorted_best_first():
    table = comparison_table("test")
    assert table["MAE"].is_monotonic_increasing
    assert table.iloc[0]["Modelo"].startswith("ensemble")


def test_comparison_table_matches_the_underlying_metrics():
    """Rounding must not change which model wins, nor the reported value."""
    from src.evaluation.dashboard_data import model_comparison_table

    raw = model_comparison_table("test").set_index("model")
    table = comparison_table("test").set_index("Modelo")
    for model in table.index:
        assert table.loc[model, "MAE"] == round(raw.loc[model, "MAE"])


def test_comparison_table_returns_a_plain_dataframe_not_a_styler():
    """A Styler pastes into Google Docs as an image; a DataFrame pastes as a real table."""
    assert isinstance(comparison_table("test"), pd.DataFrame)
    assert not hasattr(comparison_table("test"), "to_html_styled")


def test_r2_default_is_four_decimals():
    """Four, not two: at two the three leading models collapse to 0.98 / 0.97 / 0.97."""
    from src.evaluation.report_tables import R2_DECIMALS

    assert R2_DECIMALS == 4

    table = comparison_table("test").set_index("Modelo")
    assert table.loc["ensemble_equal", "R²"] == pytest.approx(0.9755)
    assert table.loc["xgboost_alone", "R²"] == pytest.approx(0.9690)
    assert table.loc["sarimax", "R²"] == pytest.approx(0.9672)


def test_default_r2_separates_the_three_leading_models():
    """The point of the finer default: these three must not share an R² value."""
    table = comparison_table("test").set_index("Modelo")
    leaders = [table.loc[m, "R²"] for m in ("ensemble_equal", "xgboost_alone", "sarimax")]
    assert len(set(leaders)) == 3


def test_r2_decimals_remains_configurable_for_a_coarser_table():
    default = comparison_table("test")
    coarse = comparison_table("test", r2_decimals=2)
    assert default["R²"].nunique() > coarse["R²"].nunique()


def test_mape_stays_at_two_decimals():
    """MAPE is unchanged: at this project's scale two decimals already separate cleanly."""
    from src.evaluation.report_tables import MAPE_DECIMALS

    assert MAPE_DECIMALS == 2
    table = comparison_table("test").set_index("Modelo")
    leaders = [table.loc[m, "MAPE (%)"] for m in ("ensemble_equal", "xgboost_alone", "sarimax")]
    assert len(set(leaders)) == 3


def test_day_type_table_is_display_safe():
    table = day_type_table("test")
    assert _has_no_long_floats(table)
    assert "Tipo de día" in table.columns
    assert "n" in table.columns
    assert table["n"].min() >= 1


def test_day_type_table_keeps_small_sample_counts_visible():
    """The festivo/puente caveat must travel with the table."""
    table = day_type_table("test").set_index("Tipo de día")
    assert table.loc["Festivo", "n"] == 5
    assert table.loc["Laborable, puente", "n"] == 3


def test_hyperparameters_table_matches_the_saved_artifacts():
    table = hyperparameters_table().set_index("Modelo")
    assert table.loc["xgboost_residual", "max_depth"] == 5
    assert table.loc["xgboost_residual", "n_estimators"] == 300
    assert table.loc["xgboost_alone", "n_estimators"] == 100
    assert table.loc["xgboost_residual", "Filas entrenamiento"] == 552
    assert table.loc["xgboost_alone", "Filas entrenamiento"] == 889


@pytest.mark.parametrize("model", ["xgboost_residual", "xgboost_alone"])
def test_feature_importance_table_is_display_safe(model):
    table = feature_importance_table(model)
    assert _has_no_long_floats(table)
    assert list(table.columns) == ["#", "Variable", "Grupo", "Ganancia"]
    assert len(table) == 15


@pytest.mark.parametrize("model", ["xgboost_residual", "xgboost_alone"])
def test_group_importance_table_is_display_safe(model):
    table = group_importance_table(model)
    assert _has_no_long_floats(table)
    assert table["Peso (%)"].sum() == pytest.approx(100.0, abs=0.1)
