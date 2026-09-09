"""VIU brand-approximate visual theme — single source of truth for all figure styling.

Imported by BOTH `dashboard_figures.py` (interactive) and `export_figures.py` (static PNG).
Neither defines a colour of its own. This mirrors the existing principle that both paths
already share the same figure builders: a second palette definition somewhere would let the
notebook and the memoria's embedded PNGs drift apart without anything failing.

────────────────────────────────────────────────────────────────────────────────────────
⚠ THE ORANGE IS AN APPROXIMATION, NOT THE OFFICIAL VIU BRAND COLOUR.

`VIU_COLORS["primary"]` below is a visual approximation. The exact Pantone/HEX from VIU's
brand manual was not available when this was written, so a plausible institutional orange
was chosen. **It is the single line to edit** if the official value is supplied — every
figure, both interactive and exported, picks the change up automatically. Nothing else in
the codebase hard-codes an orange.
────────────────────────────────────────────────────────────────────────────────────────

TYPOGRAPHY. VIU's actual corporate typefaces (Periodico Display for display text, Visuelt
Pro for body) are proprietary and are NOT assumed to be installed on any machine that runs
this project — including the kaleido/Chrome process that renders the PNGs, where a missing
font silently falls back to something arbitrary and makes exported figures differ from the
on-screen ones. A system sans-serif stack is used instead, which renders consistently
everywhere.

BACKGROUNDS ARE WHITE, DELIBERATELY. These figures are destined for a Google Docs results
chapter. A dark or transparent-with-dark-text figure pastes as a broken-looking block on a
white page, so both the plot area and the paper are explicitly white rather than left to
the renderer's default.
"""

import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------

VIU_COLORS: dict[str, str] = {
    # ⚠ APPROXIMATION — replace with the official VIU hex when available. See module note.
    "primary": "#E8590C",
    # Documented secondary / digital accent. Deliberately a DARK muted purple: at the
    # lighter #6B4E71 its relative luminance (86.7) was within 0.2 of "steel" (86.6), so
    # the single-stage and LSTM families were indistinguishable in a greyscale print even
    # though they read as clearly different hues on screen. Caught by
    # test_family_colours_differ_in_luminance_for_greyscale_printing.
    "secondary": "#4A3A52",
    "white": "#FFFFFF",
    "gray": "#8A8F98",
    "gray_light": "#D9DCE1",
    "gray_pale": "#EDEFF2",
    "near_black": "#1F2328",
    # Support tones, used only where a fifth/sixth distinguishable series is needed.
    "steel": "#3D5A80",
    "sand": "#C9A227",
    "teal": "#2E6E6B",
}

# Model-family colours, replacing the earlier default-Plotly assignment.
#
# ACCESSIBILITY: the previous palette distinguished the LSTM family (#d62728 red) from the
# ensemble family (#2ca02c green) by hue alone — the single worst pair for deuteranopia and
# protanopia, which together affect roughly 8% of men. Under red-green CVD those two
# families were near-identical, and they are exactly the two the results chapter needs a
# reader to tell apart. The replacement separates families along orange / purple / blue /
# grey, which stay distinct under both common CVD forms and also survive greyscale printing
# because their luminances differ. The VIU orange is assigned to the ensemble family, which
# is both on-brand and narratively convenient: the best model is the one that stands out.
FAMILY_COLOR: dict[str, str] = {
    "naive baseline": VIU_COLORS["gray_light"],
    "single-stage": VIU_COLORS["steel"],
    "LSTM family": VIU_COLORS["secondary"],
    "ensemble": VIU_COLORS["primary"],
}

# Feature-group colours for the importance charts. Same reasoning: no red/green pairing.
GROUP_COLOR: dict[str, str] = {
    "calendar": VIU_COLORS["primary"],
    "lag_total": VIU_COLORS["steel"],
    "lag_operator": VIU_COLORS["teal"],
    "rolling": VIU_COLORS["secondary"],
    "weather": VIU_COLORS["sand"],
    "fourier_weekly": VIU_COLORS["gray"],
    "fourier_annual": VIU_COLORS["near_black"],
}

# Ordered sequence for any chart that needs categorical colours without a family mapping.
VIU_COLORWAY: list[str] = [
    VIU_COLORS["primary"],
    VIU_COLORS["steel"],
    VIU_COLORS["secondary"],
    VIU_COLORS["gray"],
    VIU_COLORS["teal"],
    VIU_COLORS["sand"],
    VIU_COLORS["near_black"],
    VIU_COLORS["gray_light"],
]

# --------------------------------------------------------------------------------------
# Typography and template
# --------------------------------------------------------------------------------------

# System stack: renders identically in the browser and in the headless Chrome that kaleido
# drives. See the typography note in the module docstring.
FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    'Arial, sans-serif'
)

# Sized for legibility both on screen and in a printed A4 memoria at scale=2.
FONT_SIZE_BASE = 13
FONT_SIZE_TITLE = 18
FONT_SIZE_AXIS_TITLE = 13
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12

TEMPLATE_NAME = "viu"

VIU_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(
            family=FONT_FAMILY, size=FONT_SIZE_BASE, color=VIU_COLORS["near_black"]
        ),
        title=dict(
            font=dict(
                family=FONT_FAMILY,
                size=FONT_SIZE_TITLE,
                color=VIU_COLORS["near_black"],
            ),
            x=0.0,
            xanchor="left",
        ),
        # Both explicitly white: see the background note in the module docstring.
        paper_bgcolor=VIU_COLORS["white"],
        plot_bgcolor=VIU_COLORS["white"],
        colorway=VIU_COLORWAY,
        xaxis=dict(
            # Light neutral gridlines rather than Plotly's default mid-gray, which competes
            # with the data at print resolution.
            gridcolor=VIU_COLORS["gray_pale"],
            linecolor=VIU_COLORS["gray_light"],
            zerolinecolor=VIU_COLORS["gray_light"],
            title=dict(font=dict(size=FONT_SIZE_AXIS_TITLE)),
            tickfont=dict(size=FONT_SIZE_TICK),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=VIU_COLORS["gray_pale"],
            linecolor=VIU_COLORS["gray_light"],
            zerolinecolor=VIU_COLORS["gray_light"],
            title=dict(font=dict(size=FONT_SIZE_AXIS_TITLE)),
            tickfont=dict(size=FONT_SIZE_TICK),
            automargin=True,
        ),
        legend=dict(font=dict(size=FONT_SIZE_LEGEND), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=90, b=70, l=80, r=60),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE)),
    )
)


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the VIU template to a figure. One call per builder; never restyles data.

    Returns the same figure for convenient chaining.
    """
    fig.update_layout(template=VIU_TEMPLATE)
    return fig


def theme_fingerprint() -> tuple[str, ...]:
    """Stable identifier of the active styling, for cross-module equality checks.

    `export_figures.py` compares this against what the builders produced, so a future
    second palette definition fails a check instead of silently making the memoria's PNGs
    look different from the notebook.
    """
    return tuple(VIU_TEMPLATE.layout.colorway)
