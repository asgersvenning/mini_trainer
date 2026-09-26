"""Shared SVG and PNG export for release reports."""


def save_figure(fig, output, name, *, dpi=160):
    """Save a figure without SVG timestamps or trailing whitespace, then close it."""
    import matplotlib.pyplot as plt

    svg = output / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output / f"{name}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
