# Adapted and simplified from: https://github.com/koonimaru/radialtree

import numpy as np

import matplotlib as mpl
from  matplotlib import pyplot as plt
import matplotlib.cm as cm

from matplotlib.axes import Axes
from matplotlib.lines import Line2D


colormap_list = [
    "nipy_spectral",
    "terrain",
    "gist_rainbow",
    "CMRmap",
    "coolwarm",
    "gnuplot",
    "gist_stern",
    "brg",
    "rainbow",
]


def radialTreee(
    Z2,
    fontsize=8,
    ax: Axes = None,
    pallete="gist_rainbow",
    addlabels=True,
    sample_classes=None,
    colorlabels=None,
    colorlabels_legend=None,
):

    """
    Drawing a radial dendrogram from a scipy dendrogram output.
    Parameters
    ----------
    Z2 : dictionary
        A dictionary returned by scipy.cluster.hierarchy.dendrogram
    addlabels: bool
        A bool to choose if labels are shown.
    fontsize : float
        A float to specify the font size

    ax : Axes or None:
        Axes in which to draw the plot, otherwise use the currently-active Axes.
    pallete : string or None

        Matlab colormap name.
        If `None` is provided then `color_list` from Z2 is used as is.
    sample_classes : dict
        A dictionary that contains lists of sample subtypes or classes. These classes appear
        as color labels of each leaf. Colormaps are automatically assigned. Not compatible
        with options "colorlabels" and "colorlabels_legend".
        e.g., {"color1":["Class1","Class2","Class1","Class3", ....]}
    colorlabels : dict
        A dictionary to set color labels to leaves. The key is the name of the color label.
        The value is the list of RGB color codes, each corresponds to the color of a leaf.
        e.g., {"color1":[[1,0,0,1], ....]}
    colorlabels_legend : dict
        A nested dictionary to generate the legends of color labels. The key is the name of
        the color label. The value is a dictionary that has two keys "colors" and "labels".
        The value of "colors" is the list of RGB color codes, each corresponds to the class of a leaf.
        e.g., {"color1":{"colors":[[1,0,0,1], ....], "labels":["label1","label2",...]}}

    Returns
    -------
    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """
    if ax is None:
        ax: Axes = plt.gca()
    linewidth = 0.5
    R = 1
    width = R * 0.05
    space = R * 0.01

    if colorlabels != None:
        offset = (
            width * len(colorlabels) / R + space * (len(colorlabels) - 1) / R + 0.011
        )
    elif sample_classes != None:
        offset = (
            width * len(sample_classes) / R
            + space * (len(sample_classes) - 1) / R
            + 0.011
        )
    else:

        offset = 0

    xmax = np.amax(Z2["icoord"])
    xmin = np.amin(Z2["icoord"])
    ymax = np.amax(Z2["dcoord"])
    # print(
    #     f"{xmax=}",
    #     np.amin(Z2["icoord"]),
    #     (xmax - xmin) / (len(Z2["ivl"]) - 1),
    #     (len(Z2["ivl"])),
    # )
    ucolors = sorted(set(Z2["color_list"]))
    # cmap = cm.gist_rainbow(np.linspace(0, 1, len(ucolors)))
    if pallete:
        cmp = cm.get_cmap(pallete, len(ucolors))
        # print(cmp)
        if type(cmp) == mpl.colors.LinearSegmentedColormap:
            cmap = cmp(np.linspace(0, 1, len(ucolors)))
        else:
            cmap = cmp.colors

        def get_color(c):
            return cmap[ucolors.index(c)]

    else:

        def get_color(c):
            return c

    nlabels = 0
    for icoord, dcoord, c in sorted(zip(Z2["icoord"], Z2["dcoord"], Z2["color_list"])):
        # x, y = Z2['icoord'][0], Z2['dcoord'][0]
        _color = get_color(c)
        if c == "C0":  # np.abs(_xr1)<0.000000001 and np.abs(_yr1) <0.000000001:
            _color = "black"

        # transforming original x coordinates into relative circumference positions and y into radius
        # the rightmost leaf is going to [1, 0]
        r = R * (1 - np.array(dcoord) / ymax)
        _x = np.cos(
            2 * np.pi * np.array([icoord[0], icoord[2]]) / xmax
        )  # transforming original x coordinates into x circumference positions
        _xr0 = _x[0] * r[0]
        _xr1 = _x[0] * r[1]
        _xr2 = _x[1] * r[2]
        _xr3 = _x[1] * r[3]
        _y = np.sin(
            2 * np.pi * np.array([icoord[0], icoord[2]]) / xmax
        )  # transforming original x coordinates into y circumference positions
        _yr0 = _y[0] * r[0]
        _yr1 = _y[0] * r[1]
        _yr2 = _y[1] * r[2]
        _yr3 = _y[1] * r[3]
        # ax.scatter([_xr0, _xr1, _xr2, _xr3],[_yr0, _yr1, _yr2,_yr3], c="b")

        # if y[0]>0 and y[3]>0:
        # _color="black"
        # plotting radial lines
        ax.plot([_xr0, _xr1], [_yr0, _yr1], c=_color, linewidth=linewidth)
        ax.plot([_xr2, _xr3], [_yr2, _yr3], c=_color, linewidth=linewidth)

        # plotting circular links between nodes
        if _yr1 > 0 and _yr2 > 0:
            link = np.sqrt(r[1] ** 2 - np.linspace(_xr1, _xr2, 100) ** 2)
            ax.plot(np.linspace(_xr1, _xr2, 100), link, c=_color, linewidth=linewidth)
        elif _yr1 < 0 and _yr2 < 0:
            link = -np.sqrt(r[1] ** 2 - np.linspace(_xr1, _xr2, 100) ** 2)

            ax.plot(np.linspace(_xr1, _xr2, 100), link, c=_color, linewidth=linewidth)
        elif _yr1 > 0 and _yr2 < 0:
            _r = r[1]
            if _xr1 < 0 or _xr2 < 0:
                _r = -_r
            link = np.sqrt(r[1] ** 2 - np.linspace(_xr1, _r, 100) ** 2)
            ax.plot(np.linspace(_xr1, _r, 100), link, c=_color, linewidth=linewidth)
            link = -np.sqrt(r[1] ** 2 - np.linspace(_r, _xr2, 100) ** 2)
            ax.plot(np.linspace(_r, _xr2, 100), link, c=_color, linewidth=linewidth)

    label_coords = []
    # determine the coordiante of the labels and their rotation:
    for i, label in enumerate(Z2["ivl"]):
        # scipy (1.x.x) places the leaves in x = 5+i*10 , and we can use this
        # to calulate where to put the labels
        place = ((i + 1) / len(Z2["ivl"])) * 2
        upside_down = place > 0.5 and place < 1.5
        label_coords.append(
            [
                np.cos(place * np.pi) * (1.01 + offset),  # _x
                np.sin(place * np.pi) * (1.01 + offset),  # _y
                place * 180 + (180 if upside_down else 0),  # _rot,
                upside_down
            ]
        )
    if addlabels == True:
        assert len(Z2["ivl"]) == len(label_coords), (
            f'Internal error, label numbers for Z2 ({len(Z2["ivl"])})'
            f" and for calculated labels ({len(label_coords)}) must be equal!"
        )
        for (_x, _y, _rot, upside_down), label in zip(label_coords, Z2["ivl"]):
            ax.text(
                _x,
                _y,
                label,
                {"va": "center", "ha" : "right" if upside_down else "left"},
                family="monospace",
                rotation_mode="anchor",
                rotation=_rot,
                fontsize=fontsize,
            )

    if colorlabels != None:
        assert len(Z2["ivl"]) == len(label_coords), (
            "Internal error, label numbers "
            + str(len(Z2["ivl"]))
            + " and "
            + str(len(label_coords))
            + " must be equal!"
        )

        j = 0
        outerrad = R * 1.05 + width * len(colorlabels) + space * (len(colorlabels) - 1)

        # sort_index=np.argsort(Z2['icoord'])
        # print(sort_index)
        intervals = []
        for i in range(len(label_coords)):
            _xl, _yl, _rotl, _upside_down = label_coords[i - 1]
            _x, _y, _rot, _upside_down = label_coords[i]
            if i == len(label_coords) - 1:
                _xr, _yr, _rotr, _upside_down = label_coords[0]
            else:
                _xr, _yr, _rotr, _upside_down = label_coords[i + 1]
            d = ((_xr - _xl) ** 2 + (_yr - _yl) ** 2) ** 0.5
            intervals.append(d)
        colorpos = intervals  # np.ones([len(label_coords)])
        labelnames = []
        for labelname, colorlist in colorlabels.items():

            colorlist = np.array(colorlist)[Z2["leaves"]]
            outerrad = outerrad - width * j - space * j
            innerrad = outerrad - width
            patches, texts = ax.pie(
                colorpos,
                colors=colorlist,
                radius=outerrad,
                counterclock=True,
                startangle=label_coords[0][2] * 0.5,
                wedgeprops=dict(
                    width=width,
                    # edgecolor='w', #if this is active the wedges will be more clearly separated
                ),
            )


            labelnames.append(labelname)
            j += 1

        if colorlabels_legend != None:
            for i, labelname in enumerate(labelnames):
                print(colorlabels_legend[labelname]["colors"])
                colorlines = []
                for c in colorlabels_legend[labelname]["colors"]:
                    colorlines.append(Line2D([0], [0], color=c, lw=4))

                leg = ax.legend(
                    colorlines,
                    colorlabels_legend[labelname]["labels"],
                    bbox_to_anchor=(1.5 + 0.3 * i, 1.0),
                    title=labelname,
                )
                ax.add_artist(leg)
    elif sample_classes != None:
        assert len(Z2["ivl"]) == len(label_coords), (
            "Internal error, label numbers "
            + str(len(Z2["ivl"]))
            + " and "
            + str(len(label_coords))
            + " must be equal!"
        )

        j = 0
        outerrad = (
            R * 1.015 + width * len(sample_classes) + space * (len(sample_classes) - 1)
        )

        intervals = []
        for i in range(len(label_coords)):
            _xl, _yl, _rotl, _upside_down = label_coords[i - 1]
            _x, _y, _rot, _upside_down = label_coords[i]
            if i == len(label_coords) - 1:
                _xr, _yr, _rotr, _upside_down = label_coords[0]
            else:
                _xr, _yr, _rotr, _upside_down = label_coords[i + 1]
            d = ((_xr - _xl) ** 2 + (_yr - _yl) ** 2) ** 0.5
            intervals.append(d)
        colorpos = intervals  # np.ones([len(label_coords)])
        labelnames = []
        colorlabels_legend = {}
        for labelname, colorlist in sample_classes.items():

            ucolors = sorted(list(np.unique(colorlist)))
            type_num = len(ucolors)
            _cmp = cm.get_cmap(pallete, type_num + 1)
            _colorlist = [_cmp(ucolors.index(c) + 1) for c in colorlist]
            _colorlist = np.array(_colorlist)[Z2["leaves"]]
            outerrad = outerrad - width * j - space * j
            innerrad = outerrad - width
            patches, texts = ax.pie(
                colorpos,
                colors=_colorlist,
                radius=outerrad,
                counterclock=True,
                startangle=label_coords[0][2] * 0.5,
                wedgeprops=dict(
                    width=width,
                    # edgecolor='w', #if this is active the wedges will be more clearly separated
                ),
            )


            labelnames.append(labelname)
            colorlabels_legend[labelname] = {}
            colorlabels_legend[labelname]["colors"] = _cmp(np.linspace(0, 1, type_num))
            colorlabels_legend[labelname]["labels"] = ucolors
            j += 1

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorlabels != None:
        maxr = R * 1.05 + width * len(colorlabels) + space * (len(colorlabels) - 1)
    elif sample_classes != None:
        maxr = (
            R * 1.05 + width * len(sample_classes) + space * (len(sample_classes) - 1)
        )
    else:
        maxr = R * 1.05
    ax.set_xlim(-maxr, maxr)
    ax.set_ylim(-maxr, maxr)
    return ax
