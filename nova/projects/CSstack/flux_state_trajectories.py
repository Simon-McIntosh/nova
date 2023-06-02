from nep.DINA.read_scenario import scenario_data
from amigo.pyplot import plt
from nep.DINA.extract_fluxstate import fluxmap
from matplotlib.lines import Line2D
import pandas as pd

plt.set_context("talk")
ax = plt.subplots(1, 1)[1]

burn = pd.DataFrame(columns=["dt", "dpsi"])

fmap = fluxmap()
fmap.plot(ax=ax)

d2 = scenario_data(read_txt=False)
strID, dt_min, dt_filt = "15MA DT-DINA", 10, 0
folders = [f for f in d2.folders if strID in f]
for file in folders[-12:]:
    d2.load_file(file)  # read / load single file
    try:
        dt = d2.plot(
            "<PSIcoils>",
            "li(3)",
            xslice=["SOF", "EOB"],
            dt_filt=dt_filt,
            dt_min=dt_min,
            strID=strID,
        )
        if dt > dt_min:
            # color = ax.get_lines()[-1].get_color()
            color = "k"
            d2.plot("<PSIcoils>", "li(3)", xslice="SOF", marker="*", color=color)
            d2.plot("<PSIcoils>", "li(3)", xslice="SOB", marker="P", color=color)
            d2.plot("<PSIcoils>", "li(3)", xslice="EOB", marker="X", color=color)

            # store burn length
            burn.loc[file, ["dt", "dpsi"]] = d2.feature_segments.loc[
                "burn", ["dt", "dpsi"]
            ]
    except KeyError:
        pass

h = ax.get_legend_handles_labels()[0]

h.extend(
    [
        Line2D([0], [0], color="w", marker="*", markerfacecolor="k", label="SOF"),
        Line2D([0], [0], color="w", marker="P", markerfacecolor="k", label="SOB"),
        Line2D([0], [0], color="w", marker="X", markerfacecolor="k", label="EOB"),
    ]
)

ax.legend(
    handles=h[4:], ncol=1, loc="center right", bbox_to_anchor=(1.4, 0.5), frameon=False
)
ax.add_artist(fmap.legend)
