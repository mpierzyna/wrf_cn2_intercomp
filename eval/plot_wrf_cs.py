import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams.update({
    # "font.family": "serif",
    "font.size": 8,
    "text.usetex": True,

    "figure.dpi": 300,
    # "figure.labelsize": 8,

    "lines.linewidth": .75,
    "hatch.linewidth": .5,
})

# Scale figures because otherwise very small
FIG_WIDTH = 7.25  # in
FIG_WIDTH *= 1.25

ds = xr.open_dataset("../data/WRF_ens_cs/2022-04-18_22/WRF_base_000_wrfout_10min_d03_2022-04-17_12:00:00.nc")

z = ds["z"].isel(Time=0)
hgt = ds["HGT"].isel(Time=0)

fig, ax = plt.subplots(figsize=(FIG_WIDTH / 3, FIG_WIDTH / 3 / (4 / 3)))

ax.fill_between(range(len(hgt)), hgt, color="k")
for i in range(10):
    ax.plot(z[i, :], label=f"lev {i}", color="lightgrey")

# Set up x axis
ax.set_xlim(0, 9)
ax.set_xticks(np.linspace(0, 9, 11))
ax.set_xticklabels(np.arange(0, 11, 1))
ax.set_xlabel("Path, km")

# Set up y axis
exp = 2
ax.set_yscale("function", functions=(lambda x: np.power(x, 1 / exp), lambda x: np.power(x, exp)))
ax.set_yticks([0, 2, 10, 20, 40, 80, 140, 200])
ax.set_ylim(0)
ax.set_ylabel("z, m")

# Plot weighing function
ax_w = ax.twinx()
ds_w_las = pd.read_csv("./w_las.csv")
ds_w_las["x"] *= 9
ax_w.plot(ds_w_las["x"], ds_w_las["y"], color="r", label="w_las")
ax_w.set_ylim(0)
# ax_w.set_ylabel("LAS weighting function")
ax_w.set_yticks([])

fig.subplots_adjust(left=.15, right=0.95, top=.98, bottom=.15)

fig.show()
fig.savefig("plots/wrf_cs.pdf")
