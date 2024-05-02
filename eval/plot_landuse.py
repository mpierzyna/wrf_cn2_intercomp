from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.colors import ListedColormap
import pathlib

from utils import hour_formatter

print(f"This is matplotlib version {plt.matplotlib.__version__}")

# Matplotlib defaults
plt.rcParams.update({
    # "font.family": "serif",
    "font.size": 8,
    "text.usetex": True,

    "figure.dpi": 300,
    # "figure.labelsize": 8,

    "lines.linewidth": .75,
    "hatch.linewidth": .5,
})
sns.set_palette("colorblind")

# Scale figures because otherwise very small
FIG_WIDTH = 7.25  # in
FIG_WIDTH *= 1.25


def round_timestamp(ds: xr.Dataset) -> xr.Dataset:
    """Round the timestamps of the dataset to the nearest minute."""
    return ds.assign_coords(time=ds.time.dt.round("1min"))


def get_modis_lu_cmap(subset: List[int] = None) -> (ListedColormap, List[str]):
    # Land use classes
    landuse_classes = [
        "Evergreen Needleleaf Forest",
        "Evergreen Broadleaf Forest",
        "Deciduous Needleleaf Forest",
        "Deciduous Broadleaf Forest",
        "Mixed Forests",
        "Closed Shrublands",
        "Open Shrublands",
        "Woody Savannas",
        "Savannas",
        "Grasslands",
        "Permanent Wetland",
        "Croplands",
        "Urban and Built-Up",
        "Cropland/Natural Vegetation Mosaic",
        "Snow and Ice",
        "Barren or Sparsely Vegetated",
        "Ocean",
        "Wooded Tundra",
        "Mixed Tundra",
        "Bare Ground Tundra",
        "Lakes"
    ]
    if subset is not None:
        landuse_classes = [landuse_classes[i - 1] for i in subset]

    # Suggested colors for each land use category
    colors = [
        "#1f631d",  # Evergreen Needleleaf Forest - Dark Green
        "#2a924a",  # Evergreen Broadleaf Forest - Green
        "#7fbf7b",  # Deciduous Needleleaf Forest - Light Green
        "#bfd78f",  # Deciduous Broadleaf Forest - Pale Green
        "#83b567",  # Mixed Forests - Olive
        "#c5e6a6",  # Closed Shrublands - Light Olive
        "#f0f0a5",  # Open Shrublands - Light Yellow
        "#e2c889",  # Woody Savannas - Tan
        "#d4b66e",  # Savannas - Light Brown
        "#ba965d",  # Grasslands - Brown
        "#6d86c9",  # Permanent Wetland - Blue
        "#ffeb3b",  # Croplands - Yellow
        "#e74c3c",  # Urban and Built-Up - Red
        "#f39c12",  # Cropland/Natural Vegetation Mosaic - Orange
        "#ffffff",  # Snow and Ice - White
        "#dcdcdc",  # Barren or Sparsely Vegetated - Light Gray
        "#3498db",  # Ocean - Blue
        "#b3b3b3",  # Wooded Tundra - Gray
        "#c0e4f7",  # Mixed Tundra - Light Blue
        "#d7dbdd",  # Bare Ground Tundra - Light Gray
        "#3498db"  # Lakes - Blue (same as Ocean)
    ]
    if subset is not None:
        colors = [colors[i - 1] for i in subset]

    # Create a colormap with the specified colors
    cmap = ListedColormap(colors)
    return cmap, landuse_classes


def plot_modis_lu_cmap(*, horizontal: bool, subset: List[str] = None, **subplots_kwargs) -> plt.Figure:
    cmap, landuse_classes = get_modis_lu_cmap(subset=subset)

    x = np.arange(len(landuse_classes)) + 1
    x = x.reshape(1, -1)

    if horizontal:
        fig, ax = plt.subplots(**subplots_kwargs)

        ax.imshow(x, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(landuse_classes)))
        ax.set_xticklabels([f"[{i + 1}] {lc}" for i, lc in enumerate(landuse_classes)], rotation=45, ha="right")
        ax.set_yticks([])
    else:
        subplots_kwargs = {
            "figsize": (2.5, 4),
            **subplots_kwargs
        }
        fig, ax = plt.subplots(**subplots_kwargs)

        ax.imshow(x.T, cmap=cmap, aspect="auto")

        ax.set_yticks(x.ravel() - 1)
        ax.set_yticklabels(landuse_classes)
        ax.yaxis.tick_right()
        ax.set_xticks([])

        fig.subplots_adjust(left=0.01, right=0.1, top=.99, bottom=0.01)

    return fig


if __name__ == "__main__":
    # Load landuse data
    WE_SLICE = slice(78, 90)
    SN_SLICE = slice(45, 52)

    ds_lu = xr.open_dataset("../data/WRF_ens_cs/lu_hfx_ust.nc").sel(west_east=WE_SLICE, south_north=SN_SLICE)
    ds_lu = ds_lu.isel(Time=slice(6 * 12, None))  # Skip first 12 hours because of warmup
    ds_lu["hfx"] = ds_lu["hfx"] / 1.216e3  # (W/m^2) to (Km/s)

    # Load matching observations
    CESAR_ROOT = pathlib.Path("../data/CESAR")
    ds_meteo_flux = round_timestamp(xr.open_mfdataset(
        (CESAR_ROOT / "Meteo_Flux").glob("cesar_surface_flux_*.nc")  # noqa
    ))
    ds_meteo_flux = ds_meteo_flux.sel(time=slice(ds_lu["Time"].min(), ds_lu["Time"].max()))
    ds_meteo_flux["HSON"] = ds_meteo_flux["HSON"] / 1.216e3  # (W/m^2) to (Km/s)

    # Define path and locations to plot HFX and UST
    loc_bad = (80, 46)  # we, sn
    loc_good = (80, 47)  # we, sn
    loc_cabauw = (80.1, 45.75)
    we_path = np.array([80.00317314, 80.95291482, 81.90254093, 82.85205142, 83.80144627,
                        84.75072545, 85.69988893, 86.64893666, 87.59786863, 88.54668481])
    sn_path = np.array([47.00757939, 47.33140714, 47.65540619, 47.9795765, 48.30391804,
                        48.62843078, 48.95311469, 49.27796975, 49.60299591, 49.92819316])

    # %% Plot colorbar
    fig_cmap = plot_modis_lu_cmap(horizontal=False, subset=np.unique(ds_lu["lu"]).astype(int), figsize=(2.5, 1.5))
    fig_cmap.savefig("plots_new/landuse_cmap_vertical.pdf")

    # %% Plot landuse
    fig: plt.Figure
    fig, (ax_lu, ax_hfx, ax_ust) = plt.subplots(
        nrows=1, ncols=3, figsize=(FIG_WIDTH, FIG_WIDTH / 4), gridspec_kw={"wspace": 0.3}
    )
    cmap_lu, _ = get_modis_lu_cmap()

    ax_lu: plt.Axes
    ax_lu.pcolormesh(ds_lu["west_east"], ds_lu["south_north"], ds_lu["lu"], cmap=cmap_lu, vmin=1, vmax=21)
    ax_lu.scatter(we_path, sn_path, color="k", marker="o", s=10, label="Interpolation points")
    ax_lu.scatter(*loc_cabauw, color="k", marker="x", s=10, label="Cabauw tower")
    ax_lu.plot(we_path, sn_path, color="k")
    ax_lu.grid()

    ax_lu.set_xticks(np.arange(WE_SLICE.start, WE_SLICE.stop + 1, 1))
    ax_lu.set_xlabel("Grid coordinate, west - east")
    ax_lu.set_yticks(np.arange(SN_SLICE.start, SN_SLICE.stop + 1, 1))
    ax_lu.set_ylabel("Grid coordinate, south - north")

    # %% Plot shfx
    ax_hfx: plt.Axes
    ds_meteo_flux["HSON"].plot(ax=ax_hfx, label="Observation", color="C0")
    ds_lu["hfx"].sel(west_east=loc_bad[0], south_north=loc_bad[1]).plot(
        ax=ax_hfx, label="Evergreen Broadleaf Forest", color="green"
    )
    ds_lu["hfx"].sel(west_east=loc_good[0], south_north=loc_good[1]).plot(
        ax=ax_hfx, label="Croplands", color="black"
    )
    ax_hfx.hlines(0, *ds_lu["Time"].isel(Time=[0, -1]), linestyles="dashed", colors="grey")

    ax_hfx.margins(x=0)
    ax_hfx.xaxis.set_major_formatter(hour_formatter)
    ax_hfx.set_title("")
    ax_hfx.set_ylabel(r"$\overline{w'\theta'}$, K m s$^{-1}$")

    # %% Plot ust
    ax_ust: plt.Axes
    ds_meteo_flux["USTAB"].plot(ax=ax_ust, label="Observation", color="C0")
    ds_lu["ust"].sel(west_east=loc_bad[0], south_north=loc_bad[1]).plot(
        ax=ax_ust, color="green"
    )
    ds_lu["ust"].sel(west_east=loc_good[0], south_north=loc_good[1]).plot(
        ax=ax_ust, color="black"
    )

    ax_ust.margins(x=0)
    ax_ust.xaxis.set_major_formatter(hour_formatter)
    ax_ust.set_title("")
    ax_ust.set_ylabel(r"$u_*$, m s$^{-1}$")
    ax_ust.set_ylim(0)

    fig.legend(ncols=2, bbox_to_anchor=(0.5, 1), loc="upper center")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)
    fig.savefig("plots_new/landuse.pdf")
    fig.show()
