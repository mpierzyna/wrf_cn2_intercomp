import contextlib
import datetime
import pathlib
import warnings
from typing import List, Optional, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cmcrameri import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

from utils import (
    load_xlas_obs, load_las_obs,
    cn2_wrf_to_xlas_obs, cn2_wrf_to_las_obs,
    get_wrf_smoothed, hour_formatter
)

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

FIG_1_4 = (FIG_WIDTH / 4, FIG_WIDTH / 4)  # 1/4 panel (quadratic)
FIG_3_4 = (FIG_WIDTH / 4 * 3, FIG_WIDTH / 4)  # 3/4 panel (quadratic)
FIG_1_3 = (FIG_WIDTH / 3, FIG_WIDTH / 3)  # 1/3 panel (quadratic)
FIG_2_2 = (FIG_WIDTH / 2, FIG_WIDTH / 2 * (3 / 4))  # 1/2 panel (4:3 aspect ratio)

LCN2_SCORE_MIN_LAS = -16
LCN2_SCORE_MIN_XLAS = -16


class ObsWRFCn2Iterator:
    def __init__(self, cases: List[str], scint_cn2_dict: Dict[str, xr.Dataset], meteo_cn2_ds: xr.Dataset,
                 wrf_cn2_list: List[Tuple[xr.Dataset, int]], blh_key: str, enforce_sl: bool = True,
                 use_bo_corr: bool = False, agg_to_scint: bool = False):
        """
        Parameters
        ----------
        cases : List[str]
            Names of cases
        scint_cn2_dict : Dict[str, xr.Dataset]
            Dictionary of scintillometer-based Cn2 datasets, keys are instrument names
        meteo_cn2_ds : xr.Dataset
            Dataset of meteo-based Cn2
        wrf_cn2_list : List[Tuple[xr.Dataset, int]]
            List of WRF Cn2 datasets, each tuple contains the dataset and the shift to apply to XLAS time axis
        blh_key : str
            Key of BLH variable in WRF dataset
        enforce_sl : bool
            Flag to filter estimated Cn2 for SL assumption
        use_bo_corr : bool
            Flag to apply Bowen ratio correction to WRF and Meteo Cn2
        """
        self.scint_cn2_dict: Dict[str, xr.Dataset] = scint_cn2_dict

        if agg_to_scint:
            # Compute 30min averages matching scintillometer
            meteo_cn2_ds = meteo_cn2_ds.resample(time="30min").mean().interpolate_na(dim="time")
            meteo_cn2_ds = meteo_cn2_ds.assign_coords(time=meteo_cn2_ds.time + pd.Timedelta("15min"))
        self.meteo_cn2_ds: xr.Dataset = meteo_cn2_ds

        if agg_to_scint:
            # Compute 30min averages matching scintillometer
            warnings.warn("Do not use this! One should average the input data before computing Cn2!")
            for i, (ds_wrf_cn2, xlas_shift) in enumerate(wrf_cn2_list):
                bool_vars = [v for v in ds_wrf_cn2 if ds_wrf_cn2[v].dtype == bool]
                ds_wrf_cn2 = ds_wrf_cn2.resample(time="30min").mean()
                ds_wrf_cn2 = ds_wrf_cn2.assign_coords(time=ds_wrf_cn2.time + pd.Timedelta("15min"))
                for v in bool_vars:
                    ds_wrf_cn2[v] = ds_wrf_cn2[v].round().astype(bool)
                wrf_cn2_list[i] = (ds_wrf_cn2, xlas_shift)
        self.wrf_cn2_dict: Dict[str, Tuple[xr.Dataset, int]] = dict(zip(cases, wrf_cn2_list))

        self.cases = cases
        self.n_cases = len(self.cases)

        self.scints = list(scint_cn2_dict.keys())
        self.n_scints = len(self.scints)

        self.blh_key = blh_key
        self.enforce_sl = enforce_sl
        self.use_bo_corr = use_bo_corr
        self.agg_to_scint = agg_to_scint

        # Get time windows from WRF file to apply to the observation datasets
        self.time_windows: Dict[str, List] = {
            c: pd.to_datetime(self.wrf_cn2_dict[c][0]["time"].values[[0, -1]]).to_list()
            for c in self.cases
        }

        # Get stability mask from WRF
        self.is_stable_dict: Dict[str, xr.DataArray] = {
            c: self.wrf_cn2_dict[c][0]["is_stable"]
            for c in self.cases
        }

    def get_wrf_blh(self, case: str) -> xr.DataArray:
        ds_wrf = self.get_wrf_cn2_ds(case=case, scint=None)
        blh = ds_wrf[self.blh_key]
        if self.blh_key == "blh_wrf":
            blh = blh.mean("path").squeeze()
        blh = get_wrf_smoothed(blh, cutoff_period_h=3)
        return blh

    def _get_wrf_las_ds(self, case: str) -> xr.Dataset:
        """Get simulated Cn2 for LAS, filtered for MOST assumption"""
        ds_wrf_cn2 = self.get_wrf_cn2_ds(case=case, scint=None)
        return cn2_wrf_to_las_obs(ds_wrf_cn2, blh=self.get_wrf_blh(case))

    def _get_wrf_xlas_ds(self, case: str) -> xr.Dataset:
        """Get simulated Cn2 for XLAS, filtered for MOST assumption and path-averaged"""
        ds_wrf_cn2 = self.get_wrf_cn2_ds(case=case, scint=None)
        return cn2_wrf_to_xlas_obs(ds_wrf_cn2, blh=self.get_wrf_blh(case), weighted_mean=True)

    def get_scint_cn2(self, case: str, scint: str) -> xr.DataArray:
        """Get scintillometer-based Cn2 for given `case` and `scint` as xr.DataArray"""
        _, xlas_shift = self.wrf_cn2_dict[case]
        time_slice = slice(*self.time_windows[case])
        ds_scint_cn2 = self.scint_cn2_dict[scint].sel(time=time_slice)["Cn2_os"].copy()
        if scint == "XLAS":
            ds_scint_cn2 = ds_scint_cn2.shift(time=xlas_shift)
        return ds_scint_cn2

    def get_meteo_cn2_ds(self, case: str, scint: Optional[str], is_stable: Optional[bool] = None,
                         interp_to_scint: bool = False) -> xr.Dataset:
        """Get meteo-based Cn2 for given `case` collocated to `scint` as xr.DataArray

        Parameters
        ----------
        case : str
            Case name
        scint : str
            Instrument name
        is_stable : Optional[bool]
            Set stable or unstable periods of result to NaN. If None, no filtering is applied.
        interp_to_scint : bool
            Interpolate Cn2 to match scintillometer time axis (e.g., for score computation)
        """
        time_slice = slice(*self.time_windows[case])
        ds_meteo_cn2 = self.meteo_cn2_ds.sel(time=time_slice)
        if scint == "LAS":
            ds_meteo_cn2 = ds_meteo_cn2.sel(z=10)  # 10m height of LAS
        elif scint == "XLAS":
            ds_meteo_cn2 = ds_meteo_cn2.sel(z=80)  # 80m height of XLAS
        elif scint is None:
            return ds_meteo_cn2
        else:
            raise ValueError(f"Unknown instrument {scint}")

        # Make copy, so we don't accidentally modify array in every call (Bowen ratio!)
        ds_meteo_cn2 = ds_meteo_cn2.copy()

        if self.use_bo_corr:
            # Apply Bowen ratio correction to all cn2 prefixed variables
            for v in ds_meteo_cn2:
                if v.startswith("cn2"):
                    ds_meteo_cn2[v] = ds_meteo_cn2[v] * ds_meteo_cn2["bo_corr"]

        if interp_to_scint:
            # Import timestep of meteo-based Cn2 to match scintillometer
            ds_scint_cn2 = self.get_scint_cn2(case=case, scint=scint)
            ds_meteo_cn2 = ds_meteo_cn2.interp(time=ds_scint_cn2["time"])

        if self.enforce_sl:
            # Filter for MOST assumption
            blh = self.get_wrf_blh(case).interp(time=ds_meteo_cn2["time"])  # if interp to scint, interp BLH too
            most_valid = ds_meteo_cn2.z <= 0.1 * blh
            ds_meteo_cn2 = ds_meteo_cn2.where(most_valid)

        if is_stable is not None:
            # Filter for stability
            is_stable_mask = self.is_stable_dict[case].sel(time=ds_meteo_cn2["time"], method="nearest").to_numpy()
            if is_stable:
                ds_meteo_cn2 = ds_meteo_cn2.where(is_stable_mask)
            else:
                ds_meteo_cn2 = ds_meteo_cn2.where(~is_stable_mask)

        return ds_meteo_cn2

    def get_wrf_cn2_ds(self, case: str, scint: Optional[str], is_stable: Optional[bool] = None,
                       interp_to_scint: bool = False) -> xr.Dataset:
        """Get simulated Cn2 for given `case` collocated to `scint` as xr.DataArray

        Parameters
        ----------
        case : str
            Case name
        scint : str
            Instrument name
        is_stable : Optional[bool]
            Set stable or unstable periods of result to NaN. If None, no filtering is applied.
        interp_to_scint : bool
            Interpolate Cn2 to match scintillometer time axis (e.g., for score computation)
        """
        if scint == "LAS":
            ds_wrf_cn2 = self._get_wrf_las_ds(case)
        elif scint == "XLAS":
            ds_wrf_cn2 = self._get_wrf_xlas_ds(case)
        elif scint is None:
            ds_wrf_cn2, _ = self.wrf_cn2_dict[case]
        else:
            raise ValueError(f"Unknown instrument {scint}")

        # Make copy, so we don't accidentally modify array in every call (Bowen ratio!)
        ds_wrf_cn2 = ds_wrf_cn2.copy()

        if self.use_bo_corr:
            # Apply Bowen ratio correction to all cn2 prefixed variables
            for v in ds_wrf_cn2:
                if v.startswith("cn2"):
                    ds_wrf_cn2[v] = ds_wrf_cn2[v] * ds_wrf_cn2["bo_corr"]

        if interp_to_scint:
            # Import timestep of WRF-based Cn2 to match scintillometer
            ds_scint_cn2 = self.get_scint_cn2(case=case, scint=scint)
            ds_wrf_cn2 = ds_wrf_cn2.interp(time=ds_scint_cn2["time"])
            ds_wrf_cn2["is_stable"] = ds_wrf_cn2["is_stable"].sel(time=ds_scint_cn2["time"], method="nearest")

        if is_stable is not None:
            # Filter for stability
            if is_stable:
                ds_wrf_cn2 = ds_wrf_cn2.where(ds_wrf_cn2["is_stable"])
            else:
                ds_wrf_cn2 = ds_wrf_cn2.where(~ds_wrf_cn2["is_stable"])

        return ds_wrf_cn2


def plot_cn2_ts(it: ObsWRFCn2Iterator) -> (plt.Figure, plt.Figure):
    # Define cn2 limits for figures
    cn2_min, cn2_max = 1e-17, 1e-12

    # Create figure
    w = FIG_WIDTH
    h = 0.20 * FIG_WIDTH * it.n_cases

    # Alternative scintillometer names
    alt_scint_names = {
        "LAS": r"10\,m",
        "XLAS": r"80\,m"
    }

    def plot_common(ax: plt.Axes, is_stable: xr.DataArray, lcn2_min: float, row_title: str = None) -> None:
        """Common config for all time series axis"""
        ax.fill_between(
            is_stable["time"],
            cn2_min, cn2_max,
            where=is_stable,
            color="grey",
            alpha=.1,
            label="Stable"
        )

        ax.xaxis.set_major_formatter(hour_formatter)
        ax.margins(x=0)

        ax.set_yscale("log")
        ax.set_ylim(cn2_min, cn2_max)

        # Lines for orientation
        # ax.axhline(1e-13, color="grey", ls="dashed")
        ax.axhline(10 ** lcn2_min, color="grey", ls="dotted")

        # Add secondary x axis with days
        ax_day = add_day_ax(ax, is_stable.indexes["time"])

        # Adjust label padding
        ax.tick_params(axis="x", pad=1)
        ax_day.tick_params(axis="x", pad=1)

        if row_title:
            ax.text(-.175, 0.5, s=row_title, transform=ax.transAxes, ha="right", va="center", rotation=90)
            ax.set_ylabel(r"$C_n^2$, m$^{-2/3}$")

    def add_day_ax(ax: plt.Axes, time: pd.DatetimeIndex) -> plt.Axes:
        """Add day string to axes"""
        # Ticks for days at noon
        unique_days = time.round("D").unique()[:-1]
        unique_days += pd.Timedelta("12h")

        # Set top ticks using second x-axis
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(unique_days)
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))  # Month Day format

        return ax_top

    @contextlib.contextmanager
    def get_fig_axarr() -> (plt.Figure, np.ndarray):
        fig, axarr = plt.subplots(ncols=2, nrows=it.n_cases, sharey="all", figsize=(w, h))
        try:
            yield fig, axarr
        finally:
            fig.subplots_adjust(left=.1, right=.975, bottom=.075, top=.925, wspace=0.1, hspace=0.375)

    def plot_xlas_var() -> plt.Figure:
        with get_fig_axarr() as (fig, axarr):
            for i, c in enumerate(it.cases):
                for j, s in enumerate(it.scints):
                    # Scintillometer Cn2 (ground truth)
                    scint_cn2 = it.get_scint_cn2(case=c, scint=s)

                    # WRF Cn2
                    ds_wrf_cn2 = it.get_wrf_cn2_ds(case=c, scint=s)
                    wrf_cn2_var = ds_wrf_cn2["cn2_hb15_var"]
                    is_stable = it.is_stable_dict[c]

                    # Plot
                    ax: plt.Axes = axarr[i, j]
                    ax.scatter(scint_cn2["time"], scint_cn2, color="k", label="Obs.", marker="x", s=5)

                    ax.plot(wrf_cn2_var["time"], wrf_cn2_var, color="C1", label="WRF+Var")

                    # First col
                    if j == 0:
                        plot_common(ax, is_stable=is_stable, row_title=c, lcn2_min=LCN2_SCORE_MIN_LAS)
                    else:
                        plot_common(ax, is_stable=is_stable, lcn2_min=LCN2_SCORE_MIN_XLAS)

                    # First row
                    if i == 0:
                        ax.set_title(f"{alt_scint_names[s]}: variance-based $C_n^2$")

                    # Last row
                    if i == it.n_cases - 1:
                        ax.set_xlabel("Time, UTC")

        return fig

    def plot_las_threeway() -> plt.Figure:
        """Three-way comparison between Cn2 from scintillometer, meteo, and WRF for LAS (flux and grad params)"""
        with get_fig_axarr() as (fig, axarr):
            for i, (c, (ax_flux, ax_grad)) in enumerate(zip(it.cases, axarr)):
                # First row
                if i == 0:
                    ax_flux.set_title(f"{alt_scint_names['LAS']}: W71 flux-based $C_n^2$")
                    ax_grad.set_title(f"{alt_scint_names['LAS']}: W71/HB16 gradient-based $C_n^2$")

                # Last row
                if i == it.n_cases - 1:
                    ax_flux.set_xlabel("Time, UTC")
                    ax_grad.set_xlabel("Time, UTC")

                # Scintillometer Cn2 (ground truth)
                scint_cn2 = it.get_scint_cn2(case=c, scint="LAS")

                # Meteo Cn2
                ds_meteo_cn2 = it.get_meteo_cn2_ds(case=c, scint="LAS")
                meteo_cn2_flux_local = ds_meteo_cn2["cn2_w71_flux_local"]
                # meteo_cn2_flux_regional = ds_meteo_cn2["cn2_flux_regional"]
                meteo_cn2_grad = ds_meteo_cn2["cn2_w71_hb16_grad"]

                # WRF Cn2
                ds_wrf_cn2 = it.get_wrf_cn2_ds(case=c, scint="LAS")
                wrf_cn2_flux = ds_wrf_cn2["cn2_w71_flux"]
                wrf_cn2_grad = ds_wrf_cn2["cn2_w71_hb16_grad"]
                is_stable = it.is_stable_dict[c]

                # Plot flux-based
                ax_flux.scatter(scint_cn2["time"], scint_cn2, color="k", label="Obs.", marker="x", s=5)

                ax_flux.plot(meteo_cn2_flux_local["time"], meteo_cn2_flux_local, color="C0", label="MET+Flux")
                # ax_flux.plot(meteo_cn2_flux_regional["time"], meteo_cn2_flux_regional, color="C1",
                #              label="Meteo+Flux (regional)")
                ax_flux.plot(wrf_cn2_flux["time"], wrf_cn2_flux, color="C1", label="WRF+Flux")

                # first col, so set case title
                plot_common(ax_flux, is_stable=is_stable, row_title=c, lcn2_min=LCN2_SCORE_MIN_LAS)

                # Plot gradient-based
                ax_grad.scatter(scint_cn2["time"], scint_cn2, color="k", label="Obs.", marker="x", s=5)
                ax_grad.set_yscale("log")

                ax_grad.plot(meteo_cn2_grad["time"], meteo_cn2_grad, color="C0", label="MET+Grad")
                ax_grad.plot(wrf_cn2_grad["time"], wrf_cn2_grad, color="C1", label="WRF+Grad")

                plot_common(ax_grad, is_stable=is_stable, lcn2_min=LCN2_SCORE_MIN_LAS)

        return fig

    return plot_las_threeway(), plot_xlas_var()


def compute_scores_threeway(it: ObsWRFCn2Iterator) -> pd.DataFrame:
    scores = []
    is_stable_bool2str = {
        True: "Stable",
        False: "Unstable"
    }

    for c in it.cases:
        for is_stable in [True, False]:
            # Scintillometer Cn2 (ground truth)
            scint_lcn2 = np.log10(it.get_scint_cn2(case=c, scint="LAS"))
            scint_lcn2_xlas = np.log10(it.get_scint_cn2(case=c, scint="XLAS"))

            # Meteo Cn2
            ds_meteo_cn2 = it.get_meteo_cn2_ds(case=c, scint="LAS", is_stable=is_stable, interp_to_scint=True)

            # WRF Cn2
            ds_wrf_cn2 = it.get_wrf_cn2_ds(case=c, scint="LAS", is_stable=is_stable, interp_to_scint=True)
            ds_wrf_cn2_xlas = it.get_wrf_cn2_ds(case=c, scint="XLAS", is_stable=is_stable, interp_to_scint=True)

            # Collect all log10(Cn2)
            lcn2_param_dict = {
                "MET+Flux": (np.log10(ds_meteo_cn2["cn2_w71_flux_local"]), scint_lcn2, LCN2_SCORE_MIN_LAS),  # local
                # "Meteo+Flux (regional)": np.log10(ds_meteo_cn2["cn2_flux_regional"]),
                "MET+Grad": (np.log10(ds_meteo_cn2["cn2_w71_hb16_grad"]), scint_lcn2, LCN2_SCORE_MIN_LAS),
                "WRF+Flux": (np.log10(ds_wrf_cn2["cn2_w71_flux"]), scint_lcn2, LCN2_SCORE_MIN_LAS),
                "WRF+Grad": (np.log10(ds_wrf_cn2["cn2_w71_hb16_grad"]), scint_lcn2, LCN2_SCORE_MIN_LAS),
                "WRF+Var": (np.log10(ds_wrf_cn2["cn2_hb15_var"]), scint_lcn2, LCN2_SCORE_MIN_LAS),
                "WRF+Var (XLAS)": (np.log10(ds_wrf_cn2_xlas["cn2_hb15_var"]), scint_lcn2_xlas, LCN2_SCORE_MIN_XLAS),
            }

            # Compute scores
            for k, (lcn2, scint_lcn2_i, lcn2_min) in lcn2_param_dict.items():
                is_valid = (~lcn2.isnull()) & (~scint_lcn2_i.isnull())
                n_valid = int(is_valid.sum())
                lcn2 = lcn2.isel(time=is_valid)
                lcn2 = lcn2.clip(lcn2_min, None)  # clip to min value
                scint_lcn2_i = scint_lcn2_i.isel(time=is_valid)  # copy for each iteration

                bias = lcn2.mean() - scint_lcn2_i.mean()  # positive if overestimated
                lcn2 = lcn2 - bias  # remove bias
                crmse = np.sqrt(((lcn2 - scint_lcn2_i) ** 2).mean())
                corr = pearsonr(lcn2, scint_lcn2_i)[0]

                scores.append([c, is_stable_bool2str[is_stable], k, float(bias), float(crmse), corr, n_valid])

    scores_df = pd.DataFrame(scores, columns=["case", "is_stable", "comp", "Bias", "cRMSE", "r", "n_valid"])
    return scores_df


def plot_scores_threeway(it: ObsWRFCn2Iterator) -> (plt.Figure, plt.Figure):
    # Set up colormaps
    n_cmap_bias = 13
    colors_bias = cm.vik(np.linspace(0, 1, n_cmap_bias))
    cmap_bias = LinearSegmentedColormap.from_list(name="bias", colors=colors_bias, N=n_cmap_bias)

    n_cmap_scores = 10
    colors_scores = cm.tokyo(np.linspace(0, 1, n_cmap_scores + 1))[1:]
    cmap_scores = LinearSegmentedColormap.from_list(name="scores", colors=colors_scores, N=n_cmap_scores)

    cmap_combined = LinearSegmentedColormap.from_list(
        name="combined",
        colors=np.vstack([colors_bias, colors_scores]),
        N=n_cmap_bias + n_cmap_scores,
    )

    df_scores = compute_scores_threeway(it)
    df_scores_annot = df_scores.copy()  # keep unmodified copy for annotation

    score_ranges = {
        "Bias": ([-1.2, 1.2], cmap_bias),
        "cRMSE": ([.2, 1.2], cmap_scores.reversed()),
        "r": ([0, 1], cmap_scores),
    }

    # Normalize scores: Bias to [-1, 1], cRMSE to [0, 1]
    df_scores["Bias"] = df_scores["Bias"] / df_scores["Bias"].abs().max()

    (cRMSE_min, cRMSE_max), _ = score_ranges["cRMSE"]
    df_scores["cRMSE"] = (df_scores["cRMSE"] - cRMSE_min) / (cRMSE_max - cRMSE_min)

    # Clip scores to make sure they appear on the correct colorbar
    df_scores["cRMSE"] = df_scores["cRMSE"].clip(0, .975)
    df_scores["r"] = df_scores["r"].clip(0.025, 1)

    # Invert cRMSE so that small errors appear green / good
    df_scores["cRMSE"] = -1 * df_scores["cRMSE"] + 1

    # Shift and scale scores so that a) bias is centered on first part and b) scores aligned with second part
    bias_scale = n_cmap_bias / (n_cmap_bias + n_cmap_scores)
    scores_scale = n_cmap_scores / (n_cmap_bias + n_cmap_scores)
    scores_offset = bias_scale

    df_scores["Bias"] = (df_scores["Bias"] + 1) / 2 * bias_scale
    df_scores["cRMSE"] = df_scores["cRMSE"] * scores_scale + scores_offset
    df_scores["r"] = df_scores["r"] * scores_scale + scores_offset

    # Comparisons
    comps = [
        "MET+Flux",  # local
        "WRF+Flux",
        "MET+Grad",
        "WRF+Grad",
        "WRF+Var",
        "WRF+Var (XLAS)",
    ]

    fig, axarr = plt.subplots(
        nrows=it.n_cases, ncols=len(comps),
        figsize=(FIG_WIDTH, FIG_WIDTH / 12 * it.n_cases),
        sharex="all", sharey="all"
    )
    # i: row (case), j: col (comp)
    for i, c in enumerate(it.cases):
        for j, (comp, l) in enumerate(zip(comps, "abcdefghijklmnopqrstuvwxyz")):
            df = df_scores[(df_scores["case"] == c) & (df_scores["comp"] == comp)]
            df = df.set_index("is_stable").sort_index()
            df = df[["Bias", "cRMSE", "r"]]

            df_annot = df_scores_annot[(df_scores_annot["case"] == c) & (df_scores_annot["comp"] == comp)]
            df_annot = df_annot.set_index("is_stable").sort_index()
            df_annot = df_annot[["Bias", "cRMSE", "r"]]

            ax: plt.Axes = axarr[i, j]
            sns.heatmap(ax=ax, data=df, cmap=cmap_combined, vmin=0, vmax=1, annot=df_annot, fmt=".2f", cbar=False)
            ax.set_ylabel("")
            ax.tick_params(axis="x",
                           labelbottom=False, bottom=False,
                           labeltop=i == 0, top=True)

            if i == 0:
                ax.set_title(f"({l}) {comp}", y=1.45)
            if j == 0:
                ax.set_ylabel(c)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.01, top=0.8)

    # Plot colorbar
    fig_cbar, axarr = plt.subplots(ncols=len(score_ranges), figsize=(FIG_WIDTH / 1.5, .75))

    for ax, (s, ((s_min, s_max), cmap)) in zip(axarr, score_ranges.items()):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=s_min, vmax=s_max))
        cb = plt.colorbar(sm, cax=ax, orientation='horizontal')
        cb.set_label(s)

    fig_cbar.subplots_adjust(left=0.075, right=0.95, bottom=0.75, top=1)

    return fig, fig_cbar


if __name__ == '__main__':
    # %% Load WRF data
    WRF_ROOT = pathlib.Path("../data/WRF_ens_cs")
    MEMBER = 3  # Fix member for this analysis

    wrf_cases_names = [
        "Spring",
        "Summer",
        "Autumn"
    ]
    wrf_meteo_list = [
        xr.load_dataset(WRF_ROOT / "2022-04-18_22_merged.nc", engine="h5netcdf").isel(member=MEMBER),
        xr.load_dataset(WRF_ROOT / "2022-07-26_28_merged.nc", engine="h5netcdf").isel(member=MEMBER),
        xr.load_dataset(WRF_ROOT / "2022-10-07_09_merged.nc", engine="h5netcdf").isel(member=MEMBER),
    ]
    wrf_cn2_list: List[Tuple[xr.Dataset, int]] = [
        (xr.load_dataset(WRF_ROOT / "2022-04-18_22_merged_cn2.nc", engine="h5netcdf").isel(member=MEMBER), 6),
        (xr.load_dataset(WRF_ROOT / "2022-07-26_28_merged_cn2.nc", engine="h5netcdf").isel(member=MEMBER), 6),
        (xr.load_dataset(WRF_ROOT / "2022-10-07_09_merged_cn2.nc", engine="h5netcdf").isel(member=MEMBER), 0)
    ]

    for ds, (ds_cn2, _) in zip(wrf_meteo_list, wrf_cn2_list):
        # Update stability mask based on smoothed HFX
        hfx_smooth = get_wrf_smoothed(ds["HFX"].mean("path"), cutoff_period_h=3)
        ds["HFX_smooth"] = hfx_smooth
        ds_cn2["is_stable"] = hfx_smooth < 0

    # %% Load observation-based Cn2
    CESAR_ROOT = pathlib.Path("../data/CESAR")
    meteo_cn2_ds = xr.load_dataset(CESAR_ROOT / "cesar_cn2_ct2.nc")

    # %% Load scintillometer-based Cn2 (ground truth)
    scint_cn2_dict = {
        "LAS": load_las_obs("../data/CESAR/Scintillometer/CabauwRS.nc"),
        "XLAS": load_xlas_obs("../data/CESAR/Scintillometer/CabauwIJsselstein.nc"),
    }

    # %% Plot comparisons
    it = ObsWRFCn2Iterator(
        cases=wrf_cases_names,
        scint_cn2_dict=scint_cn2_dict,
        meteo_cn2_ds=meteo_cn2_ds,
        wrf_cn2_list=wrf_cn2_list,
        blh_key="blh_wrf",
        use_bo_corr=True,
        agg_to_scint=False,
    )

    submission_path = pathlib.Path("../data/submission")
    submission_path.mkdir(exist_ok=True, parents=True)

    fig_las, fig_xlas = plot_cn2_ts(it)
    fig_las.savefig("plots_new/cn2_ts_las_threeway.pdf")
    fig_las.show()

    fig_xlas.savefig("plots_new/cn2_ts_xlas_var.pdf")
    fig_xlas.show()

    fig, fig_cb = plot_scores_threeway(it)
    fig.savefig("plots_new/cn2_scores_las_threeway.pdf")
    fig.show()
    fig_cb.savefig("plots_new/cn2_scores_las_threeway_cb.pdf")
    fig_cb.show()
