# TODO - method to run external comfort and post process all

# TODO - method to run all possible processes on a given EPW file

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from ladybugtools_toolkit.external_comfort.external_comfort import (
    ExternalComfort,
    SimulationResult,
)
from ladybugtools_toolkit.prototypes.additional_plots.utci_shade_benefit import shade_benefit_heatmap_histogram

from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.external_comfort.typology import Typologies
from ladybugtools_toolkit.helpers import default_analysis_periods, safe_filename
from ladybugtools_toolkit.ladybug_extension.analysis_period import (
    describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    epw_to_dataframe,
    AnalysisPeriod,
    collection_to_series,
)
from ladybugtools_toolkit.plot import (
    degree_days,
    diurnal,
    heatmap,
    psychrometric,
    sunpath,
    timeseries,
    condensation_potential,
)
from ladybugtools_toolkit.plot._skymatrix import skymatrix
from ladybugtools_toolkit.wind.wind import DirectionBins, Wind
from tqdm import tqdm

from ladybugtools_toolkit.prototypes.additional_plots.radiant_cooling_potential import (
    radiant_cooling_potential,
)
from ladybugtools_toolkit.prototypes.additional_plots.facade_solar import (
    directional_irradiance_matrix,
    annual_surface_insolation,
)
from ladybugtools_toolkit.prototypes.additional_plots.weathersparkish import (
    cloud_cover_categories,
    hours_sunlight,
    hours_sunrise_sunset,
    solar_elevation_azimuth,
)
from ladybugtools_toolkit.ladybug_extension.pv_yield import (
    PVYieldMatrix,
    IrradianceType,
    IrradianceUnit,
)
from ladybugtools_toolkit.plot._utci import (
    utci_comfort_band_comparison,
    utci_day_comfort_metrics,
    utci_comparison_diurnal_day,
    utci_heatmap_difference,
    utci_pie,
    utci_journey,
    utci_monthly_histogram,
)

# from ladybugtools_toolkit.plot.__utci_prototypes import (
#     utci_feasibility,
#     utci_shade_benefit,
# )
from ladybugtools_toolkit.external_comfort.utci.postprocess import (
    shade_benefit_category,
)


def run_everything(epw_file: Path, output_directory: Path) -> None:
    """The path to an EPW file.

    Args:
        epw_file (Path): Epw file to run everything on.
        output_directory (Path): The path to the output directory which will be filled with goodies.

    Returns:
        None
    """

    plt.close("all")

    epw_file = Path(epw_file)
    output_directory = Path(output_directory)

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    # load EPW file
    epw = EPW(epw_file)
    epw.save(output_directory / epw_file.name)
    epw_df = epw_to_dataframe(epw, include_additional=True).droplevel([0, 1], axis=1)
    epw_name = epw_file.stem

    # Plot windroses
    wr_dir = output_directory / "windroses"
    wr_dir.mkdir(parents=True, exist_ok=True)
    w = Wind.from_epw(epw)
    db = DirectionBins(centered=True, directions=18)
    pbar = tqdm(default_analysis_periods())
    for ap in pbar:
        pbar.set_description(f"Plotting windrose for {describe_analysis_period(ap)}")
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        w.filter_by_analysis_period(ap).plot_windrose(ax=ax, direction_bins=db)
        ax.set_title(describe_analysis_period(ap, save_path=False))
        fig.savefig(
            wr_dir / f"windrose_{describe_analysis_period(ap, save_path=True)}.png",
            transparent=True,
        )
        plt.close(fig)

    pbar = tqdm(epw_df.columns)
    for var, series in epw_df.iteritems():
        # skip non-numeric columns
        try:
            float(series[0])
        except (ValueError, TypeError):
            continue

        # plot diurnals
        diurnal_dir = output_directory / "diurnals"
        diurnal_dir.mkdir(parents=True, exist_ok=True)
        for period in ["daily", "weekly", "monthly"]:
            pbar.set_description(f"Plotting {var} {period} diurnal")
            fig, ax = plt.subplots()
            diurnal(series, ax=ax, period=period)
            ax.set_title(f"{epw_name} - {var}")
            fig.savefig(
                diurnal_dir / f"{period}_{safe_filename(var)}.png", transparent=True
            )
            plt.close(fig)

        # plot heatmaps
        heatmap_dir = output_directory / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        pbar.set_description(f"Plotting {var} heatmap")
        fig, ax = plt.subplots()
        heatmap(series=series, ax=ax)
        ax.set_title(f"{epw_name}\n{var}")
        fig.savefig(heatmap_dir / f"{safe_filename(var)}.png", transparent=True)
        plt.close(fig)

        # plot time series
        timeseries_dir = output_directory / "timeseries"
        timeseries_dir.mkdir(parents=True, exist_ok=True)
        pbar.set_description(f"Plotting {var} time series")
        fig, ax = plt.subplots()
        timeseries(series=series, ax=ax)
        ax.set_title(f"{epw_name}\n{var}")
        fig.savefig(timeseries_dir / f"{safe_filename(var)}.png", transparent=True)
        plt.close(fig)

        pbar.update()

    # plot misc
    misc_dir = output_directory / "misc"
    misc_dir.mkdir(parents=True, exist_ok=True)

    # hdd/cdd
    fig = degree_days(epw)
    fig.savefig(misc_dir / "degree_days.png", transparent=True)
    plt.close(fig)

    # psychrometric chart
    fig = psychrometric(epw)
    fig.savefig(misc_dir / "psychrometric.png", transparent=True)
    plt.close(fig)

    # sky matrix
    fig, ax = plt.subplots(1, 1)
    skymatrix(epw=epw, ax=ax, density=4, analysis_period=AnalysisPeriod(timestep=15))
    fig.savefig(misc_dir / "sky_matrix.png", transparent=True)
    plt.close(fig)

    # sunpath
    fig, ax = plt.subplots(1, 1)
    sunpath(location=epw.location, ax=ax, sun_size=4)
    fig.savefig(misc_dir / "sun_path.png", transparent=True)
    plt.close(fig)

    # insolation
    dim = directional_irradiance_matrix(epw_file=epw_file)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
    annual_surface_insolation(directional_irradiance_matrix=dim, ax=ax, title=epw_name)
    fig.savefig(misc_dir / "directional_irradiance.png", transparent=True)
    plt.close(fig)

    # radiant cooling potential
    fig, ax = plt.subplots(1, 1)
    radiant_cooling_potential(dpt=epw_df["Dew Point Temperature (C)"], ax=ax)
    ax.set_title(f"{epw_name}\nRadiant cooling potential")
    fig.savefig(misc_dir / "radiant_cooling_potential.png", transparent=True)
    plt.close(fig)

    # condensation potential
    fig, ax = plt.subplots(1, 1)
    condensation_potential(
        epw_df["Dry Bulb Temperature (C)"], epw_df["Dew Point Temperature (C)"], ax=ax
    )
    ax.set_title(f"{epw_name}\nCondensation potential")
    fig.savefig(misc_dir / "condensation_potential.png", transparent=True)
    plt.close(fig)

    # weathersparkish
    fig, ax = plt.subplots(1, 1)
    cloud_cover_categories(epw=epw, ax=ax)
    ax.set_title(f"{epw_name}\nCloud cover categories")
    fig.savefig(misc_dir / "cloud_cover_categories.png", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    hours_sunlight(epw.location, ax=ax)
    ax.set_title(f"{epw_name}\nHours sunlight")
    fig.savefig(misc_dir / "hours_sunlight.png", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    hours_sunrise_sunset(epw.location, ax=ax)
    ax.set_title(f"{epw_name}\nSun-up hours")
    fig.savefig(misc_dir / "hours_sunrise_sunset.png", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    solar_elevation_azimuth(epw.location, ax=ax)
    ax.set_title(f"{epw_name}\nSolar elevation/azimuth")
    fig.savefig(misc_dir / "solar_elevation_azimuth.png", transparent=True)
    plt.close(fig)

    # pv yield/insolation
    pvym = PVYieldMatrix.from_epw(epw)
    for rt in IrradianceType:
        fig, ax = plt.subplots(1, 1)
        pvym.plot_tof(ax=ax, irradiance_type=rt, irradiance_unit=IrradianceUnit.KWH_M2
        fig.savefig(misc_dir / f"pv_yield_{rt.to_string()}.png", transparent=True)
        plt.close(fig)

    # run external comfort process
    sr = SimulationResult(
        EpwFile=epw_file,
        GroundMaterial=Materials.LBT_ConcretePavement.value,
        ShadeMaterial=Materials.FABRIC.value,
    ).run()

    ec_openfield = ExternalComfort(
        SimulationResult=sr, Typology=Typologies.OPENFIELD.value
    )
    ec_skyshelter = ExternalComfort(
        SimulationResult=sr, Typology=Typologies.SKY_SHELTER.value
    )

    # plot external comfort results
    utci_dir = output_directory / "utci"
    utci_dir.mkdir(parents=True, exist_ok=True)
    for _ec in [ec_openfield, ec_skyshelter]:
        fig, ax = plt.subplots(1, 1)
        utci_day_comfort_metrics(
            utci=collection_to_series(_ec.universal_thermal_climate_index),
            dbt=collection_to_series(_ec.dry_bulb_temperature),
            mrt=collection_to_series(_ec.mean_radiant_temperature),
            rh=collection_to_series(_ec.relative_humidity),
            ws=collection_to_series(_ec.wind_speed),
            ax=ax,
        )
        fig.savefig(
            utci_dir
            / f"utci_day_comfort_metrics_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        utci_pie(_ec.universal_thermal_climate_index, ax=ax)
        fig.savefig(
            utci_dir / f"utci_pie_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_utci_day_comfort_metrics(
            ax=ax,
        )
        fig.savefig(
            utci_dir
            / f"utci_day_comfort_metrics_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_utci_heatmap(
            ax=ax,
        )
        fig.savefig(
            utci_dir / f"utci_heatmap_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig = _ec.plot_utci_heatmap_histogram()
        fig.savefig(
            utci_dir / f"utci_heatmap_histogram_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_utci_histogram(ax=ax, xlim=(-15, 55)
        fig.savefig(
            utci_dir / f"utci_histogram_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_utci_distance_to_comfortable(ax=ax)
        fig.savefig(
            utci_dir
            / f"utci_distance_to_comfortable_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_dbt_heatmap(ax=ax)
        fig.savefig(
            utci_dir / f"dbt_heatmap_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_rh_heatmap(ax=ax)
        fig.savefig(
            utci_dir / f"rh_heatmap_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_ws_heatmap(ax=ax)
        fig.savefig(
            utci_dir / f"ws_heatmap_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        _ec.plot_mrt_heatmap(ax=ax)
        fig.savefig(
            utci_dir / f"mrt_heatmap_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        utci_monthly_histogram(_ec.universal_thermal_climate_index, ax=ax)
        fig.savefig(
            utci_dir / f"utci_monthly_histogram_{safe_filename(_ec.typology.name)}.png",
            transparent=True,
        )
        plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    utci_comfort_band_comparison(
        [
            ec_openfield.universal_thermal_climate_index,
            ec_skyshelter.universal_thermal_climate_index,
        ],
        ax=ax,
    )
    fig.savefig(
        utci_dir / f"utci_comfort_band_comparison.png",
        transparent=True,
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    utci_comparison_diurnal_day(
        [
            ec_openfield.universal_thermal_climate_index,
            ec_skyshelter.universal_thermal_climate_index,
        ],
        ax=ax,
    )
    fig.savefig(
        utci_dir / f"utci_comparison_diurnal_day.png",
        transparent=True,
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    utci_heatmap_difference(
        ec_openfield.universal_thermal_climate_index,
        ec_skyshelter.universal_thermal_climate_index,
        ax=ax,
    )
    fig.savefig(
        utci_dir / f"utci_heatmap_difference.png",
        transparent=True,
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    utci_journey([23, 19, 20, 25, 14], ax=ax)
    fig.savefig(
        utci_dir / f"utci_journey.png",
        transparent=True,
    )
    plt.close(fig)

    # fig = utci_feasibility(epw=epw)
    # fig.savefig(
    #     utci_dir / f"utci_feasibility.png",
    #     transparent=True,
    # )
    # plt.close(fig)

    fig = shade_benefit_heatmap_histogram(sr)
    fig.savefig(
        utci_dir / f"shade_benefit.png",
        transparent=True,
    )
    plt.close(fig)
    
    # usbc = shade_benefit_category(
    #     unshaded_utci=ec_openfield.universal_thermal_climate_index,
    #     shaded_utci=ec_skyshelter.universal_thermal_climate_index,
    # )
    # fig = utci_shade_benefit(usbc)
    # fig.savefig(
    #     utci_dir / f"utci_shade_benefit.png",
    #     transparent=True,
    # )
    # plt.close(fig)


if __name__ is "__main__":
    epw_file = r"C:\Users\tgerrish\Buro Happold\0053340 DDC - Project W Master - Climate\EPW_Modified\SAU_MD_Prince.Abdulmajeed.Bin.Abdulaziz.AP.404010_TMYx.2007-2021_FIXED_TG_300M.epw"
    run_everything(
        epw_file=epw_file, output_directory=r"C:\Users\tgerrish\Desktop\temp"
    )
