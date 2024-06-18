"""Methods to extract NCM templates from Access databases and save them as JSON files for use in Honeybee.
Databases can usually be found here - https://www.uk-ncm.org.uk/download.jsp?id=35
"""

# pylint: disable=E0401
import calendar
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pypyodbc
from honeybee.config import folders
from honeybee.typing import clean_string
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import (EnergyWindowFrame,
                                                 EnergyWindowMaterialGas,
                                                 EnergyWindowMaterialGlazing,
                                                 WindowConstruction)
from honeybee_energy.material.opaque import EnergyMaterial
from honeybee_energy.programtype import (ElectricEquipment, Lighting, People,
                                         ProgramType, ServiceHotWater,
                                         Setpoint, Ventilation)
from honeybee_energy.schedule.ruleset import (ScheduleDay, ScheduleRule,
                                              ScheduleRuleset,
                                              ScheduleTypeLimit)
from ladybug.dt import Date
from tqdm import tqdm

# pylint: enable=E0401


logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])

pypyodbc.lowercase = False

DATA_PATH = Path(__file__).parent.parent / "data"

DAYS = [
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
    "HOLIDAY",
]


def mirror_directory(
    source_dir: str | Path, target_dir: str | Path, overwrite: bool = False
) -> None:
    """
    Copies all files from source_dir to target_dir. If target_dir doesn't exist, it is created.
    If a file with the same name exists in target_dir, it is overwritten.

    Args:
        source_dir (str | Path): The source directory path as a string or Path object.
        target_dir (str | Path): The target directory path as a string or Path object.
    """
    source_dir = Path(source_dir).absolute()
    target_dir = Path(target_dir).absolute()

    for item in source_dir.iterdir():
        if item.is_dir():
            # If item is a directory, recursively call this function
            new_target = target_dir / item.relative_to(source_dir)
            new_target.mkdir(parents=True, exist_ok=True)
            mirror_directory(item, new_target, overwrite)
        elif item.is_file():
            # If item is a file, copy it to the target directory, overwriting if it exists and overwrite is True
            target_file_path = target_dir / item.relative_to(source_dir)
            if not target_file_path.exists() or overwrite:
                shutil.copy2(item, target_file_path)
                print(f"Copied file: {target_file_path}")


def load_access_tables(mdb_file: str | Path) -> dict[str, pd.DataFrame]:
    """Load all tables from an Access database file into pandas DataFrames.

    Args:
        mdb_file (str | Path):
            The path to the Access database file.

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary of tables in the database, with the table name as the key.
    """
    mdb_file = Path(mdb_file).absolute()
    if not mdb_file.exists():
        raise FileNotFoundError(f"{mdb_file} does not exist.")

    with pypyodbc.connect(
        "Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
        + f"Dbq={mdb_file.as_posix()};"
    ) as conn:
        cur = conn.cursor()
        tables = []
        d = {}
        for i in cur.tables(tableType="Table"):
            _, _, tbl, _, _ = i
            tables.append(tbl)

        pbar = tqdm(tables)
        for tbl in pbar:
            pbar.set_description(f"Loading {mdb_file.name}")
            try:
                sql_query = f"SELECT * FROM {tbl};"
                vals = cur.execute(sql_query).fetchall()
                cols = [column[0] for column in cur.description]
                d[tbl] = pd.DataFrame(vals, columns=cols)
            except pypyodbc.ProgrammingError:
                continue

    return d

def create_ncm_constructions_from_mdb() -> (
    tuple[list[OpaqueConstruction], list[WindowConstruction], list[EnergyWindowFrame]]
):
    """Create a list of constructions for NCM database from the MDB files containing these.

    Returns:
        tuple[list[OpaqueConstruction], list[WindowConstruction], list[EnergyWindowFrame]]:
            A tuple of opaque constructions, window constructions, and window frames. Window constructions have no frame assigned.
    """
    roughness_ref = {
        1: "VeryRough",
        2: "Rough",
        3: "MediumRough",
        4: "MediumSmooth",
        5: "Smooth",
        6: "VerySmooth",
    }

    d = load_access_tables(DATA_PATH / "NCM_db_construction_public_29Apr22.mdb")

    sectors = []
    dates = []
    types = []
    faces = []
    constructions = []
    names = []
    errs = []
    for facetype in ["Wall", "Roof", "Floor", "Door"]:
        pbar = tqdm(list(d[f"ConstbySectorDate{facetype}"].iterrows()))
        for _, row in pbar:
            pbar.set_description(f"Processing {row.CONSTRUCTION}")
            try:
                constr_id = d[f"i_{facetype.lower()}"][
                    d[f"i_{facetype.lower()}"].ID == row.C_ID
                ].CONSTRUCTION.values[0]
                layers_id = d["constructions"][
                    d["constructions"].ID == constr_id
                ].LAYERS.values[0]
                roughness = roughness_ref[
                    d["constructions"][d["constructions"].ID == constr_id][
                        "SURFACE-ROUGHNESS"
                    ].values[0]
                ]

                layer_materials = d["layer_materials"][
                    d["layer_materials"].LAYER == layers_id
                ]
                if len(layer_materials) < 1:
                    continue
                materials = []
                for _, layer_material in layer_materials.iterrows():
                    # get material from material table
                    mat = d["materials"][
                        d["materials"].ID == layer_material.MATERIAL1
                    ].squeeze()
                    if mat.TYPE == "PROPERTIES":
                        # normal material
                        material = EnergyMaterial(
                            identifier=clean_string(f"NCM_{mat.NAME}"),
                            thickness=layer_material.TH / 1000,
                            conductivity=mat.COND,
                            specific_heat=mat["S-H"],
                            density=mat.DENS,
                            roughness=roughness,
                        )
                    else:
                        # air gap material
                        material = EnergyMaterial(
                            identifier=clean_string(f"NCM_{mat.NAME}"),
                            thickness=layer_material.TH / 1000,
                            conductivity=0.556,
                            specific_heat=1000,
                            density=1.28,
                            roughness="Smooth",
                        )
                    materials.append(material)
                constr = OpaqueConstruction(
                    identifier=clean_string(f"NCM_{row.CONSTRUCTION}"),
                    materials=materials,
                )
                constructions.append(constr)
                sectors.append(row.Sector)
                dates.append(row.Date)
                types.append(row.GD)
                faces.append(facetype)
                names.append(row.CONSTRUCTION)
            except ValueError as e:
                errs.append((row.CONSTRUCTION, e))
    df = pd.DataFrame(
        [sectors, dates, types, faces, names, constructions],
        index=["SECTOR", "DATE", "TYPE", "FACE", "NAME", "CONSTRUCTION"],
    ).T

    # save constructions to separate files to help organise
    for date in df.DATE.unique():
        df_datefiltered = df[df.DATE == date]
        for sector in df_datefiltered.SECTOR.unique():
            df_datesectorfiltered = df_datefiltered[df_datefiltered.SECTOR == sector]
            ff = DATA_PATH / "custom_standards" / "constructions" / f"NCM_{date}_{sector}.json"
            with open(ff, "w") as fp:
                json.dump(
                    {
                        i.identifier: i.to_dict()
                        for i in df_datesectorfiltered.CONSTRUCTION
                    },
                    fp,
                    indent=4,
                )


def create_ncm_glazings_from_mdb() -> list[WindowConstruction]:
    """Create window constructions for NCM database from the MDB file.

    Returns:
        list[WindowConstruction]: A list of window constructions for the NCM database.
    """

    d = load_access_tables(
        r"C:\Users\tgerrish\Documents\GitHub\BHoM\LadyBugTools_Prototypes\LadyBugTools_Prototype_Engine\Python\MasterplanEnergyDemand\masterplanenergydemand\data\NCM_db_glazing_public_29Apr22.mdb"
    )

    frames = []
    pbar = tqdm(list(d["frames"].iterrows()))
    for _, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")
        frames.append(
            EnergyWindowFrame(
                identifier=clean_string(row.NAME),
                conductance=row.CONDUCTANCE,
                width=row.WIDTH / 1000 if row.WIDTH else 0.0345,
            )
        )

    glazings = []
    errs = []
    pbar = tqdm(list(d["glazings"].iterrows()))
    for _, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")
        try:
            layers = ["P1", "G1", "P2", "G2", "P3", "G3", "P4"]
            materials = []
            for layer in layers:
                if not np.isnan(row[layer]):
                    if "G" in layer:
                        gas = d["windowgas"][d["windowgas"].ID == row[layer]].squeeze()
                        gas_type = "Argon" if "argon" in gas.Name.lower() else "Air"
                        mat = EnergyWindowMaterialGas(
                            identifier=clean_string(f"{gas.Name}"),
                            gas_type=gas_type,
                            thickness=gas.Thickness / 1000,
                        )
                    else:
                        pane = d["panes"][d["panes"].ID == row[layer]].squeeze()
                        mat = EnergyWindowMaterialGlazing(
                            identifier=clean_string(f"{pane.Name}"),
                            conductivity=pane.Conductivity,
                            thickness=pane.Thickness,
                            solar_transmittance=pane["Solar transmittance"],
                            solar_reflectance=np.nan_to_num(
                                pane["Outside solar reflectance"], nan=0.075
                            ),
                            visible_transmittance=np.nan_to_num(
                                pane["Visible transmittance"], nan=0.53
                            ),
                            infrared_transmittance=np.nan_to_num(
                                pane["Infra red transmittance"], nan=0
                            ),
                            emissivity=np.nan_to_num(
                                pane["Outside Infra Red reflectance"], nan=0.84
                            ),
                            emissivity_back=np.nan_to_num(
                                pane["Inside Infra Red reflectance"], nan=0.84
                            ),
                            visible_reflectance=np.nan_to_num(
                                pane["Outside Visible reflectance"], nan=0.075
                            ),
                        )
                    materials.append(mat)
            glazings.append(
                WindowConstruction(
                    identifier=clean_string(f"NCM_{row.NAME}"), materials=materials
                )
            )
        except (ValueError, AssertionError) as e:
            errs.append((row.NAME, e))
            continue

    # write to custom_standards folder
    with open(
        DATA_PATH / "custom_standards" / "constructions" / "NCM_glazing.json", "w"
    ) as f:
        json.dump({glz.identifier: glz.to_dict() for glz in glazings}, f, indent=4)

    # TODO - frame materials are not being saved as they cause HB to fail to load custom constructions
    # with open(
    #     DATA_PATH / "custom_standards" / "constructions" / "NCM_frame.json", "w"
    # ) as f:
    #     json.dump({frame.identifier: frame.to_dict() for frame in frames}, f, indent=4)

    return glazings


def create_ncm_programs_from_mdb() -> list[ProgramType]:
    """Create program types for NCM database from the MDB file.

    Returns:
        list[ProgramType]: A list of program types for the NCM database.
    """

    d = load_access_tables(DATA_PATH / "NCM_db_activity_public_15Dec21.mdb")

    temperature_type_limit = ScheduleTypeLimit(
        identifier="Temperature", numeric_type="Continuous", unit_type="Temperature"
    )
    month_lookup = {calendar.month_abbr[i].title(): i for i in range(1, 13, 1)}

    programs: list[ProgramType] = []
    schedules: list[ScheduleRuleset] = []
    pbar = tqdm(list(d["activity"].iterrows())[0:])
    for _, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")

        # region: SETPOINT
        heat_multiple = d["annual_weekly_schedules"][
            d["annual_weekly_schedules"].ANNUAL_SCHEDULE == row.HEAT_SET_SCH
        ]
        end_date = None
        day_schedules = {}
        rules = []
        for n, (_, _row) in enumerate(list(heat_multiple.iterrows())):
            wk_sched = d["weekly_schedules"][
                d["weekly_schedules"].ID == _row.WEEKLY_SCHEDULE
            ]
            start_date = Date() if n == 0 else end_date
            end_date = Date(month=month_lookup[_row.END_MONTH], day=int(_row.END_DAY))
            for day in DAYS:
                day_schedule_series = d["daily_schedules"][
                    d["daily_schedules"].ID == wk_sched.squeeze()[day]
                ].squeeze()
                day_schedule_series = ScheduleDay.from_values_at_timestep(
                    identifier=clean_string(f"NCM_{day_schedule_series.NAME}_{day}"),
                    values=[
                        np.nan_to_num(day_schedule_series[f"h{i:02d}"], nan=100)
                        for i in range(24)
                    ],
                    timestep=1,
                ).duplicate()
                if day != "HOLIDAY":
                    rule = ScheduleRule(
                        **{
                            "schedule_day": day_schedule_series,
                            f"apply_{day.lower()}": True,
                            "end_date": end_date,
                            "start_date": start_date,
                        }
                    )
                    rules.append(rule)
                day_schedules[day] = day_schedule_series
        default_day = day_schedules["MONDAY"].duplicate()
        heating_schedule = ScheduleRuleset(
            identifier=clean_string(f"NCM_{row.NAME}_Heating_Schedule"),
            default_day_schedule=default_day,
            schedule_rules=rules,
            schedule_type_limit=temperature_type_limit,
        )

        cool_multiple = d["annual_weekly_schedules"][
            d["annual_weekly_schedules"].ANNUAL_SCHEDULE == row.COOL_SET_SCH
        ]
        end_date = None
        day_schedules = {}
        rules = []
        for n, (_, _row) in enumerate(list(cool_multiple.iterrows())):
            wk_sched = d["weekly_schedules"][
                d["weekly_schedules"].ID == _row.WEEKLY_SCHEDULE
            ]
            start_date = Date() if n == 0 else end_date
            end_date = Date(month=month_lookup[_row.END_MONTH], day=int(_row.END_DAY))

            for day in DAYS:
                day_schedule_series = d["daily_schedules"][
                    d["daily_schedules"].ID == wk_sched.squeeze()[day]
                ].squeeze()
                day_schedule_series = ScheduleDay.from_values_at_timestep(
                    identifier=clean_string(f"NCM_{day_schedule_series.NAME}_{day}"),
                    values=[
                        np.nan_to_num(day_schedule_series[f"h{i:02d}"], nan=100)
                        for i in range(24)
                    ],
                    timestep=1,
                ).duplicate()
                if day != "HOLIDAY":
                    rule = ScheduleRule(
                        **{
                            "schedule_day": day_schedule_series,
                            f"apply_{day.lower()}": True,
                            "end_date": end_date,
                            "start_date": start_date,
                        }
                    )
                    rules.append(rule)
                day_schedules[day] = day_schedule_series
        default_day = day_schedules["MONDAY"].duplicate()
        cooling_schedule = ScheduleRuleset(
            identifier=clean_string(f"NCM_{row.NAME}_Cooling_Schedule"),
            default_day_schedule=default_day,
            schedule_rules=rules,
            schedule_type_limit=temperature_type_limit,
        )

        humidifying_schedule = ScheduleRuleset.from_constant_value(
            identifier=clean_string(f"NCM_{row.NAME}_Humidifying_Schedule"),
            value=row.HUM_MIN,
        )
        dehumidifying_schedule = ScheduleRuleset.from_constant_value(
            identifier=clean_string(f"NCM_{row.NAME}_Dehumidifying_Schedule"),
            value=row.HUM_MAX,
        )

        setpoint = Setpoint(
            identifier=clean_string(f"NCM_{row.NAME}_Setpoint"),
            cooling_schedule=cooling_schedule,
            heating_schedule=heating_schedule,
            humidifying_schedule=humidifying_schedule,
            dehumidifying_schedule=dehumidifying_schedule,
        )
        schedules.extend([cooling_schedule, heating_schedule, humidifying_schedule, dehumidifying_schedule])
        # endregion

        # region: PEOPLE
        occ_multiple = d["annual_weekly_schedules"][
            d["annual_weekly_schedules"].ANNUAL_SCHEDULE == row.OCCUPANCY_SCH
        ]
        end_date = None
        day_schedules = {}
        rules = []
        for n, (_, _row) in enumerate(list(occ_multiple.iterrows())):
            wk_sched = d["weekly_schedules"][
                d["weekly_schedules"].ID == _row.WEEKLY_SCHEDULE
            ]
            start_date = Date() if n == 0 else end_date
            end_date = Date(month=month_lookup[_row.END_MONTH], day=int(_row.END_DAY))
            for day in DAYS:
                day_schedule_series = d["daily_schedules"][
                    d["daily_schedules"].ID == wk_sched.squeeze()[day]
                ].squeeze()
                day_schedule_series = ScheduleDay.from_values_at_timestep(
                    identifier=clean_string(f"NCM_{day_schedule_series.NAME}_{day}"),
                    values=[
                        np.nan_to_num(day_schedule_series[f"h{i:02d}"], nan=100)
                        for i in range(24)
                    ],
                    timestep=1,
                ).duplicate()
                if day != "HOLIDAY":
                    rule = ScheduleRule(
                        **{
                            "schedule_day": day_schedule_series,
                            f"apply_{day.lower()}": True,
                            "end_date": end_date,
                            "start_date": start_date,
                        }
                    )
                    rules.append(rule)
                day_schedules[day] = day_schedule_series
        default_day = day_schedules["MONDAY"].duplicate()
        occupancy_schedule = ScheduleRuleset(
            identifier=clean_string(f"NCM_{row.NAME}_Schedule"),
            default_day_schedule=default_day,
            schedule_rules=rules,
        )
        activity_schedule = ScheduleRuleset.from_constant_value(
            identifier=clean_string(f"NCM_{row.NAME}_Activity_Schedule"),
            value=row.METABOLIC_RATE,
        )
        people = People(
            identifier=clean_string(f"NCM_{row.NAME}_People"),
            people_per_area=row.OCCUPANCY_DENS,
            occupancy_schedule=occupancy_schedule,
            latent_fraction=row.OCCUP_PERC_LAT / 100,
            activity_schedule=activity_schedule,
        )
        schedules.extend([occupancy_schedule, activity_schedule])
        # endregion

        # region: LIGHTING
        temp = d["annual_weekly_schedules"][
            d["annual_weekly_schedules"].ANNUAL_SCHEDULE == row.LIGHTING_SCH
        ]
        end_date = None
        day_schedules = {}
        rules = []
        for n, (_, _row) in enumerate(list(temp.iterrows())):
            wk_sched = d["weekly_schedules"][
                d["weekly_schedules"].ID == _row.WEEKLY_SCHEDULE
            ]
            start_date = Date() if n == 0 else end_date
            end_date = Date(month=month_lookup[_row.END_MONTH], day=int(_row.END_DAY))
            for day in DAYS:
                day_schedule_series = d["daily_schedules"][
                    d["daily_schedules"].ID == wk_sched.squeeze()[day]
                ].squeeze()
                day_schedule_series = ScheduleDay.from_values_at_timestep(
                    identifier=clean_string(f"NCM_{day_schedule_series.NAME}_{day}"),
                    values=[
                        np.nan_to_num(day_schedule_series[f"h{i:02d}"], nan=100)
                        for i in range(24)
                    ],
                    timestep=1,
                ).duplicate()
                if day != "HOLIDAY":
                    rule = ScheduleRule(
                        **{
                            "schedule_day": day_schedule_series,
                            f"apply_{day.lower()}": True,
                            "end_date": end_date,
                            "start_date": start_date,
                        }
                    )
                    rules.append(rule)
                day_schedules[day] = day_schedule_series
        default_day = day_schedules["MONDAY"].duplicate()
        lighting_schedule = ScheduleRuleset(
            identifier=f"NCM_{clean_string(row.NAME)}_Lighting_Schedule",
            default_day_schedule=default_day,
            schedule_rules=rules,
        )
        if row.LIGHTING_DISPLAY == 0 and row.LIGHTING_LUX != 0:
            luminous_efficacy = 100
            lighting = Lighting(
                identifier=clean_string(f"NCM_{row.NAME}_Lighting"),
                watts_per_area=np.nan_to_num(row.LIGHTING_LUX, nan=0)
                / luminous_efficacy,
                schedule=lighting_schedule,
            )
        else:
            lighting = Lighting(
                identifier=clean_string(f"NCM_{row.NAME}_Lighting"),
                watts_per_area=np.nan_to_num(row.LIGHTING_DISPLAY, nan=0),
                schedule=lighting_schedule,
            )
        schedules.extend([lighting_schedule])
        # endregion

        # region: EQUIPMENT
        equip = d["annual_weekly_schedules"][
            d["annual_weekly_schedules"].ANNUAL_SCHEDULE == row.EQUIPMENT_SCH
        ]
        end_date = None
        day_schedules = {}
        rules = []
        for n, (_, _row) in enumerate(list(equip.iterrows())):
            wk_sched = d["weekly_schedules"][
                d["weekly_schedules"].ID == _row.WEEKLY_SCHEDULE
            ]
            start_date = Date() if n == 0 else end_date
            end_date = Date(month=month_lookup[_row.END_MONTH], day=int(_row.END_DAY))
            for day in DAYS:
                day_schedule_series = d["daily_schedules"][
                    d["daily_schedules"].ID == wk_sched.squeeze()[day]
                ].squeeze()
                day_schedule_series = ScheduleDay.from_values_at_timestep(
                    identifier=clean_string(f"NCM_{day_schedule_series.NAME}_{day}"),
                    values=[
                        np.nan_to_num(day_schedule_series[f"h{i:02d}"], nan=100)
                        for i in range(24)
                    ],
                    timestep=1,
                ).duplicate()
                if day != "HOLIDAY":
                    rule = ScheduleRule(
                        **{
                            "schedule_day": day_schedule_series,
                            f"apply_{day.lower()}": True,
                            "end_date": end_date,
                            "start_date": start_date,
                        }
                    )
                    rules.append(rule)
                day_schedules[day] = day_schedule_series
        default_day = day_schedules["MONDAY"].duplicate()
        equipment_schedule = ScheduleRuleset(
            identifier=clean_string(f"NCM_{row.NAME}_Equipment_Schedule"),
            default_day_schedule=default_day,
            schedule_rules=rules,
        )
        equipment = ElectricEquipment(
            identifier=clean_string(f"NCM_{row.NAME}_Equipment"),
            watts_per_area=np.nan_to_num(row.EQUIPMENT_W_M2, nan=0),
            schedule=equipment_schedule,
            latent_fraction=np.nan_to_num(row.EQUIP_PERC_LAT, nan=0) / 100,
        )
        schedules.extend([equipment_schedule])
        # endregion

        # region: VENTILATION
        ventilation = Ventilation(
            identifier=clean_string(f"NCM_{row.NAME}_Ventilation"),
            flow_per_person=np.nan_to_num(row.OA_FLOW_PERSON, nan=0) * 0.001,
            schedule=occupancy_schedule,
        )
        # endregion

        # region: HOTWATER
        # TODO - distribute hot water usage over the day according to occupancy, not every hour
        service_hot_water = ServiceHotWater(
            identifier=clean_string(f"NCM_{row.NAME}_HotWater"),
            flow_per_area=np.nan_to_num(row.HWS, nan=0) / 24,
            schedule=occupancy_schedule,
        )
        # endregion

        program = ProgramType(
            identifier=clean_string(f"NCM_{row.NAME}"),
            electric_equipment=equipment,
            lighting=lighting,
            people=people,
            service_hot_water=service_hot_water,
            setpoint=setpoint,
            ventilation=ventilation,
        )
        programs.append(program)

    # write to custom_standards folder
    with open(
        DATA_PATH / "custom_standards" / "programtypes" / "NCM_programs.json", "w"
    ) as f:
        json.dump({prog.identifier: prog.to_dict() for prog in programs}, f, indent=4)

    with open(
        DATA_PATH / "custom_standards" / "schedules" / "NCM_schedules.json", "w"
    ) as f:
        json.dump({schd.identifier: schd.to_dict() for schd in schedules}, f, indent=4)

    return programs


if __name__ == "__main__":
    create_ncm_programs_from_mdb()
    create_ncm_constructions_from_mdb()
    create_ncm_glazings_from_mdb()
    mirror_directory(DATA_PATH / "custom_standards", folders.default_standards_folder)
