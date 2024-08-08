import sys

sys.path.insert(0, './masterplanenergydemand')


from masterplanenergydemand.mped import Masterplan, Typology

# load from excel
mped: Masterplan = Masterplan.from_excel(
    excel_file=r"C:\Users\tgerrish\Documents\GitHub\BHoM\LadyBugTools_Prototypes\LadyBugTools_Prototype_Engine\Python\MasterplanEnergyDemand\masterplanenergydemand\test\test.xlsx",
    sheet_name="TestMped",
    use_defaults=True,
)

# get something
mped.typologies[0]._lift_energy_demand()