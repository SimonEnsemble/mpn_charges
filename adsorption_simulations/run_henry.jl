using PorousMaterials

charge_assignment_method = ARGS[2]
@eval PorousMaterials PATH_TO_CRYSTALS = charge_assignment_method * "_xtals"

crystal_name = ARGS[1]
xtal = Crystal(crystal_name)
molecule = Molecule("CO2")
strip_numbers_from_atom_labels!(xtal)
temp = 298.0
ljforcefield = LJForceField("UFF", r_cutoff=12.5)

# run the simulation
result = henry_coefficient(xtal, molecule, temp, ljforcefield,
					   insertions_per_volume=1000, verbose=true, filename_comment=charge_assignment_method)
