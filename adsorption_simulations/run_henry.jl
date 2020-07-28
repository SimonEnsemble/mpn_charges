using PorousMaterials

println(ARGS[1])
@eval PorousMaterials PATH_TO_CRYSTALS = ARGS[2] * "_xtals"

xtal = Crystal(ARGS[1])
molecule = Molecule("CO2")
strip_numbers_from_atom_labels!(xtal)
temp = 298.0
ljforcefield = LJForceField("UFF", r_cutoff=12.5)

# run the simulation
result = henry_coefficient(xtal, molecule, temp, ljforcefield,
					   insertions_per_volume=1000, verbose=true, filename_comment=ARGS[2])
