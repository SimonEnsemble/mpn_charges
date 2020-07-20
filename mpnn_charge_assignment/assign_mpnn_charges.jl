using PorousMaterials, CSV, NPZ, Printf, LightGraphs # use major_refactor branch

@eval PorousMaterials PATH_TO_DATA = pwd() # for covalent_radii.csv

# covalent radii
atom_to_radius = cordero_covalent_atomic_radii()
atom_to_radius[:Cd] = 1.58
atom_to_radius[:Zn] = 1.48
atom_to_radius[:Ag] = 1.65
atom_to_radius[:Pb] = 1.90
atom_to_radius[:Mo] = 1.57
atom_to_radius[:Sn] = 1.48
atom_to_radius[:Na] = 1.72
atom_to_radius[:Cu] = 1.46
atom_to_radius[:Ni] = 1.37
atom_to_radius[:Cr] = 1.41

if length(ARGS) != 1
    error("run as `julia assign_mpnn_charges.jl my_mof.cif` for a single MOF or `julia assign_mpnn_charges.jl mof_directory` for a directory containing MOFs.")
end

# function which reads in cif, assigns charges and write a new cif file with MPNN charges
function assign_charges(xtal_name::AbstractString)
    printstyled("assigning MPNN charges to ", xtal_name, "\n", color=:green)

    # read in xtal structure
    printstyled("\treading crystal structure file...\n", color=:yellow)
    xtal = Crystal(xtal_name)
    strip_numbers_from_atom_labels!(xtal)

    # define filename for edges and node features
    node_feature_filename = joinpath("temp", xtal_name * "_node_features.npy")
    edge_filename = joinpath("temp", xtal_name * ".edge_info")
    if ! isdir("temp")
        mkdir("temp")
    end

    # read in mapping from atomic species to integer for one-hot encoding
    df = CSV.read("../atom_to_int.csv")
    atom_to_int = Dict{Symbol, Int}()
    for row in eachrow(df)
        atom_to_int[Symbol(row[:first])] = row[:second]
    end

    # ensure atoms are supported
    printstyled("\tensuring atoms are supported...\n", color=:yellow)
    unique_atoms = unique(xtal.atoms.species)
    for atom in unique_atoms
        if ! (atom in keys(atom_to_int))
            error("$atom not supported.\n")
        end
    end
    rare_atoms = [:Hf, :Se, :Ir, :Pu, :Cs]
    for rare_atom in rare_atoms
        if rare_atom in unique_atoms
            error("$xtal_name contains $rare_atom, not supported (see our paper)\n")
        end
    end

    # infer bonds
    printstyled("\tinferring bonds...\n", color=:yellow)
    infer_geometry_based_bonds!(xtal, true, covalent_radii=atom_to_radius) # true for inferring bonds across the periodic boundary.
    if ! bond_sanity_check(xtal)
        error("bond graph is chemically invalid... please check your .cif\n")
    end

    # node embedding
    printstyled("\twriting node feature matrix and edge information to file...\n", color=:yellow)
    x_ν = zeros(Int, xtal.atoms.n, length(atom_to_int))
    for (i, atom) in enumerate(xtal.atoms.species)
        x_ν[i, atom_to_int[atom]] = 1
    end
    @assert sum(x_ν) == xtal.atoms.n
    npzwrite(node_feature_filename, x_ν)

    ###
    #   edges
    #   (a list and their feature = distance btwn atoms)
    ###
    edge_file = open(edge_filename, "w")
    @printf(edge_file, "src,dst,r\n")
    for ed in edges(xtal.bonds)
        i = src(ed) # source
        j = dst(ed) # destination
        r = distance(xtal.atoms, xtal.box, i, j, true)
        @printf(edge_file, "%d,%d,%f\n", i - 1, j - 1, r)
    end
    close(edge_file)

    ###
    #   python code to load in MPNN model, read in edge and node feature matrix
    #   written from the script
    #   Outputs MPNN charges
    ###
    run(`python deployment_main.py "temp" $xtal_name`)

    ###
    #   Loading in the MPNN charges and assigning them to the MOF
    #   and writing the MOF to a new `.cif` file.
    ###
    mpnn_charge_values = npzread(joinpath(pwd(), "temp", xtal_name * "_mpnn_charges.npy"))
    mpnn_charges = Charges(Float64.(mpnn_charge_values), xtal.atoms.coords)
    new_xtal = Crystal(xtal.name, xtal.box, xtal.atoms, mpnn_charges, xtal.bonds, xtal.symmetry)
    #remove_bonds!(new_xtal) # If you don't want bonds in the `.cif` file
    write_cif(new_xtal, joinpath(PorousMaterials.PATH_TO_CRYSTALS, split(xtal.name, ".")[1] * "_mpnn_charges.cif"))

    ###
    #   Deleting the temporary folder for cleanup
    ###
    run(`rm -rf temp`)
end



# directory where MOF xtal structures are stored
if isdir(ARGS[1])
    @printf("Reading in all MOFs in %s for MPNN charge assignment\n", ARGS[1])
    @eval PorousMaterials PATH_TO_CRYSTALS = ARGS[1]
    for xtal_name in readdir(ARGS[1])
        if split(xtal_name, ".")[end] != "cif" && split(xtal_name, ".")[end] != "cssr"
            @printf("%s does not have a `cif` or `cssr` file extension. Skipping...\n", xtal_name)
            continue
        end
        println(xtal_name)
        assign_charges.(xtal_name)
    end
else
    if split(ARGS[1], ".")[end] != "cif" && split(ARGS[1], ".")[end] != "cssr"
        @printf("%s does not have a `cif` or `cssr` file extension. Skipping...\n", ARGS[1])
    else
        @eval PorousMaterials PATH_TO_CRYSTALS = pwd()
        assign_charges(ARGS[1])
    end
end
