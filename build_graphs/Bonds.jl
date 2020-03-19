module Bonds

using PorousMaterials, CSV, LightGraphs, PyCall, LinearAlgebra
 # scipy = pyimport("scipy.spatial")

const scipy = PyNULL()

# https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules
function __init__()
    copy!(scipy, pyimport("scipy.spatial"))
end

function cordero_covalent_atomic_radii()
    df = CSV.read("covalent_radii.csv", comment="#")
    atom_to_radius = Dict{Symbol, Float64}()
    for atom in eachrow(df)
        atom_to_radius[Symbol(atom[:atom])] = atom[:covalent_radius_A]
    end
    # Change the cordero covalent atomic radius of a few metals to fix bonds between 
    #  metals and (mostly) N-terminated and O-terminated linkers
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
    return atom_to_radius
end

const RADII = cordero_covalent_atomic_radii()

"""
    a = adjacency_matrix(crystal, apply_pbc)

Compute the adjacency matrix `a` of the crystal, where `a[i, j]` is the
distance between atom `i` and `j`. This matrix is symmetric. If `apply_pbc` is `true`, 
periodic boundary conditions are applied when computing the distance.

# Arguments
* `crystal::Crystal`: crystal structure
* `apply_pbc::Bool` whether or not to apply periodic boundary conditions when computing the distance

# Returns
* `a::Array{Float64, 2}`: symmetric, square adjacency matrix with zeros on the diagonal
"""
function adjacency_matrix(crystal::Crystal, apply_pbc::Bool)
    A = zeros(crystal.atoms.n, crystal.atoms.n)
    for i = 1:crystal.atoms.n
        for j = (i+1):crystal.atoms.n
            A[i, j] = distance(crystal.atoms, crystal.box, i, j, apply_pbc)
            A[j, i] = A[i, j] # symmetry
        end
    end
    # the diagonal is zeros
    return A
end

"""
    ids_neighbors, xs, rs = neighborhood(crystal, i, r, am)

Find and characterize the neighborhood of atom `i` in the crystal `crystal`.
A neighboorhood is defined as all atoms within a distance `r` from atom `i`.
The adjacency matrix `am` is used to find the distances of all other atoms in the crystal from atom `i`.

# Returns
* `ids_neighbors::Array{Int, 1}`: indices of `crystal.atoms` within the neighborhood of atom `i`.
* `xs::Array{Array{Float64, 1}, 1}`: array of Cartesian positions of the atoms surrounding atom `i`. 
the nearest image convention has been applied to find the nearest periodic image. also, the coordinates of atom `i`
have been subtracted off from these coordinates so that atom `i` lies at the origin of this new coordinate system.
The first vector in `xs` is `[0, 0, 0]` corresponding to atom `i`.
the choice of type is for the Voronoi decomposition in Scipy. 
* `rs::Array{Float64, 1}`: list of distances of the neighboring atoms from atom `i`.
"""
function neighborhood(crystal::Crystal, i::Int, r::Float64, am::Array{Float64, 2})
    # get indices of atoms within a distance r of atom i
    #  the greater than zero part is to not include itself
    ids_neighbors = findall((am[:, i] .> 0.0) .& (am[:, i] .< r))

    # rs is the list of distance of these neighbors from atom i
    rs = [am[i, id_n] for id_n in ids_neighbors]
    @assert all(rs .< r)

    # xs is a list of Cartesian coords of the neighborhood
    #   coords of atom i are subtracted off
    #   first entry is coords of atom i, the center, the zero vector
    #   remaining entries are neighbors
    # this list is useful to pass to Voronoi for getting Voronoi faces
    #    of the neighborhood.
    xs = [[0.0, 0.0, 0.0]] # this way atom zero is itself
    for j in ids_neighbors
        # subtract off atom i, apply nearest image
        xf = crystal.atoms.coords.xf[:, j] - crystal.atoms.coords.xf[:, i]
        nearest_image!(xf)
        x = crystal.box.f_to_c * xf
        push!(xs, x)
    end

    return ids_neighbors, xs, rs
end

"""
    ids_shared_voro_face = shared_voronoi_faces(ids_neighbors, xs)

Of the neighboring atoms, find those that share a Voronoi face.
Returns ids in the original crystal passed to `neighborhood`
"""
function _shared_voronoi_faces(ids_neighbors::Array{Int, 1},
                              xs::Array{Array{Float64, 1}, 1})
    # first element of xs is the point itself, the origin
    @assert(length(ids_neighbors) == length(xs) - 1)

    voro = scipy.Voronoi(xs)
    rps = voro.ridge_points # connections with atom zero are connections with atom i.
    ids_shared_voro_face = Int[] # corresponds to xs, not to atoms of crystal
    for k = 1:size(rps)[1]
        if sort(rps[k, :])[1] == 0 # a shared face with atom i!
            push!(ids_shared_voro_face, sort(rps[k, :])[2])
        end
    end
    # zero based indexing in Scipy accounted for since xs[0] is origin, atom i.
    return ids_neighbors[ids_shared_voro_face]
end

function bonded_atoms(crystal::Crystal, i::Int, am::Array{Float64, 2}; r::Float64=6.0, tol::Float64=0.25)
    species_i = crystal.atoms.species[i]
    
    ids_neighbors, xs, rs = neighborhood(crystal, i, r, am)

    ids_shared_voro_faces = _shared_voronoi_faces(ids_neighbors, xs)

    ids_bonded = Int[]
    for j in ids_shared_voro_faces
        species_j = crystal.atoms.species[j]
        # sum of covalent radii
        radii_sum = RADII[species_j] + RADII[species_i]
        if am[i, j] < radii_sum + tol
            push!(ids_bonded, j)
        end
    end
    return ids_bonded
end

function bonds!(crystal::Crystal, apply_pbc::Bool)
    if ne(crystal.bonds) > 0
        @warn crystal.name * " already has bonds"
    end
    am = adjacency_matrix(crystal, apply_pbc)

    for i = 1:crystal.atoms.n
        for j in bonded_atoms(crystal, i, am)
            add_edge!(crystal.bonds, i, j)
        end
    end
end

export bonds!

end
