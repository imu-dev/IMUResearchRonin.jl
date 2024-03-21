module IMUReplRonin

using HDF5
using JSON
using TOML
using Quaternions

include("io.jl")
include("preprocessing.jl")
include("dataset.jl")

export Dataset

export outdir, fetch_filenames

end
