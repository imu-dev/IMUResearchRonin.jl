"""
    read_info_from_file(path)

Read the `info.json` file correspodning to a single HDF5 data file.
"""
read_info_from_file(path) =
    open(joinpath(path, "info.json")) do file
        return JSON.parse(read(file, String))
    end

"""
    read_data_from_file(path)

Read the `data.hdf5` file with data.
"""
read_data_from_file(path) =
    h5open(joinpath(path, "data.hdf5"), "r") do file
        return read(file)
    end

path_to_examples_dir() = joinpath(dirname(pathof(IMUResearchRonin)), "..", "examples")

"""Path to the output directory"""
function outdir(path_to_env_file=joinpath(path_to_examples_dir(), ".env.toml"))
    return open(path_to_env_file) do io
        return TOML.parse(io)["out"]
    end
end

"""Fetch acc/gyr data + labels from a list of files"""
function fetch_filenames(list::String="list_train";
                         path_to_env_file=joinpath(path_to_examples_dir(), ".env.toml"))
    locs = open(path_to_env_file) do io
        return TOML.parse(io)
    end
    return fetch_filenames(locs["root"], locs[list])
end

"""
    fetch_filenames(root_dir, file_with_filelist)

Read a `file_with_filelist` and return a list of paths pointing to files with
data. It is assumed that all data are found in `root_dir` and that
`file_with_filelist` contains a newline-delimited list of filenames relative to
`root_dir`. Lines starting with `#` are ignored.
"""
function fetch_filenames(root_dir, file_with_filelist)
    fs = open(file_with_filelist) do file
        return filter(l -> !startswith(l, "#"), readlines(file))
    end
    return joinpath.(root_dir, fs)
end

# """Take in the data and slice it and dice it for training with Neural Nets"""
# function layout_data(ds::AbstractVector{<:Dataset}, ar::StridedDataArranger;
#                      move_to_gpu=false, batchsize=128, shuffle=true)
#     adata = ar.(ds)
#     X = cat(first.(adata)...; dims=3)
#     y = mapreduce(x -> getindex(x, 2), hcat, adata)
#     λ(dataset_id, frame_ids) = map(id -> (; dataset_id, frame_id=id), frame_ids)
#     ids = mapreduce(x -> λ(x[1], getindex(x[2], 3)),
#                     vcat,
#                     enumerate(adata))

#     if move_to_gpu
#         return Flux.DataLoader((X, y) |> gpu; batchsize, shuffle), ids
#     end
#     return Flux.DataLoader((X, y); batchsize, shuffle), ids
# end

# array2vec_of_elem(a::AbstractArray; dims=1) = [copy(x) for x in eachslice(a; dims)]

# function warmup_train_split(X::AbstractVector, gt_window)
#     X_warmup = @view X[1:gt_window]
#     X_train = @view X[(gt_window + 1):end]
#     return X_warmup, X_train
# end
