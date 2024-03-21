function layout_data(data::Vector{<:AbstractArray}, arranger::DataArranger)
    ad = arranger.(SingleTimeseriesLayout() => SingleArrayLayout(), data)
    return ad
end

function default_data_loader(list::String="list_train";
                             device::Lux.LuxDeviceUtils.AbstractLuxDevice,
                             max_ori_error=20.0,
                             grv_only=false,
                             ground_truth_window=200,
                             θ_features=(stride=10, window=200, pad=10),
                             θ_targets=(stride=10, window=1, pad=10),
                             batchsize=128,
                             shuffle=true)
    filenames = fetch_filenames(list)
    data = Dataset.(filenames; max_ori_error, grv_only, window=ground_truth_window)
    Xs = layout_data(getfield.(data, :features), slidingwindow(Float32; θ_features...))
    Ys = layout_data(getfield.(data, :targets), slidingwindow(Float32; θ_targets...))
    X = mergealonglastdim(Xs) |> device
    Y = mergealonglastdim(Ys) |> device
    return DataLoader((X, Y); batchsize, shuffle)
end

# Copied from the source code of `Flux.jl`
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y))
        size(ŷ, d) == size(y, d) ||
            throw(DimensionMismatch("loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"))
    end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1

function mse(ŷ, y; agg=mean)
    _check_sizes(ŷ, y)
    return agg(abs2.(ŷ .- y))
end