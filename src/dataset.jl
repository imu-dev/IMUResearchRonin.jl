"""
    Dataset

The main struct that holds the data.

# Constructors

    Dataset(path; grv_only=false, max_ori_error=20.0, window=200)

Create a new `Dataset` instance from the data stored in the directory `path`.
"""
struct Dataset
    """Time vector"""
    tt::Vector{Float64}

    """Input features. Shape: (feature_dim, n_samples)."""
    features::Matrix{Float64}

    """Target values. Shape: (target_dim, n_samples)."""
    targets::Matrix{Float64}

    """
    Orientations. Shape: (4, n_samples). Derived.
    """
    orientations::Matrix{Float64}

    """
    (Ground truth) pose. Shape: (3, n_samples). Computed based on Tango
    recording.
    """
    gt_pos::Matrix{Float64}

    """Metainformation about the dataset."""
    info::Dict{String,Any}

    """
    If `true`, the data collected under the name `game_rv` are going to be used
    as the orientation source. In particular, no search for the best orientation
    source is performed.
    """
    grv_only::Bool

    """
    If the orientation error of the `game_rv` field is lower than this value, 
    then the data collected under the name `game_rv` are going to be used
    as the orientation source irrespective of which source has the smallest
    error.
    """
    max_ori_error::Float64

    """
    The size of the window based on which the mean velocity is computed for the
    ground truth.

    !!! tip
        When intended learning is done in a format:
        - sequence to point: `window` of input features is used to predict the
                            mean velocity for the interval, then this `window`
                            should be equal to the `window` used for creating
                            batches for learning.
        - sequence to sequence: `window` of input features is used to predict a
                                sequence of corresponding smoothed instantenous
                                velocity values, then this `window` should be
                                some value <= `window` used for creating batches
                                for learning.
    """
    window::Int

    function Dataset(path; grv_only=false, max_ori_error=20.0, window=200)
        info = read_info_from_file(path)
        info["path"] = splitpath(path)[end]
        info["ori_source"], ori, info["source_ori_error"] = select_orientation_source(path;
                                                                                      grv_only,
                                                                                      max_ori_error)
        data = read_data_from_file(path)
        gyro = data["synced"]["gyro_uncalib"] .- info["imu_init_gyro_bias"]
        acce = info["imu_acce_scale"] .* (data["synced"]["acce"] .- info["imu_acce_bias"])
        tt = data["synced"]["time"]
        tango_pos = data["pose"]["tango_pos"]
        init_tango_ori = quat(data["pose"]["tango_ori"][:, 1]...)

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = mat2quats(ori)
        rot_imu_to_tango = quat(info["start_calibration"]...)
        init_rotor = init_tango_ori * rot_imu_to_tango * conj(ori_q[1])
        ori_q = init_rotor .* ori_q

        δt = tt[(1 + window):end] .- tt[1:(end - window)]
        glob_v = (tango_pos[:, (1 + window):end] .- tango_pos[:, 1:(end - window)]) ./
                 δt'

        gyro_q = mat2quats(threedimmat2quatmat(gyro))
        acce_q = mat2quats(threedimmat2quatmat(acce))

        glob_gyro = mat(ori_q .* gyro_q .* conj(ori_q))[2:end, :]
        glob_acce = mat(ori_q .* acce_q .* conj(ori_q))[2:end, :]

        start_frame = get(info, "start_frame", 0) + 1
        # we need to trim the last index, because we don't have ground truth for it.
        # Note that the ground truth comes from position, which is defined on windows
        # of length `window + 1` (from which velocity is derived for a window of
        # length `window`)
        θ = (tt=tt[start_frame:end],
             features=vcat(glob_gyro, glob_acce)[:, start_frame:(end - 1)],
             targets=glob_v[1:2, start_frame:end],
             orientations=mat(ori_q)[:, start_frame:(end - 1)],
             gt_pos=tango_pos[:, start_frame:(end - 1)],
             info,
             grv_only,
             max_ori_error,
             window)
        return new(θ...)
    end
end

num_trainable_timepoints(d::Dataset) = size(d.targets, 2)

feature_dim(::Dataset) = 6
target_dim(::Dataset) = 2
aux_dim(::Dataset) = 8

number_of_obs(d::Dataset) = size(d.targets, 2)
Base.length(d::Dataset) = number_of_obs(d)
number_of_obs(ds::AbstractVector{<:Dataset}) = sum(number_of_obs, ds)

mat2quats(m) = map(c -> quat(c...), eachcol(m))
threedimmat2quatmat(m) = vcat(ones(1, size(m, 2)), m)

Base.vec(q::Quaternion) = [q.s, q.v1, q.v2, q.v3]
mat(qs::Vector{<:Quaternion}) = mapreduce(vec, hcat, qs)

features(d::Dataset) = d.features
targets(d::Dataset) = d.targets
aux(d::Dataset) = vcat(d.tt', d.orientations, d.gt_pos)

function meta(d::Dataset)
    p = d.info["path"]
    dev = d.info["device"]
    es = d.info["ori_source"]
    e = round(d.info["source_ori_error"]; digits=3)
    return "$p: device: $dev, ori_error ($es): $e"
end

function Base.show(io::IO, mime::MIME"text/plain", d::Dataset)
    println(io, "Dataset")
    println(io, "-------")
    println(io, "\ntt:")
    show(io, mime, d.tt)
    println(io)
    println(io, "\nfeatures:")
    show(io, mime, d.features)
    println(io)
    println(io, "\ntargets:")
    show(io, mime, d.targets)
    println(io)
    println(io, "\norientations:")
    show(io, mime, d.orientations)
    println(io)
    println(io, "\ngt_pos:")
    show(io, mime, d.gt_pos)
    println(io)
    println(io, "\ninfo:")
    show(io, mime, d.info)
    println(io)
    println(io, "\ngrv_only: ", d.grv_only)
    println(io, "\nmax_ori_error: ", d.max_ori_error)
    println(io, "\nwindow: ", d.window)
    return nothing
end