using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUResearchRonin", "examples"))
using Revise

using Dates
using IMUDevNNTrainingLib
using IMUResearchRonin
using Lux
using LuxCUDA
using Optimisers
using MLUtils
using Random
using ParameterSchedulers
using ParameterSchedulers: Stateful
using ProgressMeter
using StatsBase
using Zygote

include(joinpath(@__DIR__, "..", "utils.jl"))
include(joinpath(@__DIR__, "..", "training_common.jl"))

USE_GPUs = true
VALIDATE = true

rng = Xoshiro(0)
device = USE_GPUs ? Lux.gpu_device() : Lux.cpu_device()
WINDOW = (features=400, target=300, ground_truth=100)
RANDOM_SHIFT = 100
train_loader = default_data_loader("list_train";
                                   device,
                                   ground_truth_window=WINDOW.ground_truth,
                                   θ_features=(stride=200, window=WINDOW.features,
                                               pad=RANDOM_SHIFT),
                                   θ_targets=(stride=200, window=WINDOW.target,
                                              pad=RANDOM_SHIFT),
                                   batchsize=64)
val_loader = default_data_loader("list_val";
                                 device,
                                 ground_truth_window=WINDOW.ground_truth,
                                 θ_features=(stride=200, window=WINDOW.features,
                                             pad=RANDOM_SHIFT),
                                 θ_targets=(stride=200, window=WINDOW.target,
                                            pad=RANDOM_SHIFT),
                                 batchsize=256)

model = Chain(StatefulRecurrentCell(LSTMCell(6 => 100)),
              StatefulRecurrentCell(LSTMCell(100 => 100)),
              StatefulRecurrentCell(LSTMCell(100 => 100)),
              Dense(100 => 10),
              Dense(10 => 2))

chkp = Checkpointer(; dir=joinpath(outdir(), "lstm"),
                    continue_from=:last,
                    save_every=1)
(state, log, plateau_detector,
start_epoch) = default_checkpoint_loader(chkp;
                                         rng,
                                         model,
                                         move_to_device=device,
                                         init_learning_rate=Float32(3e-3),
                                         patience=20);

layout = SingleArrayLayout()
transformations = (random_shift=RandomShift(RANDOM_SHIFT),
                   random_rotation=RandomPlanarRotation(2π, 1:2))

function preprocess_data(input, labels)
    input, labels = transformations.random_shift(layout, input, labels)
    gyr, acc = @view(input[1:3, :, :]), @view(input[4:6, :, :])
    gyr, acc, labels = transformations.random_rotation(layout, gyr, acc, labels)
    input = cat(gyr, acc; dims=1)
    input = reshape(input, layout => TimeseriesLayout())
    labels = reshape(labels, layout => TimeseriesLayout())
    return input, labels
end

function computeloss(model, parameters, states, (x, y))
    states = Lux.update_state(states, :carry, nothing)
    for i in 1:(WINDOW.ground_truth)
        _, states = model(x[i], parameters, states)
    end

    total = 0.0
    for i in 1:(WINDOW.target)
        y°, states = model(x[i + WINDOW.ground_truth], parameters, states)
        total += mse(y°, y[i])
    end
    return total / WINDOW.target, states, (;)
end

start_info(layout, train_loader)

for epoch in start_epoch:100
    state, train_log = trainstep(; epoch,
                                 data_loader=train_loader,
                                 preprocess_data,
                                 computeloss,
                                 state,
                                 processed_data_layout=TimeseriesLayout(),
                                 plateau_detector)
    if VALIDATE
        val_log = validationstep(; epoch,
                                 data_loader=val_loader,
                                 preprocess_data,
                                 computeloss,
                                 state,
                                 processed_data_layout=TimeseriesLayout())
        push!(log, (; train=train_log, val=val_log))
    else
        push!(log, (; train=train_log))
    end

    checkpoint(chkp, epoch; state, log, plateau_detector)

    if train_log.avg_loss < 1e-3
        println("stopping after $epoch epochs")
        break
    end
end