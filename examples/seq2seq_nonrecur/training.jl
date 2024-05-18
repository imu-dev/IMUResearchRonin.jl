using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUResearchRonin", "examples"))
using Revise

using Dates
using IMUDevNNArchitectures
using IMUDevNNArchitectures.TCN
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
model = Chain(TCN.tcn(6 => 36;
                      channels=[32, 64, 128, 256, 72],
                      kernel_size=3,
                      dropout=0.2),
              Conv((1,), 36 => 2),
              Dropout(0.2))
chkp = Checkpointer(; dir=joinpath(outdir(), "tcn"),
                    continue_from=:last,
                    save_every=5)
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
    labels = permutedims(labels, (2, 1, 3))
    input = permutedims(input, (2, 1, 3))
    return input, labels
end

function computeloss(model, parameters, states, (x, y))
    y°, states = model(x, parameters, states)
    y° = @view y°[(WINDOW.ground_truth + 1):end, :, :]
    return mse(y°, y), states, (; y_pred=y°)
end

start_info(layout, train_loader)

for epoch in start_epoch:10
    state, train_log = trainstep(; epoch,
                                 data_loader=train_loader,
                                 preprocess_data,
                                 computeloss,
                                 state,
                                 processed_data_layout=SingleArrayLayout(),
                                 plateau_detector)

    if VALIDATE
        val_log = validationstep(; epoch,
                                 data_loader=val_loader,
                                 preprocess_data,
                                 computeloss,
                                 state,
                                 processed_data_layout=SingleArrayLayout())
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