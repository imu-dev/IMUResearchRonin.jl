using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUReplRonin", "examples"))
using Revise

using Dates
using IMUDevNNArchitectures
using IMUDevNNArchitectures.ResNet1d
using IMUDevNNTrainingLib
using IMUReplRonin
using Lux
using Optimisers
using MLUtils
using Random
using ParameterSchedulers
using ParameterSchedulers: Stateful
using ProgressMeter
using StatsBase
using Zygote

const NNTrLib = IMUDevNNTrainingLib

include(joinpath(@__DIR__, "..", "utils.jl"))

USE_GPUs = false
VALIDATE = true

rng = Random.default_rng()
Random.seed!(rng, 0)

device = if USE_GPUs
    Lux.gpu_device()
else
    Lux.cpu_device()
end

train_loader = default_data_loader("list_train"; device)
val_loader = default_data_loader("list_val"; device, batchsize=512)
model = resnet((200, 6) => 2, [2, 2, 2, 2];
               strides=[1, 2, 2, 2],
               base_plane=64, kernel_size=3,
               outputblock_builder=(in_out) -> ResNet1d.outputblock(in_out;
                                                                    hidden=512,
                                                                    dropout=0.5,
                                                                    transition_size=128))
chkp = Checkpointer(; dir=joinpath(outdir(), "resnet"),
                    continue_from=:last,
                    save_every=1)
chkp_data, start_epoch = start(chkp; move_to_device=device);
parameters, states, opt_state, log, plateau_detector = if isnothing(chkp_data)
    # Model
    ps, st = Lux.setup(rng, model)
    ResNet1d.zero_init!(model, ps)

    # Optimiser
    init_learning_rate = Float32(1e-4)
    opt = Optimisers.Adam(init_learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    # Plateau detector
    pd = PlateauDetector(; scheduler=Stateful(Exp(init_learning_rate, 0.8f0)))

    ps, st, opt_state, [], pd
else
    ps, st, opt_state, log, others = chkp_data
    pd = others[:plateau_detector]
    ps, st, opt_state, log, pd
end;

layout = SingleArrayLayout()
transformations = (random_shift=RandomShift(10),
                   random_rotation=RandomPlanarRotation(2π, 1:2))
function preprocess_data(input, labels)
    input, labels = transformations.random_shift(layout, input, labels)
    gyr, acc = @view(input[1:3, :, :]), @view(input[4:6, :, :])
    gyr, acc, labels = transformations.random_rotation(layout, gyr, acc, labels)
    input = cat(gyr, acc; dims=1)
    labels = nnflatten(labels)
    input = permutedims(input, (2, 1, 3))
    return input, labels
end

function computeloss(input, labels, model, parameters, states)
    y°, states = model(input, parameters, states)
    return mse(y°, labels), y°, states
end

function losssummary(losses, batch_sizes)
    has_inf_loss = any(isinf, losses)
    losses_and_batch_sizes = filter(isfinite ∘ first, collect(zip(losses, batch_sizes)))
    avg_loss = mean(first.(losses_and_batch_sizes), weights(last.(losses_and_batch_sizes)))
    return avg_loss, has_inf_loss
end

start_info(layout, train_loader)

for epoch in start_epoch:30
    t0 = now()

    # Training loop
    echo_epoch(epoch)
    losses = Float32[]
    batch_sizes = Int[]
    prog = ProgressMeter.Progress(length(train_loader); dt=1.0, desc="Training...")
    for (i, (input, labels)) in enumerate(train_loader)
        input, labels = preprocess_data(input, labels)

        # Pass the data through the network and compute the gradients
        (loss, y°, st), back = pullback(computeloss, input, labels, model, ps, st)

        # store the loss and discard the step in a degenerate case
        push!(losses, loss)
        push!(batch_sizes, num_samples(layout, input))
        if !isfinite(loss)
            @warn "loss is $loss on item $i" epoch
            # update the progress bar
            next!(prog)
            continue
        end

        # Fetch the previously computed gradients and update the parameters
        gs = back((one(loss), nothing, nothing))[4]
        opt_state, ps = Optimisers.update(opt_state, ps, gs)

        # Update learning rate if needed
        if NNTrLib.step!(plateau_detector, loss)
            Optimisers.adjust!(opt_state, plateau_detector)
        end

        # update the progress bar
        next!(prog)
    end

    avg_loss, has_inf_loss = losssummary(losses, batch_sizes)
    train_log = (; avg_loss, losses, has_inf_loss)
    echo_summary(; epoch, avg_loss, elapsed=now() - t0)

    if VALIDATE
        t0_val = now()
        losses = Float32[]
        batch_sizes = Int[]
        prog = ProgressMeter.Progress(length(val_loader); dt=1.0, desc="Validating...")
        for (input, labels) in val_loader
            input, labels = preprocess_data(input, labels)

            # Pass the data through the network
            (loss, y°, st) = computeloss(input, labels, model, ps, st)

            # store the loss
            push!(losses, loss)
            push!(batch_sizes, num_samples(layout, input))
            next!(prog)
        end
        avg_val_loss, has_inf_loss = losssummary(losses, batch_sizes)
        val_log = (; avg_loss=avg_val_loss, losses, has_inf_loss)
        echo_summary(; epoch, avg_loss=avg_val_loss, elapsed=now() - t0_val,
                     title="Validation")
        push!(log, (; train=train_log, val=val_log))
    else
        push!(log, (; train=train_log))
    end

    checkpoint(chkp, epoch; parameters=ps, states=st, opt_state, log, plateau_detector)

    if avg_loss < 1e-3
        println("stopping after $epoch epochs")
        break
    end
end