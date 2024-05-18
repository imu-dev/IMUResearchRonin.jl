unzip_pairs(xx) = (first.(xx), last.(xx))

function losssummary(losses, batch_sizes)
    has_inf_loss = any(isinf, losses)
    losses, batch_sizes = unzip_pairs(filter(isfinite ∘ first,
                                             collect(zip(losses, batch_sizes))))
    avg_loss = mean(losses, weights(batch_sizes))
    return avg_loss, has_inf_loss
end

function trainstep(; epoch,
                   data_loader::DataLoader,
                   preprocess_data::Function,
                   computeloss::Function,
                   state::Lux.Experimental.TrainState,
                   processed_data_layout,
                   plateau_detector)
    t0 = now()
    # Training loop
    echo_epoch(epoch)
    losses = Float32[]
    batch_sizes = Int[]
    prog = ProgressMeter.Progress(length(data_loader); dt=1.0, desc="Training...")
    for (i, (input, labels)) in enumerate(data_loader)
        input, labels = preprocess_data(input, labels)

        # Pass the data through the network and compute the gradients
        gs, loss, _, state = Lux.Experimental.compute_gradients(AutoZygote(), computeloss,
                                                                (input, labels), state)

        # store the loss and discard the step in a degenerate case
        push!(losses, loss)
        push!(batch_sizes, num_samples(processed_data_layout, input))
        if !isfinite(loss)
            @warn "loss is $loss on item $i in epoch $epoch"
            # update the progress bar
            next!(prog)
            continue
        end

        # Fetch the previously computed gradients and update the parameters
        state = Lux.Experimental.apply_gradients(state, gs)

        # Update learning rate if needed
        if plateau_detector(loss)
            @info "Plateau detected!"
            Optimisers.adjust!(state.optimizer_state, plateau_detector)
        end

        # update the progress bar
        next!(prog)
    end

    avg_loss, has_inf_loss = losssummary(losses, batch_sizes)
    train_log = (; avg_loss, losses, has_inf_loss)
    echo_summary(; epoch, avg_loss, elapsed=now() - t0)
    return state, train_log
end

function validationstep(; epoch,
                        data_loader::DataLoader,
                        preprocess_data::Function,
                        computeloss::Function,
                        state::Lux.Experimental.TrainState,
                        processed_data_layout)
    t0_val = now()
    losses = Float32[]
    batch_sizes = Int[]
    prog = ProgressMeter.Progress(length(data_loader); dt=1.0, desc="Validating...")
    st_ = Lux.testmode(state.states)
    for (input, labels) in data_loader
        input, labels = preprocess_data(input, labels)

        # Pass the data through the network
        loss, st_, _ = computeloss(model, state.parameters, st_, (input, labels))

        # store the loss
        push!(losses, loss)
        push!(batch_sizes, num_samples(processed_data_layout, input))
        next!(prog)
    end
    avg_val_loss, has_inf_loss = losssummary(losses, batch_sizes)
    echo_summary(; epoch, avg_loss=avg_val_loss, elapsed=now() - t0_val,
                 title="Validation")
    return (; avg_loss=avg_val_loss, losses, has_inf_loss)
end