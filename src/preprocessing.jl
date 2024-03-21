"""
Select orientation from one of:
    - gyro integration
    - game rotation vector or
    - EKF orientation.

Arguments
---------
- path: path to the compiled data. It should contain "data.hdf5" and "info.json".
- max_ori_error: maximum allow alignment error.
- grv_only: When set to True, only game rotation vector will be used.
            When set to False:
                * If game rotation vector's alignment error is smaller than "max_ori_error", use it.
                * Otherwise, the orientation will be whichever gives lowest alignment error.
            To force using the best of all sources, set "grv_only" to False and "max_ori_error" to -1.
            To force using game rotation vector, set "max_ori_error" to any number greater than 360.

Returns
-------
- source_name: a string. One of 'gyro_integration', 'game_rv' and 'ekf'.
- ori: the selected orientation.
- ori_error: the end-alignment error of selected orientation.
"""
function select_orientation_source(path; max_ori_error=20.0, grv_only=true, use_ekf=true)
    info = read_info_from_file(path)
    ori_errors = (gyro=info["gyro_integration_error"],
                  grv=info["grv_ori_error"],
                  ekf=info["ekf_ori_error"])
    init_gyro_bias = info["imu_init_gyro_bias"]

    ori_choice = if grv_only || ori_errors.grv < max_ori_error
        :grv
    elseif use_ekf
        argmin(ori_errors)
    else
        argmin(ori_errors[(:gyro, :grv)])
    end

    data = read_data_from_file(path)

    if ori_choice == :gyro
        tt = data["synced"]["time"]
        gyro = data["synced"]["gyro_uncalib"] .- init_gyro_bias
        ori = gyro_integration(tt, gyro, data["synced"]["game_rv"][:, 1])
        return "gyro_integration", ori, ori_errors.gyro
    elseif ori_choice == :grv
        return "game_rv", data["synced"]["game_rv"], ori_errors.grv
    elseif ori_choice == :ekf
        return "ekf", data["pose"]["ekf_ori"], ori_errors.ekf
    end
    return error("Unknown orientation source: $ori_choice")
end

"""
Integrate gyroscope into orientation.
https://www.lucidar.me/en/quaternions/quaternion-and-gyroscope/
"""
function gyro_integration(tt, gyro, init_q)
    n = size(gyro, 2)
    out = Vector{Float64}(undef, 4, n)
    out[:, 1] .= init_q
    δt = diff(tt)
    for i in range(2, n)
        out[:, i] .= out[:, i - 1] +
                     angular_velocity_to_quaternion_derivative(out[:, i - 1],
                                                               gyro[:, i - 1]) .* δt[i - 1]
        out[:, i] ./= norm(out[:, i])
    end
    return out
end

"""
Time derivative of the angular velocity expressed in the quaternion form:
q̇ = 0.5⋅Ω(ω)q

https://ahrs.readthedocs.io/en/latest/filters/angular.html#id11
"""
function angular_velocity_to_quaternion_derivative(q, ω)
    Ω = [0 -ω[1] -ω[2] -ω[3];
         ω[1] 0 ω[3] -ω[2];
         ω[2] -ω[3] 0 ω[1];
         ω[3] ω[2] -ω[1] 0]
    return 0.5 * Ω * q
end
