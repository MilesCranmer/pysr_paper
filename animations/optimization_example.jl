using Optim
using Random

function get_optimization_history(;
    true_f, test_f, c0, c_bounds, alg=NelderMead(), iterations=10
)
    n = 5_000
    X = rand(MersenneTwister(0), n) .* 20 .- 10
    Y = rand(MersenneTwister(1), n) .* 20 .- 10
    Z = rand(MersenneTwister(2), n) .* 20 .- 10
    nconst = length(c0)

    function f(c; X, Y, Z, c_hist)
        push!(c_hist, copy(c))
        loss = sum([
            abs2(true_f(x, y, z) - test_f(x, y, z; c=c)) for (x, y, z) in zip(X, Y, Z)
        ])
        for i in 1:nconst
            low = c_bounds[i][1]
            high = c_bounds[i][2]
            if c[i] < low
                loss += 5 * (c[i] - low)^2
            elseif c[i] > high
                loss += 5 * (c[i] - high)^2
            end
        end
        return loss
    end

    c_hist = [copy(c0)]
    Random.seed!(3)
    res = optimize(
        c -> f(c; X, Y, Z, c_hist), c0, alg, Optim.Options(; iterations=iterations)
    )
    for _ in 1:ceil(Int, 0.1 * length(c_hist))
        push!(c_hist, copy(res.minimizer))
    end
    c_hist = c_hist[findall(
        c -> all([ci >= low && ci <= high for (ci, (low, high)) in zip(c, c_bounds)]),
        c_hist,
    )]

    return c_hist
end
