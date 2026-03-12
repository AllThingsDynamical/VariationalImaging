include("utils.jl")
using ProgressMeter


struct Momentum_BGCHParams
    dt::Float64
    ϵ::Float64
    λ0::Float64
    γ::Float64
    C1::Float64
    C2::Float64
    iters::Int
    L1::Float64
    L2::Float64

    function Momentum_BGCHParams(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
        dt > 0 || error("dt must be positive")
        ϵ > 0 || error("ϵ must be positive")
        λ0 >= 0 || error("λ0 must be nonnegative")
        γ >= 0 || error("γ must be nonnegative")
        C1 > 1 / ϵ || error("C1 must be > 1/ϵ")
        C2 > λ0 || error("C2 must be > λ0")
        iters > 0 || error("iters must be positive")
        L1 > 0 || error("L1 must be positive")
        L2 > 0 || error("L2 must be positive")

        new(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
    end
end


struct BGCHProblem
    experiment
    params::Momentum_BGCHParams
end

struct BGCHCache
    μ::Matrix{Float64}
    lhs::Matrix{Float64}
    λmask::Matrix{Float64}
end

function BGCHCache(problem::BGCHProblem)
    U = problem.experiment.damaged
    N, M = size(U)
    p = problem.params

    μ = laplacian_eigs(N, M, p.L1, p.L2)

    lhs =
        (1 / p.dt^2 + p.γ / p.dt - p.C2) .+
        p.ϵ .* μ.^2 .+
        p.C1 .* μ

    λmask = p.λ0 .* problem.experiment.mask

    return BGCHCache(μ, lhs, λmask)
end

function bgch_momentum_step(
    Ukm1::Matrix{Float64},
    Uk::Matrix{Float64},
    problem::BGCHProblem,
    cache::BGCHCache
)
    p = problem.params
    f = problem.experiment.image

    rhs =
        (2 / p.dt^2 + p.γ / p.dt) .* Uk .-
        (1 / p.dt^2) .* Ukm1 .+
        (1 / (2 * p.ϵ)) .* laplacian(Fprime(Uk), cache.μ) .-
        p.C1 .* laplacian(Uk, cache.μ) .+
        cache.λmask .* (f .- Uk) .-
        p.C2 .* Uk

    rhs_hat = dct2(rhs)
    Ukp1_hat = rhs_hat ./ cache.lhs

    return idct2(Ukp1_hat)
end

function solve_momentum(problem::BGCHProblem)
    U0 = copy(problem.experiment.damaged)
    Um1 = copy(U0)   # zero initial velocity
    Uk = copy(U0)

    cache = BGCHCache(problem)

    Us = Matrix{Float64}[copy(U0)]
    Qc = [psnr(problem.experiment.image, Uk)]

    @showprogress for k in 1:problem.params.iters
        Ukp1 = bgch_momentum_step(Um1, Uk, problem, cache)
        ps = psnr(problem.experiment.image, Ukp1)
        push!(Us, Ukp1)
        push!(Qc, ps)

        Um1 = Uk
        Uk = Ukp1
    end

    return Us, Qc
end


BORING = false
if BORING
    function Momentum_BGCHParams(;
        dt::Float64 = 1e-2,
        ϵ::Float64 = 0.1,
        λ0::Float64 = 1e8,
        γ::Float64 = 1.5e6,
        C1::Float64 = 4e3,
        C2::Float64 = 1e8 + 1.0,
        iters::Int = 1200,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        return Momentum_BGCHParams(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
    end

    begin
        img = make_boring_image(200, 200)
        params = Momentum_BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve_momentum(problem)
    end

    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/boring/boring_momentum_flow.gif", fps=5)
    end


    figure = heatmap(sol[end], axis=false, color=:grays, title="Momentum flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/boring/boring_momentum_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Momentum flow")
    savefig("experiments/Figures/boring/boring_error_momentum_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/boring/boring_psnr_momentum_flow.png")
end

CIRCLE = false
if CIRCLE
    function Momentum_BGCHParams(;
        dt::Float64 = 1e-2,
        ϵ::Float64 = 0.1,
        λ0::Float64 = 1e8,
        γ::Float64 = 1.5e6,
        C1::Float64 = 4e3,
        C2::Float64 = 1e8 + 1.0,
        iters::Int = 1600,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        return Momentum_BGCHParams(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
    end


    begin
        img = make_circle_experiment(256, 256)
        params = Momentum_BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve_momentum(problem)
    end


    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/circle/circle_momentum_flow.gif", fps=5)
    end

    figure = heatmap(sol[end], axis=false, color=:grays, title="Momentum flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/circle/circle_momentum_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Momentum flow")
    savefig("experiments/Figures/circle/circle_error_momentum_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/circle/circle_psnr_momentum_flow.png")
end

FANCY = false
if FANCY
 function Momentum_BGCHParams(;
        dt::Float64 = 1e-2,
        ϵ::Float64 = 0.1,
        λ0::Float64 = 1e12,
        γ::Float64 = 2e10,
        C1::Float64 = 4e3,
        C2::Float64 = 1e12 + 1.0,
        iters::Int = 20_000,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        return Momentum_BGCHParams(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
    end


    begin
        img = make_fancy_image1()
        params = Momentum_BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve_momentum(problem)
    end


    begin
        anim = @animate for i=1:500:length(sol)
            heatmap(sol[i], title="$i", color=:grays)
        end
        gif(anim, "experiments/Figures/F1/F1_momentum_flow.gif", fps=5)
    end

    figure = heatmap(sol[end], axis=false, color=:grays, title="Momentum flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/F1/F1_momentum_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Momentum flow")
    savefig("experiments/Figures/F1/F1_error_momentum_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/F1/F1_psnr_momentum_flow.png")
end 

CROSS = false
if CROSS
 function Momentum_BGCHParams(;
        dt::Float64 = 1e-2,
        ϵ::Float64 = 0.5,
        λ0::Float64 = 1e8,
        γ::Float64 = 1.5e6,
        C1::Float64 = 4e3,
        C2::Float64 = 1e8 + 1.0,
        iters::Int = 500,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        return Momentum_BGCHParams(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
    end


    begin
        img = img = make_cross_image(200, 200; cross_halfwidth=12, square_halfwidth=30)
        params = Momentum_BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve_momentum(problem)
    end


    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/cross/cross_momentum_flow.gif", fps=5)
    end


    figure = heatmap(sol[end], axis=false, color=:grays, title="Momentum flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/cross/cross_momentum_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Momentum flow")
    savefig("experiments/Figures/cross/cross_error_momentum_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/cross/cross_psnr_momentum_flow.png")
end

function Momentum_BGCHParams(;
        dt::Float64 = 1e-2,
        ϵ::Float64 = 0.02,
        λ0::Float64 = 1e10,
        γ::Float64 = 1.5e8,
        C1::Float64 = 4e3,
        C2::Float64 = 1e10 + 1.0,
        iters::Int = 2000,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        return Momentum_BGCHParams(dt, ϵ, λ0, γ, C1, C2, iters, L1, L2)
end


    begin
        img = img = make_checker_image()
        params = Momentum_BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve_momentum(problem)
    end

    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/checker/checker_momentum_flow.gif", fps=5)
    end

    figure = heatmap(sol[end], axis=false, color=:grays, title="Momentum flow - ϵ = $(problem.params.ϵ)", clim=(0.0,1.0))
    savefig("experiments/Figures/checker/checker_momentum_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Momentum flow")
    savefig("experiments/Figures/checker/checker_error_momentum_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/checker/checker_psnr_momentum_flow.png")