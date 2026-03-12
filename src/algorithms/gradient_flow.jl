include("utils.jl")
using ProgressMeter

struct BGCHParams
    dt::Float64
    ϵ::Float64
    λ0::Float64
    C1::Float64
    C2::Float64
    iters::Int
    L1::Float64
    L2::Float64

    function BGCHParams(dt,ϵ,λ0,C1,C2,iters,L1,L2)

        C1 > 1/ϵ || error("C1 must be > 1/ϵ")
        C2 > λ0 || error("C2 must be > λ0")

        new(dt,ϵ,λ0,C1,C2,iters,L1,L2)
    end
end


struct BGCHProblem
    experiment
    params::BGCHParams
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
    lhs = 1 .+ p.dt .* (p.ϵ .* μ.^2 .+ p.C1 .* μ .+ p.C2)
    λmask = p.λ0 .* problem.experiment.mask
    return BGCHCache(μ, lhs, λmask)
end

function bgch_step(U::Matrix, problem::BGCHProblem, cache::BGCHCache)
    p = problem.params
    f = problem.experiment.image
    rhs =
    U .+
    p.dt .* (
        (1/p.ϵ) .* laplacian(Fprime(U), cache.μ) .-
        p.C1 .* laplacian(U, cache.μ) .+
        cache.λmask .* (f .- U) .+
        p.C2 .* U
    )
    rhs_hat = dct2(rhs)
    U_next_hat = rhs_hat ./ cache.lhs
    return idct2(U_next_hat)
end

function solve(problem::BGCHProblem)
    U = copy(problem.experiment.damaged)
    cache = BGCHCache(problem)
    Us = [U]
    Qc = [psnr(problem.experiment.image, U)]
    @showprogress for k in 1:problem.params.iters
        U = bgch_step(U, problem, cache)
        ps = psnr(problem.experiment.image, U)
        push!(Us, U)
        push!(Qc, ps)
    end
    return Us, Qc
end

BORING = false
if BORING
    function BGCHParams(;
        dt::Float64 = 1.0,
        ϵ::Float64 = 0.1,
        λ0::Float64 = 1e8,
        C1::Float64 = 4e3,
        C2::Float64 = 1e8 + 1.0,
        iters::Int = 1200,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        dt > 0 || error("dt must be positive")
        ϵ > 0 || error("ϵ must be positive")
        λ0 >= 0 || error("λ0 must be nonnegative")
        iters > 0 || error("iters must be positive")
        L1 > 0 || error("L1 must be positive")
        L2 > 0 || error("L2 must be positive")

        C1 > 1 / ϵ || error("C1 must satisfy C1 > 1/ϵ")
        C2 >= λ0 || error("C2 must satisfy C2 >= λ0")

        return BGCHParams(dt, ϵ, λ0, C1, C2, iters, L1, L2)
    end


    begin
        img = make_boring_image(200, 200)
        params = BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve(problem)
    end

    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/boring/boring_gradient_flow.gif", fps=5)
    end


    figure = heatmap(sol[end], axis=false, color=:grays, title="Gradient flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/boring/boring_gradient_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Gradient flow")
    savefig("experiments/Figures/boring/boring_error_gradient_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/boring/boring_psnr_gradient_flow.png")
end

CIRCLE = false
if CIRCLE
 function BGCHParams(;
        dt::Float64 = 1.0,
        ϵ::Float64 = 0.1,
        λ0::Float64 = 1e8,
        C1::Float64 = 4e3,
        C2::Float64 = 1e8 + 1.0,
        iters::Int = 1600,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        dt > 0 || error("dt must be positive")
        ϵ > 0 || error("ϵ must be positive")
        λ0 >= 0 || error("λ0 must be nonnegative")
        iters > 0 || error("iters must be positive")
        L1 > 0 || error("L1 must be positive")
        L2 > 0 || error("L2 must be positive")

        C1 > 1 / ϵ || error("C1 must satisfy C1 > 1/ϵ")
        C2 >= λ0 || error("C2 must satisfy C2 >= λ0")

        return BGCHParams(dt, ϵ, λ0, C1, C2, iters, L1, L2)
    end


    begin
        img = make_circle_experiment(256, 256)
        params = BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve(problem)
    end


    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/circle/circle_gradient_flow.gif", fps=5)
    end


    figure = heatmap(sol[end], axis=false, color=:grays, title="Gradient flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/circle/circle_gradient_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Gradient flow")
    savefig("experiments/Figures/circle/circle_error_gradient_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/circle/circle_psnr_gradient_flow.png")
end

FANCY = false
if FANCY
 function BGCHParams(;
        dt::Float64 = 1.0,
        ϵ::Float64 = 0.1,
        λ0::Float64 = 1e11,
        C1::Float64 = 4e3,
        C2::Float64 = 1e11 + 1.0,
        iters::Int = 20_000,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        dt > 0 || error("dt must be positive")
        ϵ > 0 || error("ϵ must be positive")
        λ0 >= 0 || error("λ0 must be nonnegative")
        iters > 0 || error("iters must be positive")
        L1 > 0 || error("L1 must be positive")
        L2 > 0 || error("L2 must be positive")

        C1 > 1 / ϵ || error("C1 must satisfy C1 > 1/ϵ")
        C2 >= λ0 || error("C2 must satisfy C2 >= λ0")

        return BGCHParams(dt, ϵ, λ0, C1, C2, iters, L1, L2)
    end


    begin
        img = make_fancy_image1()
        params = BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve(problem)
    end

     begin
        anim = @animate for i=1:500:length(sol)
            heatmap(sol[i], title="$i", color=:grays)
        end
        gif(anim, "experiments/Figures/F1/F1_gradient_flow.gif", fps=5)
    end

    figure = heatmap(sol[end], axis=false, color=:grays, title="Gradient flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/F1/F1_gradient_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Gradient flow")
    savefig("experiments/Figures/F1/F1_error_gradient_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/F1/F1_psnr_gradient_flow.png")
end

CROSS=false
if CROSS
function BGCHParams(;
        dt::Float64 = 1.0,
        ϵ::Float64 = 0.5,
        λ0::Float64 = 1e8,
        C1::Float64 = 4e3,
        C2::Float64 = 1e8 + 1.0,
        iters::Int = 500,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        dt > 0 || error("dt must be positive")
        ϵ > 0 || error("ϵ must be positive")
        λ0 >= 0 || error("λ0 must be nonnegative")
        iters > 0 || error("iters must be positive")
        L1 > 0 || error("L1 must be positive")
        L2 > 0 || error("L2 must be positive")

        C1 > 1 / ϵ || error("C1 must satisfy C1 > 1/ϵ")
        C2 >= λ0 || error("C2 must satisfy C2 >= λ0")

        return BGCHParams(dt, ϵ, λ0, C1, C2, iters, L1, L2)
    end


    begin
        img = make_cross_image(200, 200; cross_halfwidth=12, square_halfwidth=30)
        params = BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve(problem)
    end


    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/cross/cross_gradient_flow.gif", fps=5)
    end

    figure = heatmap(sol[end], axis=false, color=:grays, title="Gradient flow - ϵ = $(problem.params.ϵ)")
    savefig("experiments/Figures/cross/cross_gradient_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Gradient flow")
    savefig("experiments/Figures/cross/cross_error_gradient_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/cross/cross_psnr_gradient_flow.png")
end

function BGCHParams(;
        dt::Float64 = 0.1,
        ϵ::Float64 = 0.02,
        λ0::Float64 = 1e10,
        C1::Float64 = 4e3,
        C2::Float64 = 1e10 + 1.0,
        iters::Int = 2000,
        L1::Float64 = 1.0,
        L2::Float64 = 1.0
    )
        dt > 0 || error("dt must be positive")
        ϵ > 0 || error("ϵ must be positive")
        λ0 >= 0 || error("λ0 must be nonnegative")
        iters > 0 || error("iters must be positive")
        L1 > 0 || error("L1 must be positive")
        L2 > 0 || error("L2 must be positive")

        C1 > 1 / ϵ || error("C1 must satisfy C1 > 1/ϵ")
        C2 >= λ0 || error("C2 must satisfy C2 >= λ0")

        return BGCHParams(dt, ϵ, λ0, C1, C2, iters, L1, L2)
    end


    begin
        img = make_checker_image()
        params = BGCHParams()
        problem = BGCHProblem(img, params)
        sol, errs = solve(problem)
    end


    begin
        anim = @animate for i=1:50:length(sol)
            heatmap(sol[i], title="$i", clim=(0.0, 1.0), color=:grays)
        end
        gif(anim, "experiments/Figures/checker/checker_gradient_flow.gif", fps=5)
    end

        figure = heatmap(sol[end], axis=false, color=:grays, title="Gradient flow - ϵ = $(problem.params.ϵ)", clim=(0.0,1.0))
    savefig("experiments/Figures/checker/checker_gradient_flow_ϵ = $(problem.params.ϵ).png")


    figure = heatmap(abs.(sol[end] .- problem.experiment.image), axis=false, color=:grays, title="Reconstruction difference - Gradient flow")
    savefig("experiments/Figures/checker/checker_error_gradient_flow_ϵ = $(problem.params.ϵ).png")

    figure = plot(1:length(errs), errs, xlabel="Iterations", ylabel="PSNR", label="PSNR")
    savefig("experiments/Figures/checker/checker_psnr_gradient_flow.png")