using FFTW
using LinearAlgebra
using Statistics

function dct2(u)
    dct(dct(u,1),2)
end

function idct2(u)
    idct(idct(u,1),2)
end

function laplacian_eigs(N,M,L1=1.0,L2=1.0)
    μ = zeros(Float64,N,M)
    for k in 0:N-1
        for l in 0:M-1

            μ[k+1,l+1] =
                (4*N^2/L1^2)*sin(pi*k/(2N))^2 +
                (4*M^2/L2^2)*sin(pi*l/(2M))^2

        end
    end
    return μ
end

function laplacian(u, μ)
    A = dct2(u)
    return idct2(-μ .* A)
end

function biharmonic(u, μ)
    A = dct2(u)
    return idct2((μ.^2) .* A)
end

function Fprime(u)
    return 1 .- 6 .* u .+ 6 .* u.^2
end


function psnr(u, uhat; L=1.0)
    mse = mean((u .- uhat).^2)
    return 10 * log10(L^2 / mse)
end


TEST = false
if TEST
    bimg = make_boring_image(300, 600)
    N, M = size(bimg.damaged)
    μ = laplacian_eigs(N, M)
    lap = laplacian(bimg.damaged, μ)
    blap = biharmonic(bimg.damaged, μ)
    f_img = Fprime(bimg.damaged)

    figure1 = heatmap(bimg.damaged)
    figure2 = heatmap(lap)
    figure3 = heatmap(blap)
    figure4 = heatmap(f_img)

    plot(figure1, figure2, figure3, figure4)
end

using Plots
using LaTeXStrings
using Measures

# Global publication theme
default(
    fontfamily = "Computer Modern",
    linewidth = 1.2,
    markersize = 4,
    legendfontsize = 9,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 14,
    framestyle = :box,
    grid = false,
    minorgrid = true,
    tickdirection = :out,
    foreground_color_border = :black,
    foreground_color_axis = :black,
    foreground_color_text = :black,
    background_color = :white,
    dpi = 500,
    margin = 7.5mm)

# Consistent color cycle (colorblind-safe, print-friendly)
const PUB_COLORS = [
    RGB(0.0, 0.2, 0.6),   # deep blue
    RGB(0.8, 0.2, 0.2),   # red
    RGB(0.2, 0.6, 0.2),   # green
    RGB(0.6, 0.4, 0.0),   # ochre
    RGB(0.4, 0.2, 0.6)    # purple
]
palette(PUB_COLORS)

# Convenience wrapper for axis labels (LaTeX by default)
xlabel!(s) = xlabel!(L"$s$")
ylabel!(s) = ylabel!(L"$s$")