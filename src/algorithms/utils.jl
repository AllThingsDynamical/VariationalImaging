using FFTW
using LinearAlgebra

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