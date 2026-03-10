using LinearAlgebra

function lap1d(n, h)
    e = ones(n)
    L = diagm(-1 => e[2:end], 0 => -2e, 1 => e[1:end-1])
    return L / h^2
end

function Laplacian(img::Matrix)
    nx, ny = size(img)
    hx, hy = 1, 1
    Lx = lap1d(nx, hx)
    Ly = lap1d(ny, hy)
    return Lx*img + img*Ly'
end

begin
    filename = "data/inpainting/example-1.png"
    img = read_grayscale_image(filename, 10, 10)
    epsilon = 5e-1
    Delta_img = Laplacian(img)
    figure = heatmap(Delta_img)
end