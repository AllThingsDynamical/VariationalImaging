using Plots

struct CrossImage
    image::Matrix{Float64}
    damaged::Matrix{Float64}
    mask::Matrix{Float64}
end

function generate_cross_image(N, M; cross_halfwidth=12)
    img = zeros(Float64, N, M)

    cy = div(N + 1, 2)
    cx = div(M + 1, 2)

    # horizontal band
    y1 = max(1, cy - cross_halfwidth)
    y2 = min(N, cy + cross_halfwidth)
    img[y1:y2, :] .= 1.0

    # vertical band
    x1 = max(1, cx - cross_halfwidth)
    x2 = min(M, cx + cross_halfwidth)
    img[:, x1:x2] .= 1.0

    return img
end

function damage_center_square(img; square_halfwidth=15, damaged_value=0.5)
    damaged = copy(img)
    N, M = size(img)

    cy = div(N + 1, 2)
    cx = div(M + 1, 2)

    y1 = max(1, cy - square_halfwidth)
    y2 = min(N, cy + square_halfwidth)
    x1 = max(1, cx - square_halfwidth)
    x2 = min(M, cx + square_halfwidth)

    damaged[y1:y2, x1:x2] .= damaged_value
    return damaged
end

function mask_center_square(N, M; square_halfwidth=25)
    mask = ones(Float64, N, M)

    cy = div(N + 1, 2)
    cx = div(M + 1, 2)

    y1 = max(1, cy - square_halfwidth)
    y2 = min(N, cy + square_halfwidth)
    x1 = max(1, cx - square_halfwidth)
    x2 = min(M, cx + square_halfwidth)

    mask[y1:y2, x1:x2] .= 0.0
    return mask
end

function make_cross_image(N=150, M=180; cross_halfwidth=12, square_halfwidth=15)
    img = generate_cross_image(N, M; cross_halfwidth=cross_halfwidth)
    damaged = damage_center_square(img; square_halfwidth=square_halfwidth)
    mask = mask_center_square(N, M; square_halfwidth=square_halfwidth)
    return CrossImage(img, damaged, mask)
end

function plot_and_save(fimg::CrossImage, path::String)
    img = fimg.image
    damaged = fimg.damaged
    mask = fimg.mask

    p1 = heatmap(img,
        color = :grays,
        axis = false,
        title = "Binary Cross Image")

    p2 = heatmap(damaged,
        color = :grays,
        axis = false,
        title = "Damaged Image")

    p3 = heatmap(mask,
        color = :grays,
        axis = false,
        title = "Mask")

    combined = plot(p1, p2, p3, layout=(1,3), size=(1200,400))

    display(combined)

    savefig(p1, joinpath(path, "cross_image.png"))
    savefig(p2, joinpath(path, "cross_damaged.png"))
    savefig(p3, joinpath(path, "cross_mask.png"))
end

include("../algorithms/utils.jl")

cimg = make_cross_image(200, 200; cross_halfwidth=12, square_halfwidth=30)
plot_and_save(cimg, "experiments/Figures/cross/")