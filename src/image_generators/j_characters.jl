using FileIO
using Images
using Plots
using Random

struct CheckerImage
    image
    damaged
    mask
end

function generate_image(path; threshold=0.65, stride=20)

    img = load(path)
    g = Float64.(Gray.(img))
    B = Float64.(g .< threshold)

    B = reverse(B, dims=(1,2))        # 180° rotation
    B = B[1:stride:end, 1:stride:end] # downsample

    return B
end

function damage_image(img)

    B = copy(img)
    N, M = size(B)

    K = Int(N/1)

    idx = rand(1:N, K)
    idy = rand(1:M, K)

    B[idx, idy] .= 0.5

    return B
end

function make_mask(img)

    mask = ones(Float64, size(img))

    damaged_locations = img .== 0.5
    mask[damaged_locations] .= 0.0

    return mask
end

function make_checker_image(path="src/image_generators/checker_board.png")

    img = generate_image(path)

    damaged = damage_image(img)

    mask = make_mask(damaged)

    return CheckerImage(img, damaged, mask)
end


function plot_and_save(cimg::CheckerImage, path::String)

    img = cimg.image
    damaged = cimg.damaged
    mask = cimg.mask

    p1 = heatmap(img,
        color=:grays,
        axis=false,
        title="Binary Image")

    p2 = heatmap(damaged,
        color=:grays,
        axis=false,
        title="Damaged Image")

    p3 = heatmap(mask,
        color=:grays,
        axis=false,
        title="Mask")

    combined = plot(p1,p2,p3, layout=(1,3), size=(1200,400))

    display(combined)

    savefig(p1, joinpath(path,"checker_image.png"))
    savefig(p2, joinpath(path,"checker_damaged.png"))
    savefig(p3, joinpath(path,"checker_mask.png"))
end


img = make_checker_image()
plot_and_save(img, "experiments/Figures/checker/")