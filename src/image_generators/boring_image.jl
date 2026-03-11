using Plots
struct BoringImage
    image::Matrix{Float64}
    damaged::Matrix{Float64}
    mask::Matrix{Float64}
end

function generate_boring_image(N,M; stripe_width=30)
    img = zeros(Float64,N,M)
    for j in 1:M
        if mod(j,2*stripe_width) < stripe_width
            img[:,j] .= 1.0
        end
    end
    return img
end

function damage_horizontal(img; y1=110, y2=140)
    damaged = copy(img)
    damaged[y1:y2,:] .= 0.5
    return damaged
end

function mask_horizontal(N,M; y1=110, y2=140)
    mask = ones(Float64,N,M)
    mask[y1:y2,:] .= 0.0
    return mask
end

function make_boring_image(N=150,M=180)
    img = generate_boring_image(N, M)
    damaged = damage_horizontal(img)
    mask = mask_horizontal(N,M)
    return BoringImage(img,damaged,mask)
end

function plot_and_save(fimg::BoringImage, path::String)
    img = fimg.image
    damaged = fimg.damaged
    mask = fimg.mask

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

    savefig(p1, joinpath(path,"boring_image.png"))
    savefig(p2, joinpath(path,"boring_damaged.png"))
    savefig(p3, joinpath(path,"boring_mask.png"))
end

include("../algorithms/utils.jl")
bimg = make_boring_image(200,200)
plot_and_save(bimg, "experiments/Figures/boring/")