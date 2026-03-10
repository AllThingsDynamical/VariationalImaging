using Plots

function generate_image(N,M)
    x = range(0,1,length=N)
    y = range(0,1,length=M)

    img = zeros(N,M)

    for i in 1:N
        for j in 1:M

            img[i,j] =
                0.5 +
                0.25*sin(4π*x[i])*cos(4π*y[j]) +
                0.25*sin(8π*x[i])*cos(8π*y[j])

        end
    end
    return img
end

function damage_image(img)
    damaged = copy(img)
    damaged[50:200,40:90] .= 0.5
    return damaged
end

function threshold_image(img, τ=0.5)
    binary = img .> τ
    return Float64.(binary)
end

function make_mask(N,M)
    mask = ones(N,M)
    mask[50:200,40:90] .= 0
    return mask
end

struct FancyImage1
    image
    damaged
    mask
end

function make_fancy_image1(N=256,M=128)
    img = generate_image(N,M)
    binary_img = threshold_image(img,0.5)
    damaged = damage_image(binary_img)
    mask = make_mask(N,M)
    return FancyImage1(binary_img, damaged, mask)
end


function plot_and_save(fimg::FancyImage1, path::String)

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

    savefig(p1, joinpath(path,"f1_image.png"))
    savefig(p2, joinpath(path,"f1_damaged.png"))
    savefig(p3, joinpath(path,"f1_mask.png"))
end

img = make_fancy_image1()
plot_and_save(img, "experiments/Figures/F1/")