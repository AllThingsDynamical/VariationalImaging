struct CircleImage
    image::Matrix{Float64}
    damaged::Matrix{Float64}
    mask::Matrix{Float64}
end

function generate_circle(N,M; radius=0.35)
    img = zeros(Float64,N,M)
    x = range(-1,1,length=N)
    y = range(-1,1,length=M)
    for i in 1:N
        for j in 1:M

            if x[i]^2 + y[j]^2 < radius^2
                img[i,j] = 1.0
            end

        end
    end
    return img
end

function damage_circle(img; y1=110, y2=140)
    damaged = copy(img)
    damaged[y1:y2,:] .= 0.5
    return damaged
end

function mask_circle(N,M; y1=110, y2=140)
    mask = ones(Float64,N,M)
    mask[y1:y2,:] .= 0.0
    return mask
end

function make_circle_image(N,M)
    img = generate_circle(N,M)
    damaged = damage_circle(img)
    mask = mask_circle(N,M)
    return CircleImage(img,damaged,mask)
end

function plot_and_save(fimg::CircleImage, path::String)

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

    savefig(p1, joinpath(path,"circle_image.png"))
    savefig(p2, joinpath(path,"circle_damaged.png"))
    savefig(p3, joinpath(path,"circle_mask.png"))
end

function make_circle_experiment(N,M; radius=0.35, y1=110, y2=140)

    x = range(-1,1,length=N)
    y = range(-1,1,length=M)

    img = zeros(Float64,N,M)

    for i in 1:N
        for j in 1:M
            if x[i]^2 + y[j]^2 < radius^2
                img[i,j] = 1.0
            end
        end
    end

    damaged = copy(img)
    damaged[y1:y2,:] .= 0.5

    mask = ones(Float64,N,M)
    mask[y1:y2,:] .= 0.0

    return CircleImage(img, damaged, mask)

end

circle = make_circle_experiment(256,256)
plot_and_save(circle,"experiments/Figures/circle/")