using FileIO
using Images
using ColorTypes
using Plots

function read_grayscale_image(filename::String, k1::Int, k2::Int)
    file = load(filename)
    image = Gray.(file)
    Im = Float64.(channelview(image))
    img = Im[k1:end-k1, k2:end-k2]
    return img
end

begin
    filename = "data/inpainting/example-1.png"
    img = read_grayscale_image(filename, 10, 10)
    print(size(img))
    figure = heatmap(img)
end