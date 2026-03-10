function offset_potential(img::Matrix)
    mf1 = img
    mf2 = 1 .- img
    mf3 = 1 .- 2*img
    return 2 .* mf1 .* mf2 .* mf3
end

begin
    filename = "data/inpainting/example-1.png"
    img = read_grayscale_image(filename, 10, 10)
    epsilon = 9e-1
    Delta_img = Laplacian(img)
    p_img = offset_potential(img)
    t1 = -epsilon^2 * Delta_img + p_img
    t2 = Laplacian(t1)

    figure1 = heatmap(img)
    figure2 = heatmap(Delta_img)
    figure3 = heatmap(p_img)
    figure4 = heatmap(t1)
    figure5 = heatmap(t2)

    plot(figure1, figure2, figure3, figure4, figure5, size=(1000,400))
end