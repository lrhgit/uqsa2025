def conditional_slices_interactive(zm, w, jpdf):

    Nrv = zm.shape[0]

    # --- UI widgets ---
    Ns_slider  = widgets.IntSlider(value=500, min=50, max=5000, step=50, description="Ns")
    Ndz_slider = widgets.IntSlider(value=6,   min=2,  max=20,   step=1,  description="Slices")

    out = widgets.Output()

    def update(*args):
        with out:
            out.clear_output(wait=True)
            fig = compute_slice_figure(zm, w, jpdf, Ns_slider.value, Ndz_slider.value)
            display(fig)

    Ns_slider.observe(update, 'value')
    Ndz_slider.observe(update, 'value')

    update()   # initial draw

    ui = widgets.VBox([
        widgets.HBox([Ns_slider, Ndz_slider]),
        out
    ])

    return ui
