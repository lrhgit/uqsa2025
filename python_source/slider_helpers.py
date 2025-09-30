
from ipywidgets import VBox, HBox, interactive_output
from IPython.display import display

def build_slider_interface(slider_dict, model_function, sliders_per_row=4):
    '''
    Display sliders and link them to the model function.

    Parameters:
    - slider_dict: dict of {name: FloatSlider}
    - model_function: function to call with slider values
    - sliders_per_row: number of sliders to display per row (default 4)
    '''
    slider_list = list(slider_dict.values())
    rows = [HBox(slider_list[i:i+sliders_per_row]) for i in range(0, len(slider_list), sliders_per_row)]
    ui = VBox(rows)
    output = interactive_output(model_function, slider_dict)
    display(ui, output)

from ipywidgets import FloatSlider

def make_slider_dict(prefixes, count, default_values=None):
    slider_dict = {}
    for prefix in prefixes:
        for i in range(1, count + 1):
            name = f"{prefix}{i}"
            value = default_values.get(name, 0.2) if default_values else 0.2
            if prefix == "delta" and i <= 2:
                value = 0.5  # spesialverdi for de første deltaverdiene
            elif prefix == "a" and i == 1:
                value = 0.4
            elif prefix == "alpha" and i == 1:
                value = 0.4
            slider_dict[name] = FloatSlider(
                value=value, min=0.0, max=1.0, step=0.05, description=name
            )
    return slider_dict
