
from ipywidgets import VBox, HBox, interactive_output
from IPython.display import display
import ipywidgets as widgets

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

def make_slider_dict(prefixes, count, default_values=None, slider_ranges=None):
    """
    Create a dictionary of FloatSliders.

    Parameters
    ----------
    prefixes : list[str]
        Parameter name prefixes, e.g. ["a", "alpha"].
    count : int
        Number of sliders per prefix.
    default_values : dict, optional
        Initial values keyed by slider name.
    slider_ranges : dict, optional
        Mapping prefix -> (min, max, step).
    """
    if default_values is None:
        default_values = {}

    if slider_ranges is None:
        slider_ranges = {}

    slider_dict = {}

    for prefix in prefixes:
        vmin, vmax, step = slider_ranges.get(prefix, (0.0, 1.0, 0.05))

        for i in range(1, count + 1):
            name = f"{prefix}{i}"


            slider_dict[name] = widgets.FloatSlider(
                value=default_values.get(name, 0.2),
                min=vmin,
                max=vmax,
                step=step,
                description=name,
                continuous_update=False,
                style={"description_width": "70px"},
                layout=widgets.Layout(width="260px"),
            )

            

    return slider_dict

def make_int_slider(name, value, vmin, vmax, step=1, width="220px"):
    return widgets.IntSlider(
        value=value,
        min=vmin,
        max=vmax,
        step=step,
        description=name,
        continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width=width),
    )

