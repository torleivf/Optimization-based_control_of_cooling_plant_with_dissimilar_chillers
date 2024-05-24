import pandas as pd
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from src.oks.MPC.actions import get_active_chiller_numbers
from src.oks.MPC.model_parameters import get_default_chillers

supply_colour = "blue"
supply_dash = (0, (5, 5))
supply_short_label = "Supply"
return_colour = "orange"
return_dash = (0, (5, 6))
return_short_label = "Return"
river_colour = "xkcd:pale blue"
river_dash = (0, (5, 4))
river_short_label = "River"
plant_colour = "xkcd:greenish yellow"
plant_dash = (0, (5, 4))
plant_short_label = "Plant"
river_chilled_water_colour = "yellow"
river_chilled_water_dash = (0, (5, 4))
river_chilled_water_short_label = "In"
active_flow_colour = "cyan"
active_flow_dash = (0, (5, 6))
active_flow_short_label = "Active"
active_approximation_dot = (0, (1, 3))


def get_chiller_short_label(chiller):
    return f"Chiller$_{chiller.number}$"


def get_standard_plot(
    plot_data,
    model,
    chillers=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    active_chillers = get_active_chiller_numbers(
        model=model,
        chillers=chillers,
    )
    fig, axs = plt.subplots(
        5,
        1,
        figsize=(11, 8),
        sharex=True,
    )
    i = 0
    ax = axs[i]
    plot_historical_water_temperatures(
        ax,
        plot_data,
        outputs_only=True,
    )
    plot_model_water_temperatures(
        ax,
        model,
        outputs_only=True,
    )
    for c in active_chillers:
        plot_historical_chiller_output_temperatures(
            ax,
            plot_data,
            only_chiller_number=c,
        )
        plot_model_chiller_output_temperatures(
            ax,
            model,
            only_chiller_number=c,
        )
    ax.set_title("Output water temperature")
    i += 1
    ax = axs[i]
    plot_historical_water_temperatures(
        ax,
        plot_data,
        thick_supply_water_line=True,
        inputs_only=True,
    )
    plot_model_water_temperatures(
        ax,
        model,
        thick_supply_water_line=True,
        inputs_only=True,
    )
    for c in active_chillers:
        plot_historical_chiller_input_temperatures(
            ax,
            plot_data,
            only_chiller_number=c,
            chillers=chillers,
        )
        plot_model_chiller_input_temperatures(
            ax,
            model,
            only_chiller_number=c,
            chillers=chillers,
        )
    ax.set_title("Input water temperature")
    i += 1
    ax = axs[i]
    plot_historical_cooling_load(ax, plot_data)
    plot_model_cooling_load(ax, model)
    plot_historical_river_chilling_power(ax, plot_data)
    plot_model_river_chilling_power(ax, model)
    plot_historical_chiller_output_power(
        ax,
        plot_data,
        chillers=chillers,
    )
    plot_model_chiller_output_power(
        ax,
        model,
        chillers=chillers,
    )
    ax.set_title("Output power")
    i += 1
    ax = axs[i]
    plot_historical_chiller_input(
        ax,
        plot_data,
        chillers=chillers,
    )
    plot_model_chiller_input(
        ax,
        model,
        chillers=chillers,
    )
    ax.set_title("Power consumption")
    i += 1
    ax = axs[i]
    plot_historical_volumetric_flow(ax, plot_data)
    plot_model_volumetric_flows(ax, model, plot_chillers=True, chillers=chillers)
    ax.set_title("Volumetric flow")

    reformat_time_axis(ax)
    add_figure_legends(fig, axs, chillers)

    # close the plot to avoid memory leak
    plt.close()

    return fig


def get_standard_plot_legends_figure(chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    fig, axs = plt.subplots(5, 1, figsize=(11, 8), sharex=True, layout="tight")
    i = 0
    ax = axs[i]
    plot_output_water_temperature_legends(ax, chillers=chillers)
    ax.set_title("Output water temperature")
    i += 1
    ax = axs[i]
    plot_input_water_temperature_legends(ax, chillers=chillers)
    ax.set_title("Input water temperature")
    i += 1
    ax = axs[i]
    plot_output_power_legends(ax, chillers=chillers)
    ax.set_title("Output power")
    i += 1
    ax = axs[i]
    plot_input_power_legends(ax, chillers=chillers)
    ax.set_title("Power consumption")
    i += 1
    ax = axs[i]
    plot_volumetric_flow_legends(ax, chillers=chillers)
    ax.set_title("Volumetric flow")

    reformat_time_axis(ax)

    return fig


def plot_output_water_temperature_legends(ax, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    model_supply_line = mlines.Line2D(
        [],
        [],
        color=supply_colour,
        linestyle=supply_dash,
        label=f"Supply water temperature - {supply_short_label}",
    )
    model_legend = ax.legend(
        handles=[
            model_supply_line,
        ],
        loc="upper left",
    )
    ax.add_artist(model_legend)
    chiller_output_water_legend_handles = []
    for chiller in chillers:
        chiller_output_water_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=chiller.colour,
                linestyle=chiller.model_line_style,
                label=f"Output water "
                f"temperature of chiller {chiller.number}"
                f" - {get_chiller_short_label(chiller)}",
            )
        )
    chiller_output_water_legend = ax.legend(
        handles=chiller_output_water_legend_handles, loc="upper right"
    )
    ax.add_artist(chiller_output_water_legend)

    ax.set_ylabel("$\degree$C")
    ax.set_yticks([0, 5, 7, 10, 12, 15, 20])


def plot_input_water_temperature_legends(ax, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    model_return_line = mlines.Line2D(
        [],
        [],
        color=return_colour,
        linestyle=return_dash,
        label=f"Return water temperature - {return_short_label}",
    )
    model_river_chilled_line = mlines.Line2D(
        [],
        [],
        color=river_chilled_water_colour,
        linestyle=river_chilled_water_dash,
        label=f"River chilled water temperature"
        f" - {river_chilled_water_short_label}",
    )
    model_legend = ax.legend(
        handles=[
            model_return_line,
            model_river_chilled_line,
        ],
        loc="upper left",
    )
    ax.add_artist(model_legend)
    chiller_input_water_legend_handles = []
    for chiller in chillers:
        chiller_input_water_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=chiller.colour,
                linestyle=chiller.model_line_style,
                label=f"Input water "
                f"temperature of chiller {chiller.number}"
                f" - {get_chiller_short_label(chiller)}",
            )
        )
    chiller_input_water_legend = ax.legend(
        handles=chiller_input_water_legend_handles, loc="upper right"
    )
    ax.add_artist(chiller_input_water_legend)

    ax.set_ylabel("$\degree$C")
    ax.set_yticks([7, 10, 13, 16, 20, 25, 30])


def plot_output_power_legends(ax, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    model_supply_line = mlines.Line2D(
        [],
        [],
        color=supply_colour,
        linestyle=supply_dash,
        label=f"Cooling load at the hospital - {supply_short_label}",
    )
    model_river_chilling_line = mlines.Line2D(
        [],
        [],
        color=river_colour,
        linestyle=river_dash,
        label=f"Chilling from the river water heat exchanger" f" - {river_short_label}",
    )
    model_legend = ax.legend(
        handles=[
            model_supply_line,
            model_river_chilling_line,
        ],
        loc="upper left",
    )
    ax.add_artist(model_legend)
    chiller_output_power_legend_handles = []
    for chiller in chillers:
        chiller_output_power_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=chiller.colour,
                linestyle=chiller.model_line_style,
                label=f"Chilling from chiller {chiller.number}"
                f" - "
                f"{get_chiller_short_label(chiller)}",
            )
        )
    chiller_output_power_legend = ax.legend(
        handles=chiller_output_power_legend_handles, loc="upper right"
    )
    ax.add_artist(chiller_output_power_legend)

    ax.set_ylabel("MW")
    ax.set_yticks([0, 2, 4, 12])


def plot_input_power_legends(ax, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    chiller_input_power_legend_handles = []
    for chiller in chillers:
        chiller_input_power_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=chiller.colour,
                linestyle=chiller.model_line_style,
                label=f"Power consumption of chiller {chiller.number}"
                f" - {get_chiller_short_label(chiller)}",
            )
        )
    chiller_input_power_legend = ax.legend(
        handles=chiller_input_power_legend_handles, loc="upper center"
    )
    ax.add_artist(chiller_input_power_legend)

    ax.set_ylabel("MW")
    ax.set_yticks([0, 0.6, 2, 4, 6])


def plot_volumetric_flow_legends(ax, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    model_supply_line = mlines.Line2D(
        [],
        [],
        color=supply_colour,
        linestyle=supply_dash,
        label=f"Volumetric flow of supply water to the hospital"
        f" - {supply_short_label}",
    )
    model_plant_line = mlines.Line2D(
        [],
        [],
        color=plant_colour,
        linestyle=plant_dash,
        label=f"Volumetric flow at the cooling plant" f" - {plant_short_label}",
    )
    model_active_line = mlines.Line2D(
        [],
        [],
        color=active_flow_colour,
        linestyle=active_flow_dash,
        label=f"Total volumetric flow through active chillers"
        f" - {active_flow_short_label}",
    )
    model_active_approximation_line = mlines.Line2D(
        [],
        [],
        color=active_flow_colour,
        linestyle=active_approximation_dot,
        label=f"Approximation of total volumetric flow through active chillers"
        f" - {active_flow_short_label}",
    )
    model_legend = ax.legend(
        handles=[
            model_supply_line,
            model_plant_line,
            model_active_line,
            model_active_approximation_line,
        ],
        loc="upper left",
    )
    ax.add_artist(model_legend)
    chiller_volumetric_flow_legend_handles = []
    for chiller in chillers:
        chiller_volumetric_flow_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=chiller.colour,
                linestyle=chiller.model_line_style,
                label=f"Volumetric flow through "
                f"chiller {chiller.number}"
                f" - "
                f"{get_chiller_short_label(chiller)}",
            )
        )
    chiller_volumetric_flow_legend = ax.legend(
        handles=chiller_volumetric_flow_legend_handles, loc="upper right"
    )
    ax.add_artist(chiller_volumetric_flow_legend)

    ax.set_ylabel("m$^3$/h")
    ax.set_yticks([0, 140, 330, 500, 750, 1000, 1250])


def add_figure_legends(fig, axs, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    model_supply_line = mlines.Line2D(
        [],
        [],
        color=supply_colour,
        linestyle=supply_dash,
        label=f"{supply_short_label}",
    )
    model_return_line = mlines.Line2D(
        [],
        [],
        color=return_colour,
        linestyle=return_dash,
        label=f"{return_short_label}",
    )
    model_river_chilled_line = mlines.Line2D(
        [],
        [],
        color=river_chilled_water_colour,
        linestyle=river_chilled_water_dash,
        label=f"{river_chilled_water_short_label}",
    )
    model_river_line = mlines.Line2D(
        [], [], color=river_colour, linestyle=river_dash, label=f"{river_short_label}"
    )
    model_plant_line = mlines.Line2D(
        [], [], color=plant_colour, linestyle=plant_dash, label=f"{plant_short_label}"
    )
    model_active_line = mlines.Line2D(
        [],
        [],
        color=active_flow_colour,
        linestyle=active_flow_dash,
        label=f"{active_flow_short_label}",
    )
    chiller_legend_handles = []
    for chiller in chillers:
        chiller_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=chiller.colour,
                linestyle=chiller.model_line_style,
                label=f"{get_chiller_short_label(chiller)}",
            )
        )

    i = 0
    ax = axs[i]
    ax.legend(
        handles=[
            model_supply_line,
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    i += 1
    ax = axs[i]
    ax.legend(
        handles=[
            model_return_line,
            model_river_chilled_line,
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    i += 1
    ax = axs[i]
    ax.legend(
        handles=[
            model_supply_line,
            model_river_line,
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    i += 1
    ax = axs[i]
    ax.legend(
        handles=[
            *chiller_legend_handles,
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    i += 1
    ax = axs[i]
    ax.legend(
        handles=[
            model_supply_line,
            model_plant_line,
            model_active_line,
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    fig.tight_layout()


def plot_historical_water_temperatures(
    ax,
    dataframe,
    outputs_only=False,
    inputs_only=False,
    thick_supply_water_line=True,
):
    py_index = dataframe.index.to_pydatetime()
    if not inputs_only:
        conditional_arguments = {}
        if thick_supply_water_line:
            conditional_arguments["linewidth"] = 3
        ax.plot(
            py_index, dataframe["CT12"], color=supply_colour, **conditional_arguments
        )
    if not outputs_only:
        ax.plot(
            py_index,
            dataframe["CT11"],
            color=return_colour,
        )
    ax.set_ylabel("$\degree$C")
    ax.grid(True)


def plot_historical_cooling_load(ax, dataframe):
    py_index = dataframe.index.to_pydatetime()
    ax.plot(py_index, dataframe["Q11"], color=supply_colour)
    ax.set_ylabel("MW")
    ax.grid(True)


def plot_historical_chiller_output_power(
    ax,
    dataframe,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = dataframe.index.to_pydatetime()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            ax.plot(
                py_index,
                dataframe[chiller.effect_apis_id],
                color=chiller.colour,
            )
    ax.set_ylabel("MW")
    ax.grid(True)


def plot_historical_chiller_input(
    ax,
    dataframe,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = dataframe.index.to_pydatetime()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            ax.plot(
                py_index,
                dataframe[chiller.input_power_apis_id]
                * chiller.input_power_coefficient,
                color=chiller.colour,
            )
    ax.set_ylabel("MW")
    ax.grid(True)


def plot_historical_chiller_output_temperatures(
    ax, dataframe, chillers=None, only_chiller_number=None
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = dataframe.index.to_pydatetime()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            ax.plot(
                py_index,
                dataframe[chiller.output_temperature_apis_id],
                color=chiller.colour,
            )
    ax.set_ylabel("$\degree$C")
    ax.grid(True)


def plot_historical_chiller_input_temperatures(
    ax,
    dataframe,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = dataframe.index.to_pydatetime()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            ax.plot(
                py_index,
                dataframe[chiller.input_temperature_apis_id],
                color=chiller.colour,
            )
    ax.set_ylabel("$\degree$C")
    ax.grid(True)


def plot_historical_volumetric_flow(ax, dataframe):
    py_index = dataframe.index.to_pydatetime()
    ax.plot(py_index, dataframe["CF11"], color=supply_colour)
    ax.set_ylabel("m$^3$/h")
    ax.grid(True)


def plot_model_volumetric_flows(
    ax,
    model,
    chillers=None,
    plot_chillers=False,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    supply_volumetric_values = (
        model.supply_water_volumetric_flow.extract_values().values()
    )
    ax.plot(
        py_index,
        supply_volumetric_values,
        color=supply_colour,
        linestyle=supply_dash,
    )
    plant_volumetric_values = model.plant_volumetric_flow.extract_values().values()
    ax.plot(
        py_index,
        plant_volumetric_values,
        color=plant_colour,
        linestyle=plant_dash,
    )
    if plot_chillers:
        all_chiller_flow_modes = model.chiller_mode.extract_values()
        all_chiller_active_flow_approximation = (
            model.active_chillers_volumetric_flow_approximation.extract_values()
        )
        active_flow = [0 for _ in model.time]
        for chiller in chillers:
            chiller_flow = [
                all_chiller_flow_modes[(tau, chiller.number)]
                * chiller.max_volumetric_flow
                for tau in model.time
            ]
            ax.plot(
                py_index,
                chiller_flow,
                color=chiller.colour,
                linestyle=chiller.model_line_style,
            )
            active_flow_approximation = [
                all_chiller_active_flow_approximation[(tau, chiller.number)]
                for tau in model.time
            ]
            ax.plot(
                py_index,
                active_flow_approximation,
                color=active_flow_colour,
                linestyle=active_approximation_dot,
            )
            for i, tau in enumerate(model.time):
                active_flow[i] += (
                    all_chiller_flow_modes[(tau, chiller.number)]
                    * chiller.max_volumetric_flow
                )
        ax.plot(
            py_index,
            active_flow,
            color=active_flow_colour,
            linestyle=active_flow_dash,
        )

    ax.set_ylabel("m$^3$/h")
    ax.grid(True)
    ax.margins(x=0)


def plot_model_chiller_input(
    ax,
    model,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    all_chiller_values = model.chiller_input_power.extract_values()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            chiller_input = [
                all_chiller_values[(tau, chiller.number)] for tau in model.time
            ]
            ax.plot(
                py_index,
                chiller_input,
                color=chiller.colour,
                linestyle=chiller.model_line_style,
            )
    ax.set_ylabel("MW")
    ax.grid(True)
    ax.margins(x=0)


def plot_model_chiller_output_power(
    ax,
    model,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    all_chiller_values = model.chiller_output_power.extract_values()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            chiller_output_power = [
                all_chiller_values[(tau, chiller.number)] for tau in model.time
            ]
            ax.plot(
                py_index,
                chiller_output_power,
                color=chiller.colour,
                linestyle=chiller.model_line_style,
            )
    ax.set_ylabel("MW")
    ax.grid(True)
    ax.margins(x=0)


def plot_historical_river_chilling_power(ax, dataframe):
    py_index = dataframe.index.to_pydatetime()
    ax.plot(py_index, dataframe["Q12"], color=river_colour)
    ax.set_ylabel("MW")
    ax.grid(True)


def plot_model_river_chilling_power(ax, model):
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    river_chilling_power = [
        model.parameters.get_river_chiller_output_power(model, tau)
        for tau in model.time
    ]
    ax.plot(py_index, river_chilling_power, color=river_colour, linestyle="dashed")
    ax.set_ylabel("MW")
    ax.grid(True)


def plot_model_cooling_load(ax, model):
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    cooling_load = [model.parameters.get_cooling_load(model, tau) for tau in model.time]
    ax.plot(
        py_index,
        cooling_load,
        color=supply_colour,
        linestyle=supply_dash,
    )
    ax.set_ylabel("MW")
    ax.grid(True)


def plot_model_water_temperatures(
    ax,
    model,
    outputs_only=False,
    inputs_only=False,
    thick_supply_water_line=False,
):
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    if not inputs_only:
        conditional_arguments = {}
        if thick_supply_water_line:
            conditional_arguments["linewidth"] = 3
        supply_water_temperature = [
            model.parameters.get_supply_water_temperature(model, tau)
            for tau in model.time
        ]
        ax.plot(
            py_index,
            supply_water_temperature,
            color=supply_colour,
            linestyle=supply_dash,
            **conditional_arguments,
        )
    if not outputs_only:
        ax.plot(
            py_index,
            model.return_water_temperature.extract_values().values(),
            color=return_colour,
            linestyle=return_dash,
        )
        river_chilled_water_temperature = [
            model.parameters.get_river_chilled_water_temperature(model, tau)
            for tau in model.time
        ]
        ax.plot(
            py_index,
            river_chilled_water_temperature,
            color=river_chilled_water_colour,
            linestyle=river_chilled_water_dash,
        )
    ax.set_ylabel("$\degree$C")
    ax.grid(True)
    ax.margins(x=0)


def plot_model_chiller_output_temperatures(
    ax,
    model,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    chillers_to_plot = []
    if only_chiller_number is not None:
        chillers_to_plot.append(only_chiller_number)

    all_chiller_values = model.chiller_output_water_temperature.extract_values()
    for chiller in chillers:
        if chiller.number in chillers_to_plot:
            chiller_output_temperature = [
                all_chiller_values[(tau, chiller.number)] for tau in model.time
            ]
            ax.plot(
                py_index,
                chiller_output_temperature,
                color=chiller.colour,
                linestyle=chiller.model_line_style,
            )
    ax.set_ylabel("$\degree$C")
    ax.grid(True)


def plot_model_chiller_input_temperatures(
    ax,
    model,
    chillers=None,
    only_chiller_number=None,
):
    if chillers is None:
        chillers = get_default_chillers()
    py_index = [
        model.time_horizon_start_date.value.to_pydatetime() + pd.Timedelta(minutes=tau)
        for tau in model.time
    ]
    all_chiller_values = model.chiller_input_water_temperature.extract_values()
    for chiller in chillers:
        if only_chiller_number is None or chiller.number == only_chiller_number:
            chiller_input_temperature = [
                all_chiller_values[(tau, chiller.number)] for tau in model.time
            ]
            ax.plot(
                py_index,
                chiller_input_temperature,
                color=chiller.colour,
                linestyle=chiller.model_line_style,
            )
    ax.set_ylabel("$\degree$C")
    ax.grid(True)


def reformat_time_axis(ax):
    tick_labels = ax.get_xticklabels()
    for i, tick_label in enumerate(tick_labels):
        if i % 2 != 0:
            tick_label.set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.set_xlabel("Time")
    ax.margins(x=0)
