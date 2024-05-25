import streamlit as st
import pyomo.environ as pyo

from src.oks.MPC import initialization, actions
from src.oks.OSS import data, plotters

progress_bar = st.progress(0, text="Initializing...")

failed_optimization_text = st.empty()

chiller_mode_text = st.empty()

plot_holder = st.empty()

optimization_timelimit = 60
offset = 0


def update_measurements():
    """
    This function should retrieve new measurements from APIS. In the demo version, it just increments or resets the
    offset.
    :return:
    """
    global offset
    offset += 1
    if offset > 50:
        offset = 0


while True:
    progress_bar.progress(1, text="Loading data...")
    update_measurements()
    optimization_data = data.get_optimization_data(offset=offset)

    progress_bar.progress(5, text="Setting up model...")
    model = initialization.get_model(optimization_data=optimization_data)

    progress_bar.progress(10, text="Optimizing...")
    solver = pyo.SolverFactory("cbc.exe")
    try:
        results = solver.solve(model, timelimit=optimization_timelimit)
    except Exception:
        failed_optimization_text.markdown(
            f"Failed to optimize in {optimization_timelimit} seconds. "
            f"Trying again with new measurements..."
        )
        continue
    else:
        if results.solver.status == pyo.SolverStatus.ok:
            failed_optimization_text.empty()
        else:
            failed_optimization_text.markdown(
                "Optimization failed. Trying again with new measurements..."
            )
            continue

    progress_bar.progress(93, text="Recommending chiller actions...")
    with chiller_mode_text:
        chiller_mode_description = actions.get_chiller_mode_description(
            model=model, chillers=model.parameters.chillers
        )
        chiller_mode_text.markdown(chiller_mode_description)

    progress_bar.progress(95, text="Plotting...")
    fig = plotters.get_standard_plot(
        plot_data=optimization_data, model=model, chillers=model.parameters.chillers
    )

    with plot_holder:
        st.pyplot(fig)
    progress_bar.empty()
