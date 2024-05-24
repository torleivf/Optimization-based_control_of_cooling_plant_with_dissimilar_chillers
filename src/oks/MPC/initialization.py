from pyomo import environ as pyo

from src.oks.MPC import model_parameters
from src.oks.MPC.models import set_model_constraints
from src.oks.MPC.objective import (
    set_optimization_constraints,
    set_optimization_objective,
)


def get_model(
    parameters,
):
    model = initialize_model(parameters=parameters)

    initialize_model_parameters(model=model)

    # Actuators
    model.chiller_mode = pyo.Var(model.time, model.chiller, domain=pyo.Binary)

    # Variables
    model.supply_water_temperature = pyo.Var(model.time)
    model.supply_water_heating = pyo.Var(model.time)
    model.plant_volumetric_flow = pyo.Var(model.time)
    model.return_water_heating = pyo.Var(model.time)
    model.return_water_overflow_heating = pyo.Var(model.time)
    model.chiller_input_power = pyo.Var(model.time, model.chiller)
    model.chiller_input_power_steady_state = pyo.Var(model.time, model.chiller)
    model.chiller_output_power = pyo.Var(model.time, model.chiller)
    model.chiller_supply_water_heating = pyo.Var(model.time, model.chiller)
    model.chiller_output_water_temperature = pyo.Var(model.time, model.chiller)
    model.chiller_output_water_heating = pyo.Var(model.time, model.chiller)
    model.chiller_input_water_temperature = pyo.Var(model.time, model.chiller)
    model.chiller_input_water_heating = pyo.Var(model.time, model.chiller)
    model.chiller_input_water_return_heating = pyo.Var(model.time, model.chiller)
    model.chiller_input_water_backflow_heating_subtraction = pyo.Var(
        model.time, model.chiller
    )

    model.cooling_load = pyo.Var(model.time)

    # Layered variables
    model.chiller_return_water_overflow_heating_subtraction = pyo.Var(
        model.time, model.chiller
    )

    # Constraint helpers
    model.chiller_startup_mode = pyo.Var(model.time, model.chiller, domain=pyo.Binary)
    model.supply_water_temperature_below_setpoint = pyo.Var(
        model.time, domain=pyo.Binary
    )
    model.return_water_overflow_binary = pyo.Var(model.time, domain=pyo.Binary)

    # Slack variables
    model.supply_water_temperature_violation = pyo.Var(
        model.time, domain=pyo.NonNegativeReals
    )
    model.cooling_load_capacity_violation = pyo.Var(
        model.time, domain=pyo.NonNegativeReals
    )

    # Set constraints and objective function
    set_model_constraints(model)

    set_optimization_constraints(model)

    set_optimization_objective(
        model,
    )

    return model


def initialize_model(
    parameters: model_parameters.ModelParameters,
) -> pyo.ConcreteModel:
    """
    Initialize pyomo model with context parameters and sets.
    :param parameters: ModelParameters for context
    :return:
    """
    model = pyo.ConcreteModel()

    # Context parameters
    model.time_horizon_start_date = pyo.Param(
        initialize=parameters.get_time_horizon_start_date, within=pyo.Any
    )
    model.time_step = pyo.Param(initialize=parameters.get_time_step)
    model.time_horizon = pyo.Param(initialize=parameters.get_time_horizon)

    # Sets
    model.time = pyo.RangeSet(0, model.time_horizon, model.time_step)
    model.chiller = pyo.RangeSet(parameters.get_chiller_count())

    # All parameters
    model.parameters = parameters

    return model


def initialize_model_parameters(
    model: pyo.ConcreteModel,
):
    parameters = model.parameters
    # Cooling plant parameters
    model.maximum_supply_water_temperature = pyo.Param(
        initialize=parameters.get_maximum_supply_water_temperature
    )
    model.maximum_chiller_capacity = pyo.Param(
        model.chiller, initialize=parameters.get_maximum_chiller_capacity
    )
    model.maximum_chiller_volumetric_flow = pyo.Param(
        model.chiller, initialize=parameters.get_maximum_chiller_volumetric_flow
    )
    model.chiller_energy_cost = pyo.Param(
        model.chiller, initialize=parameters.get_chiller_energy_cost
    )
    model.input_output_effect_lag_timesteps = pyo.Param(
        model.chiller, initialize=parameters.get_input_output_effect_lag_timesteps
    )
    model.chiller_average_coefficient_of_performance = pyo.Param(
        model.chiller, initialize=parameters.get_average_coefficient_of_performance
    )

    model.water_density = pyo.Param(initialize=parameters.get_water_density)
    model.water_heat_capacity = pyo.Param(initialize=parameters.get_water_heat_capacity)

    # Initial parameters
    model.initial_supply_water_temperature = pyo.Param(
        initialize=parameters.get_initial_supply_water_temperature
    )
    model.initial_chiller_output_water_temperature = pyo.Param(
        model.chiller,
        initialize=parameters.get_initial_chiller_output_water_temperature,
    )
    model.initial_chiller_input_water_temperature = pyo.Param(
        model.chiller, initialize=parameters.get_initial_chiller_input_water_temperature
    )
    model.initial_chiller_flow_mode = pyo.Param(
        model.chiller, initialize=parameters.get_initial_chiller_mode
    )
    model.initial_chiller_input_power = pyo.Param(
        model.chiller, initialize=parameters.get_initial_chiller_input_power
    )
    model.initial_chiller_output_power = pyo.Param(
        model.chiller, initialize=parameters.get_initial_chiller_output_power
    )

    # Predicted parameters
    model.cooling_load_demand = pyo.Param(
        model.time, initialize=parameters.get_cooling_load_demand
    )
    model.supply_water_volumetric_flow = pyo.Param(
        model.time, initialize=parameters.get_supply_water_volumetric_flow
    )
    model.active_chillers_volumetric_flow_approximation = pyo.Param(
        model.time,
        model.chiller,
        initialize=parameters.get_active_chillers_volumetric_flow_approximation,
    )
    model.return_water_temperature = pyo.Param(
        model.time, initialize=parameters.get_return_water_temperature
    )
