import pyomo.environ as pyo

from . import model_parameters

epsilon = 1e-12


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
        model.chiller, initialize=parameters.get_initial_chiller_output_water_temperature
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


def initialize_model_constraints(
    m,
    supply_water_temperature_is_var=True,
    chiller_output_water_temperature_is_var=True,
    chiller_input_water_temperature_is_var=True,
    chiller_output_power_is_var=True,
    chiller_input_power_is_var=True,
    startup_mode_is_var=True,
):
    if supply_water_temperature_is_var:
        m.supply_water_heating_constraint = pyo.Constraint(
            m.time, rule=_supply_water_heating
        )
        set_plant_volumetric_flow_constraints(m)
        set_chiller_supply_water_heating_constraints(m)
        m.return_water_overflow_binary_min_constraint = pyo.Constraint(
            m.time, rule=_return_water_overflow_binary_min
        )
        m.return_water_overflow_binary_max_constraint = pyo.Constraint(
            m.time, rule=_return_water_overflow_binary_max
        )
        set_return_water_overflow_heating_constraints(m)
    if chiller_output_water_temperature_is_var:
        m.chiller_output_water_temperature_constraint = pyo.Constraint(
            m.time, m.chiller, rule=_chiller_output_water_temperature
        )
        set_chiller_output_water_heating_constraints(m)
    if chiller_input_water_temperature_is_var:
        m.chiller_input_water_temperature_constraint = pyo.Constraint(
            m.time, m.chiller, rule=_chiller_input_water_temperature
        )
        set_input_water_heating_constraints(m)
    if chiller_output_power_is_var:
        m.chilling_output_power_constraint = pyo.Constraint(
            m.time, m.chiller, rule=_chiller_output_power
        )
    if chiller_input_power_is_var:
        set_chiller_input_power_constraints(m)
    if startup_mode_is_var:
        set_chiller_startup_mode_constraints(m)


def initialize_optimization_objective(
    model,
):
    # Objective function coefficients
    h_per_time_step = model.time_step / 60
    model.supply_temperature_violation_cost_multiplier = pyo.Param(default=1000000)
    model.chiller_startup_cost_multiplier = pyo.Param(default=100000)
    model.chiller_energy_cost_multiplier = pyo.Param(default=h_per_time_step)
    model.capacity_violation_cost_multiplier = pyo.Param(default=400)
    model.supply_water_temperature_below_setpoint_cost_multiplier = pyo.Param(
        default=50000
    )

    # Objective function rule
    def _objective(_model):
        supply_water_violation_cost = (
                _model.supply_temperature_violation_cost_multiplier
                * sum(_model.supply_water_temperature_violation[t] for t in _model.time)
        )

        chiller_startup_cost = _model.chiller_startup_cost_multiplier * sum(
            _model.chiller_startup_mode[t, c] for c in _model.chiller for t in _model.time
        )

        energy_cost = _model.chiller_energy_cost_multiplier * sum(
            _model.chiller_input_power[t, c] * _model.chiller_energy_cost[c]
            for c in _model.chiller
            for t in _model.time
        )

        cooling_load_capacity_violation_cost = (
                _model.capacity_violation_cost_multiplier
                * sum(_model.cooling_load_capacity_violation[t] for t in _model.time)
        )

        supply_water_temperature_below_setpoint_cost = (
                _model.supply_water_temperature_below_setpoint_cost_multiplier
                * sum(_model.supply_water_temperature_below_setpoint[t] for t in _model.time)
        )

        return (
            supply_water_violation_cost
            + energy_cost
            + cooling_load_capacity_violation_cost
            + chiller_startup_cost
            + supply_water_temperature_below_setpoint_cost
        )

    model.objective = pyo.Objective(rule=_objective)


def get_model(
    parameters,
):
    model = initialize_model(parameters=parameters)

    initialize_model_parameters(
        model=model)

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
    model.chiller_input_water_backflow_heating_subtraction = pyo.Var(model.time, model.chiller)

    model.cooling_load = pyo.Var(model.time)

    # Layered variables
    model.chiller_return_water_overflow_heating_subtraction = pyo.Var(model.time, model.chiller)

    # Constraint helpers
    model.chiller_startup_mode = pyo.Var(model.time, model.chiller, domain=pyo.Binary)
    model.supply_water_temperature_below_setpoint = pyo.Var(model.time, domain=pyo.Binary)
    model.return_water_overflow_binary = pyo.Var(model.time, domain=pyo.Binary)

    # Slack variables
    model.supply_water_temperature_violation = pyo.Var(model.time, domain=pyo.NonNegativeReals)
    model.cooling_load_capacity_violation = pyo.Var(model.time, domain=pyo.NonNegativeReals)

    # Model constraints
    initialize_model_constraints(model)

    # Optimization constraints
    model.initial_chiller_mode_constraint = pyo.Constraint(
        model.chiller, rule=_initial_chiller_mode
    )
    model.supply_water_temperature_violation_constraint = pyo.Constraint(
        model.time, rule=_supply_water_temperature_violation
    )
    model.active_chiller_capacity_heuristic_constraint = pyo.Constraint(
        model.time, rule=_active_chiller_capacity_heuristic
    )

    model.only_start_chiller_once_constraint = pyo.Constraint(
        model.chiller, rule=_only_start_chiller_once
    )

    model.chiller_mode_change_block_constraint = pyo.Constraint(
        model.time, model.chiller, rule=_only_change_chiller_mode_during_initial_time_steps
    )

    initialize_optimization_objective(
        model,
    )

    return model


def _initial_chiller_mode(m, c):
    return m.chiller_mode[0, c] == m.initial_chiller_flow_mode[c]


def get_chiller_flow_mode(m, t, c):
    if t < 0:
        return m.parameters.get_historic_chiller_flow_mode(
            _model=m,
            time=t,
            chiller=c,
        )
    return m.chiller_mode[t, c]


def get_chiller_input_mode(m, t, c):
    flow_mode_equivalent_time = (
        t
        - m.parameters.get_startup_flow_timesteps(
            _model=m,
            chiller=c,
        )
        * m.time_step
    )
    return get_chiller_flow_mode(m, flow_mode_equivalent_time, c)


def get_chiller_input_power(m, t, c):
    if t < 0:
        return m.parameters.get_historic_chiller_input_power(
            _model=m,
            time=t,
            chiller=c,
        )
    return m.chiller_input_power[t, c]


def get_plant_volumetric_flow_variable(m, t, c=None):
    return m.plant_volumetric_flow[t]


def get_plant_volumetric_flow_binary_variable(m, t, c=None):
    return m.return_water_overflow_binary[t]


def get_total_active_chiller_volumetric_flow(m, t, c=None):
    return sum(
        m.maximum_chiller_volumetric_flow[c] * get_chiller_flow_mode(m, t, c)
        for c in m.chiller
    )


def get_supply_water_volumetric_flow(m, t, c=None):
    return m.supply_water_volumetric_flow[t]


def get_maximum_plant_volumetric_flow(m, t, c=None):
    return (
        sum(m.maximum_chiller_volumetric_flow[c] for c in m.chiller)
        + m.supply_water_volumetric_flow[t]
    )


def get_plant_volumetric_flow_big_m(m, t, c=None):
    return 2 * get_maximum_plant_volumetric_flow(m, t, c)


def set_plant_volumetric_flow_constraints(m):
    # set internal supply water flow as supply water flow when there is return water overflow
    # set internal supply water flow as active chiller flow when there is no return water overflow
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_plant_volumetric_flow_variable,
        binary_variable_function=get_plant_volumetric_flow_binary_variable,
        function=get_supply_water_volumetric_flow,
        variable_maximum_function=get_plant_volumetric_flow_big_m,
        constraint_name_root="_plant_volumetric_flow",
        second_function=get_total_active_chiller_volumetric_flow,
    )


def _supply_water_heating(m, t):
    return m.supply_water_heating[t] == m.return_water_overflow_heating[t] + sum(
        m.chiller_supply_water_heating[t, c] for c in m.chiller
    )


def get_overflow(m, t, c=None):
    return m.supply_water_volumetric_flow[t] - sum(
        m.maximum_chiller_volumetric_flow[c] * get_chiller_flow_mode(m, t, c)
        for c in m.chiller
    )


def get_overflow_big_m_max(m, t, c=None):
    return m.supply_water_volumetric_flow[t]


def get_overflow_big_m_min(m, t, c=None):
    return m.supply_water_volumetric_flow[t] - sum(
        m.maximum_chiller_volumetric_flow[c] for c in m.chiller
    )


def _return_water_overflow_binary_min(m, t):
    return (
        get_overflow(m, t)
        <= get_overflow_big_m_max(m, t) * m.return_water_overflow_binary[t]
    )


def _return_water_overflow_binary_max(m, t):
    return get_overflow(m, t) >= epsilon + (get_overflow_big_m_min(m, t) - epsilon) * (
        1 - m.return_water_overflow_binary[t]
    )


# Big M regional function constraints

def apply_model_constraint(model, root_name, chiller_constraint=False):
    def decorator(function):
        function.__name__ = f"{root_name}{function.__name__}"
        if chiller_constraint:
            function = model.Constraint(model.time, model.chiller)(function)
        else:
            function = model.Constraint(model.time)(function)
        return function

    return decorator


def set_big_m_time_function_constraints(
    model,
    function_variable_function,
    binary_variable_function,
    function,
    variable_maximum_function,
    constraint_name_root,
    variable_minimum_function=None,
    chiller_constraint=False,
    second_function=None,
    skip_zero=False,
):
    """
    Set constraints forcing function_variable to be zero (or second function) when binary_variable is zero and to equal
    function when binary_variable is one.
    """
    if second_function is None:

        def second_function(m, t, c=None):
            return 0

    if variable_minimum_function is None:

        def variable_minimum_function(m, t, c=None):
            return -variable_maximum_function(m, t, c)

    @apply_model_constraint(
        model, constraint_name_root, chiller_constraint=chiller_constraint
    )
    def _min_constraint(m, t, c=None):
        if skip_zero:
            if t == 0:
                return pyo.Constraint.Skip
        return function_variable_function(m, t=t, c=c) >= second_function(
            m, t=t, c=c
        ) + variable_minimum_function(m, t=t, c=c) * binary_variable_function(
            m, t=t, c=c
        )

    @apply_model_constraint(
        model, constraint_name_root, chiller_constraint=chiller_constraint
    )
    def _max_constraint(m, t, c=None):
        if skip_zero:
            if t == 0:
                return pyo.Constraint.Skip
        return function_variable_function(m, t=t, c=c) <= second_function(
            m, t=t, c=c
        ) + variable_maximum_function(m, t=t, c=c) * binary_variable_function(
            m, t=t, c=c
        )

    @apply_model_constraint(
        model, constraint_name_root, chiller_constraint=chiller_constraint
    )
    def _f_min_constraint(m, t, c=None):
        if skip_zero:
            if t == 0:
                return pyo.Constraint.Skip
        return function_variable_function(m, t=t, c=c) >= function(
            m, t=t, c=c
        ) - variable_maximum_function(m, t=t, c=c) * (
            1 - binary_variable_function(m, t=t, c=c)
        )

    @apply_model_constraint(
        model, constraint_name_root, chiller_constraint=chiller_constraint
    )
    def _f_max_constraint(m, t, c=None):
        if skip_zero:
            if t == 0:
                return pyo.Constraint.Skip
        return function_variable_function(m, t=t, c=c) <= function(
            m, t=t, c=c
        ) - variable_minimum_function(m, t=t, c=c) * (
            1 - binary_variable_function(m, t=t, c=c)
        )


# Return water overflow heating


def get_return_water_overflow_heating_chiller_subtraction_variable(m, t, c=None):
    return m.chiller_return_water_overflow_heating_subtraction[t, c]


def get_return_water_overflow_chiller_subtraction_binary_variable(m, t, c=None):
    return get_chiller_flow_mode(m, t, c)


def get_overflow_heating_chiller_subtraction_big_m(m, t, c=None):
    return (
        100
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )


def get_overflow_heating_chiller_subtraction(m, t, c=None):
    overflow_heating_subtraction = (
        m.parameters.get_river_chilled_water_temperature(m, t)
        * m.water_density
        * m.water_heat_capacity
        * m.maximum_chiller_volumetric_flow[c]
    )
    return overflow_heating_subtraction


def get_return_water_overflow_heating_variable(m, t, c=None):
    return m.return_water_overflow_heating[t]


def get_return_water_overflow_binary_variable(m, t, c=None):
    return m.return_water_overflow_binary[t]


def get_overflow_heating_big_m(m, t, c=None):
    maximum_overflow = m.supply_water_volumetric_flow[t] + sum(
        m.maximum_chiller_volumetric_flow[c] for c in m.chiller
    )
    maximum_overflow_heating = (
        100 * m.water_density * m.water_heat_capacity * maximum_overflow
    )
    return maximum_overflow_heating


def get_overflow_heating(m, t, c=None):
    overflow_heating = m.parameters.get_river_chilled_water_temperature(
        m, t
    ) * m.water_density * m.water_heat_capacity * m.supply_water_volumetric_flow[
        t
    ] - sum(
        m.chiller_return_water_overflow_heating_subtraction[t, c] for c in m.chiller
    )
    return overflow_heating


def set_return_water_overflow_heating_constraints(m):
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_return_water_overflow_heating_chiller_subtraction_variable,
        binary_variable_function=get_return_water_overflow_chiller_subtraction_binary_variable,
        function=get_overflow_heating_chiller_subtraction,
        variable_maximum_function=get_overflow_heating_chiller_subtraction_big_m,
        constraint_name_root="_return_water_overflow_heating_chiller_subtraction",
        chiller_constraint=True,
    )
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_return_water_overflow_heating_variable,
        binary_variable_function=get_return_water_overflow_binary_variable,
        function=get_overflow_heating,
        variable_maximum_function=get_overflow_heating_big_m,
        constraint_name_root="_return_water_overflow_heating",
    )


# chiller supply water chilling


def get_chiller_supply_water_heating_variable(m, t, c):
    return m.chiller_supply_water_heating[t, c]


def get_chiller_supply_water_heating_binary_variable(m, t, c):
    return get_chiller_flow_mode(m, t, c)


def get_chiller_supply_water_heating_big_m(m, t, c):
    return (
        100
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )


def get_chiller_supply_water_heating(m, t, c):
    heating = (
        m.chiller_output_water_temperature[t, c]
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )
    return heating


def set_chiller_supply_water_heating_constraints(m):
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_chiller_supply_water_heating_variable,
        binary_variable_function=get_chiller_supply_water_heating_binary_variable,
        function=get_chiller_supply_water_heating,
        variable_maximum_function=get_chiller_supply_water_heating_big_m,
        constraint_name_root="_chiller_supply_water_heating",
        chiller_constraint=True,
    )


# chiller input water backflow


def get_supply_water_temperature_approximation(m, t, c=None):
    return m.chiller_output_water_temperature[t, c]


def get_active_flow_approximation(m, t, c=None):
    return m.active_chillers_volumetric_flow_approximation[t, c]


def get_chiller_input_water_backflow_heating_subtraction_variable(m, t, c):
    return m.chiller_input_water_backflow_heating_subtraction[t, c]


def get_chiller_input_water_backflow_heating_subtraction_binary_variable(m, t, c):
    return 1 - m.return_water_overflow_binary[t]


def get_chiller_input_water_backflow_heating_subtraction_big_m(m, t, c):
    max_absolute_chiller_backflow = (
        m.maximum_chiller_volumetric_flow[c] + m.supply_water_volumetric_flow[t]
    )
    return 100 * max_absolute_chiller_backflow * m.water_density * m.water_heat_capacity


def get_backflow_approximation(m, t, c=None):
    return get_active_flow_approximation(m, t, c) - m.supply_water_volumetric_flow[t]


def get_chiller_backflow_approximation(m, t, c):
    backflow = get_backflow_approximation(m, t, c)
    chiller_backflow_portion = m.maximum_chiller_volumetric_flow[
        c
    ] / get_active_flow_approximation(m, t, c)
    chiller_backflow = backflow * chiller_backflow_portion
    return chiller_backflow


def get_chiller_input_water_backflow_heating_subtraction(m, t, c):
    chiller_backflow = get_chiller_backflow_approximation(m, t, c)
    chiller_backflow_heating = (
        get_supply_water_temperature_approximation(m, t, c)
        * chiller_backflow
        * m.water_density
        * m.water_heat_capacity
    )
    chiller_backflow_heating_subtraction = (
        m.parameters.get_river_chilled_water_temperature(m, t)
        * chiller_backflow
        * m.water_density
        * m.water_heat_capacity
    ) - chiller_backflow_heating
    return chiller_backflow_heating_subtraction


def get_chiller_input_water_heating_variable(m, t, c):
    return m.chiller_input_water_heating[t, c]


def get_chiller_input_water_heating_binary_variable(m, t, c):
    return get_chiller_flow_mode(m, t, c)


def get_chiller_input_water_heating_big_m(m, t, c):
    return (
        100
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )


def get_chiller_input_water_heating(m, t, c):
    chiller_input_water_return_heating = (
        m.parameters.get_river_chilled_water_temperature(m, t)
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )
    return (
        chiller_input_water_return_heating
        - m.chiller_input_water_backflow_heating_subtraction[t, c]
    )


def get_inactive_chiller_input_water_heating(m, t, c):
    if t == 0:
        return (
            m.initial_chiller_input_water_temperature[c]
            * m.maximum_chiller_volumetric_flow[c]
            * m.water_density
            * m.water_heat_capacity
        )
    return (
        m.chiller_input_water_temperature[t - m.time_step, c]
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )


def set_input_water_heating_constraints(m):
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_chiller_input_water_backflow_heating_subtraction_variable,
        binary_variable_function=get_chiller_input_water_backflow_heating_subtraction_binary_variable,
        function=get_chiller_input_water_backflow_heating_subtraction,
        variable_maximum_function=get_chiller_input_water_backflow_heating_subtraction_big_m,
        constraint_name_root="_chiller_input_water_backflow_heating_subtraction",
        chiller_constraint=True,
    )
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_chiller_input_water_heating_variable,
        binary_variable_function=get_chiller_input_water_heating_binary_variable,
        function=get_chiller_input_water_heating,
        variable_maximum_function=get_chiller_input_water_heating_big_m,
        constraint_name_root="_chiller_input_water_heating",
        chiller_constraint=True,
        second_function=get_inactive_chiller_input_water_heating,
    )


def _chiller_input_water_temperature(m, t, c):
    if t == 0:
        return m.chiller_input_water_temperature[t, c] == (
            m.initial_chiller_input_water_temperature[c] * m.chiller_mode[t, c]
            + m.initial_supply_water_temperature * (1 - m.chiller_mode[t, c])
        )
    return m.chiller_input_water_temperature[t, c] == m.chiller_input_water_heating[
        t, c
    ] / (m.maximum_chiller_volumetric_flow[c] * m.water_density * m.water_heat_capacity)


def get_chiller_output_water_heating_variable(m, t, c):
    return m.chiller_output_water_heating[t, c]


def get_chiller_ouput_water_heating_binary_variable(m, t, c):
    return m.chiller_mode[t - m.time_step, c]


def get_chiller_output_water_big_m(m, t, c):
    return (
        100
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )


def get_chiller_output_water_heating(m, t, c):
    heating = (
        m.chiller_input_water_temperature[t - m.time_step, c]
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )
    cooling = m.chiller_output_power[t - m.time_step, c]
    return heating - cooling


def get_inactive_chiller_output_water_heating(m, t, c):
    return (
        m.chiller_output_water_temperature[t - m.time_step, c]
        * m.maximum_chiller_volumetric_flow[c]
        * m.water_density
        * m.water_heat_capacity
    )


def set_chiller_output_water_heating_constraints(m):
    set_big_m_time_function_constraints(
        model=m,
        function_variable_function=get_chiller_output_water_heating_variable,
        binary_variable_function=get_chiller_ouput_water_heating_binary_variable,
        function=get_chiller_output_water_heating,
        variable_maximum_function=get_chiller_output_water_big_m,
        constraint_name_root="_chiller_output_water_heating",
        chiller_constraint=True,
        second_function=get_inactive_chiller_output_water_heating,
        skip_zero=True,
    )


def _chiller_output_water_temperature(m, t, c):
    if t == 0:
        return m.chiller_output_water_temperature[
            t, c
        ] == m.parameters.get_initial_chiller_output_water_temperature(
            _model=m,
            chiller=c,
        )
    return m.chiller_output_water_temperature[t, c] == m.chiller_output_water_heating[
        t, c
    ] / (m.maximum_chiller_volumetric_flow[c] * m.water_density * m.water_heat_capacity)


def _chiller_output_power(m, t, c):
    if t == 0:
        return m.chiller_output_power[t, c] == m.initial_chiller_output_power[c]
    relevant_input_power = get_chiller_input_power(
        m, t - m.input_output_effect_lag_timesteps[c] * m.time_step, c
    )
    return m.chiller_output_power[t, c] == (
        m.chiller_output_power[t - m.time_step, c]
        + (
            relevant_input_power * m.chiller_average_coefficient_of_performance[c]
            -
            m.chiller_output_power[t - m.time_step, c]
        )
        * m.parameters.get_chiller_in_out_discrete_time_constant_coefficient(m, c)
    )

def get_supply_heat_exceeding_return_heat_big_m_max(m, t):
    return (
        (100 - m.parameters.get_supply_water_setpoint_temperature(m, t))
        * get_maximum_plant_volumetric_flow(m, t)
        * m.water_density
        * m.water_heat_capacity
    )


def get_supply_heat_exceeding_return_heat_big_m_min(m, t):
    return (
        -m.parameters.get_supply_water_setpoint_temperature(m, t)
        * get_maximum_plant_volumetric_flow(m, t)
        * m.water_density
        * m.water_heat_capacity
    )


def get_setpoint_heat_exceeding_supply_heat(m, t):
    supply_heat = m.supply_water_heating[t]
    supply_setpoint_heat = (
        m.parameters.get_supply_water_setpoint_temperature(m, t)
        * get_plant_volumetric_flow_variable(m, t)
        * m.water_density
        * m.water_heat_capacity
    )
    return supply_setpoint_heat - supply_heat


def _supply_water_temperature_below_setpoint_binary_min(m, t):
    return get_setpoint_heat_exceeding_supply_heat(
        m, t
    ) <= get_supply_heat_exceeding_return_heat_big_m_max(m, t) * (
        m.supply_water_temperature_below_setpoint[t]
    )


def _supply_water_temperature_below_setpoint_binary_max(m, t):
    return get_setpoint_heat_exceeding_supply_heat(m, t) >= epsilon + (
        get_supply_heat_exceeding_return_heat_big_m_min(m, t) - epsilon
    ) * (1 - m.supply_water_temperature_below_setpoint[t])


def get_chiller_input_power_steady_state_variable(m, t, c):
    return m.chiller_input_power_steady_state[t, c]


def get_chiller_input_power_steady_state_binary_variable(m, t, c):
    return get_chiller_input_mode(m, t, c)


def get_chiller_input_power_steady_state(m, t, c):
    high_steady_state = m.parameters.get_chiller_input_power_high_prediction(
        m, t, c
    ) * (1 - m.supply_water_temperature_below_setpoint[t])
    low_steady_state = (
        m.parameters.get_chiller_input_power_low_prediction(m, t, c)
        * m.supply_water_temperature_below_setpoint[t]
    )
    return high_steady_state + low_steady_state


def get_chiller_input_power_big_m(m, t, c):
    return m.parameters.get_chiller_input_power_high_prediction(m, t, c)


def _chiller_input_power(m, t, c):
    if t == 0:
        return m.chiller_input_power[
            t, c
        ] == m.parameters.get_initial_chiller_input_power(m, c)
    return m.chiller_input_power[t, c] == (
        get_chiller_input_power(m, t - m.time_step, c)
        + (
            m.chiller_input_power_steady_state[t, c]
            - get_chiller_input_power(m, t - m.time_step, c)
        )
        * m.parameters.get_chiller_input_power_evolution_discrete_time_constant_coefficient(
            m, t, c
        )
    )


def set_chiller_input_power_constraints(m_):
    m_.supply_water_temperature_below_setpoint_binary_min_constraint = pyo.Constraint(
        m_.time, rule=_supply_water_temperature_below_setpoint_binary_min
    )
    m_.supply_water_temperature_below_setpoint_binary_max_constraint = pyo.Constraint(
        m_.time, rule=_supply_water_temperature_below_setpoint_binary_max
    )
    set_big_m_time_function_constraints(
        model=m_,
        function_variable_function=get_chiller_input_power_steady_state_variable,
        binary_variable_function=get_chiller_input_power_steady_state_binary_variable,
        function=get_chiller_input_power_steady_state,
        variable_maximum_function=get_chiller_input_power_big_m,
        constraint_name_root="_chiller_input_power_steady_state",
        chiller_constraint=True,
    )
    m_.input_power_constraint = pyo.Constraint(
        m_.time, m_.chiller, rule=_chiller_input_power
    )


# Binary constraints

def set_chiller_startup_mode_constraints(m):
    m.chiller_startup_upper_mode_constraint = pyo.Constraint(
        m.time, m.chiller, rule=_chiller_startup_upper_mode
    )
    m.chiller_startup_lower_history_constraint = pyo.Constraint(
        m.time, m.chiller, rule=_chiller_startup_lower_history
    )
    m.chiller_startup_upper_history_constraint = pyo.Constraint(
        m.time, m.chiller, rule=_chiller_startup_upper_history
    )


def _chiller_startup_upper_mode(m, t, c):
    if t == 0:
        return m.chiller_startup_mode[t, c] == 0
    return m.chiller_startup_mode[t, c] <= get_chiller_flow_mode(m, t, c)


def _chiller_startup_lower_history(m, t, c):
    if t == 0:
        return pyo.Constraint.Skip
    return (
        m.chiller_startup_mode[t, c]
        >= get_chiller_flow_mode(m, t, c) - m.chiller_mode[t - m.time_step, c]
    )


def _chiller_startup_upper_history(m, t, c):
    if t == 0:
        return pyo.Constraint.Skip
    return m.chiller_startup_mode[t, c] <= 1 - m.chiller_mode[t - m.time_step, c]


# Soft constraints
def _supply_water_temperature_violation(m, t):
    return m.supply_water_heating[t] - m.supply_water_temperature_violation[t] <= (
        m.maximum_supply_water_temperature
        * m.plant_volumetric_flow[t]
        * m.water_density
        * m.water_heat_capacity
    )


# Heuristic soft constraints
def _active_chiller_capacity_heuristic(m, t):
    river_chiller_output_power = m.parameters.get_river_chiller_output_power(m, t)
    return m.cooling_load_demand[
        t
    ] - river_chiller_output_power - m.cooling_load_capacity_violation[t] <= sum(
        get_chiller_flow_mode(m, t, c) * m.maximum_chiller_capacity[c]
        for c in m.chiller
    )


# Heuristic chiller mode constraints
def _only_start_chiller_once(m, c):
    return 1 >= sum(m.chiller_startup_mode[t_, c] for t_ in m.time)


def _only_change_chiller_mode_during_initial_time_steps(m, t, c):
    if t <= m.parameters.get_theta_free(m):
        return pyo.Constraint.Skip
    return m.chiller_mode[t, c] == m.chiller_mode[t - m.time_step, c]
