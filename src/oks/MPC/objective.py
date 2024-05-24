from pyomo import environ as pyo

from src.oks.MPC.models import get_chiller_flow_mode


def set_optimization_objective(
    m,
):
    # Objective function coefficients
    h_per_time_step = m.time_step / 60
    m.supply_temperature_violation_cost_multiplier = pyo.Param(default=1000000)
    m.chiller_startup_cost_multiplier = pyo.Param(default=100000)
    m.chiller_energy_cost_multiplier = pyo.Param(default=h_per_time_step)
    m.capacity_violation_cost_multiplier = pyo.Param(default=400)
    m.supply_water_temperature_below_setpoint_cost_multiplier = pyo.Param(default=50000)

    m.objective = pyo.Objective(rule=_objective)


def set_optimization_constraints(
    m,
):
    m.initial_chiller_mode_constraint = pyo.Constraint(
        m.chiller, rule=_initial_chiller_mode
    )
    m.supply_water_temperature_violation_constraint = pyo.Constraint(
        m.time, rule=_supply_water_temperature_violation
    )
    m.active_chiller_capacity_heuristic_constraint = pyo.Constraint(
        m.time, rule=_active_chiller_capacity_heuristic
    )

    m.only_start_chiller_once_constraint = pyo.Constraint(
        m.chiller, rule=_only_start_chiller_once
    )

    m.chiller_mode_change_block_constraint = pyo.Constraint(
        m.time,
        m.chiller,
        rule=_only_change_chiller_mode_during_initial_time_steps,
    )


def _objective(m):
    supply_water_violation_cost = m.supply_temperature_violation_cost_multiplier * sum(
        m.supply_water_temperature_violation[t] for t in m.time
    )

    chiller_startup_cost = m.chiller_startup_cost_multiplier * sum(
        m.chiller_startup_mode[t, c] for c in m.chiller for t in m.time
    )

    energy_cost = m.chiller_energy_cost_multiplier * sum(
        m.chiller_input_power[t, c] * m.chiller_energy_cost[c]
        for c in m.chiller
        for t in m.time
    )

    cooling_load_capacity_violation_cost = m.capacity_violation_cost_multiplier * sum(
        m.cooling_load_capacity_violation[t] for t in m.time
    )

    supply_water_temperature_below_setpoint_cost = (
        m.supply_water_temperature_below_setpoint_cost_multiplier
        * sum(m.supply_water_temperature_below_setpoint[t] for t in m.time)
    )

    return (
        supply_water_violation_cost
        + energy_cost
        + cooling_load_capacity_violation_cost
        + chiller_startup_cost
        + supply_water_temperature_below_setpoint_cost
    )


def _supply_water_temperature_violation(m, t):
    return m.supply_water_heating[t] - m.supply_water_temperature_violation[t] <= (
        m.maximum_supply_water_temperature
        * m.plant_volumetric_flow[t]
        * m.water_density
        * m.water_heat_capacity
    )


def _active_chiller_capacity_heuristic(m, t):
    river_chiller_output_power = m.parameters.get_river_chiller_output_power(m, t)
    return m.cooling_load_demand[
        t
    ] - river_chiller_output_power - m.cooling_load_capacity_violation[t] <= sum(
        get_chiller_flow_mode(m, t, c) * m.maximum_chiller_capacity[c]
        for c in m.chiller
    )


def _only_start_chiller_once(m, c):
    return 1 >= sum(m.chiller_startup_mode[t_, c] for t_ in m.time)


def _only_change_chiller_mode_during_initial_time_steps(m, t, c):
    if t <= m.parameters.get_theta_free(m):
        return pyo.Constraint.Skip
    return m.chiller_mode[t, c] == m.chiller_mode[t - m.time_step, c]


def _initial_chiller_mode(m, c):
    return m.chiller_mode[0, c] == m.initial_chiller_flow_mode[c]
