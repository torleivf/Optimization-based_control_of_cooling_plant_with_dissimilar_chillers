import datetime
import enum
import itertools
import math
import pandas as pd

from src.oks.OSS.data import get_optimization_data


class EnergySource(enum.Enum):
    HEAT = 1
    ELECTRICITY = 2


class Chiller:
    def __init__(
        self,
        chiller_type,
        number,
        maximum_capacity,  # MW
        max_volumetric_flow,  # m3/h
        theta_output,  # minutes
        theta_startup,  # minutes
        energy_source: EnergySource,
        average_coefficient_of_performance,  # MW/MW
        tau_output,
        tau_input,
        colour,
        model_line_style,
        effect_apis_id,
        input_power_apis_id,
        output_temperature_apis_id,
        input_temperature_apis_id,
        input_power_prediction_minutes=40,
    ):
        self.type = chiller_type
        self.number = number
        self.maximum_capacity = maximum_capacity
        self.max_volumetric_flow = max_volumetric_flow
        self.energy_source = energy_source
        # Electricity is in kW while heat is in MW in the data
        self.input_power_coefficient = (
            0.001 if energy_source == EnergySource.ELECTRICITY else 1
        )
        self.theta_output = theta_output
        self.tau_output = tau_output
        self.average_coefficient_of_performance = average_coefficient_of_performance
        self.theta_startup = theta_startup
        self.input_power_prediction_minutes = input_power_prediction_minutes
        self.maximum_useful_history_length = max(
            theta_output,
            theta_startup,
            input_power_prediction_minutes + theta_startup,
        )
        self.max_input_power = maximum_capacity / average_coefficient_of_performance
        self.input_power_high_prediction = self.max_input_power * 0.6
        self.input_power_low_prediction = (
            self.max_input_power * 0.4
            if energy_source == EnergySource.ELECTRICITY
            else 0
        )
        self.tau_input = tau_input

        # plot and data retrieval settings
        self.colour = colour
        self.model_line_style = model_line_style
        self.effect_apis_id = effect_apis_id
        self.input_power_apis_id = input_power_apis_id
        self.output_temperature_apis_id = output_temperature_apis_id
        self.input_temperature_apis_id = input_temperature_apis_id

        # initial values and history
        self.historic_values_initialized = False
        self.initial_chiller_flow_mode = 0
        self.initial_chiller_input_power = 0
        self.initial_chiller_output_power = 0
        self.initial_chiller_output_water_temperature = 0
        self.initial_chiller_input_water_temperature = 0
        self.historic_chiller_flow_modes = {}
        self.historic_input_powers = {}
        self.historic_input_power_average = 0

    def set_historic_values(
        self,
        dataframe,
        time_horizon_start,
        parameters=None,
    ):
        if parameters is None:
            parameters = Parameters()
        self.initial_chiller_input_power = (
            dataframe[self.input_power_apis_id][time_horizon_start]
            * self.input_power_coefficient
        )
        if math.isnan(self.initial_chiller_input_power):
            self.initial_chiller_input_power = 0
        self.initial_chiller_output_power = dataframe[self.effect_apis_id][
            time_horizon_start
        ]
        if math.isnan(self.initial_chiller_output_power):
            self.initial_chiller_output_power = 0
        self.initial_chiller_output_water_temperature = dataframe[
            self.output_temperature_apis_id
        ][time_horizon_start]
        if math.isnan(self.initial_chiller_output_water_temperature):
            self.initial_chiller_output_water_temperature = 0
        self.initial_chiller_input_water_temperature = dataframe[
            self.input_temperature_apis_id
        ][time_horizon_start]
        if math.isnan(self.initial_chiller_input_water_temperature):
            self.initial_chiller_input_water_temperature = 0
        self.initial_chiller_flow_mode = self.compute_chiller_flow_mode(
            input_power=self.initial_chiller_input_power,
        )
        input_power_prediction_start_minute = (
            self.theta_startup - self.input_power_prediction_minutes
        )
        active_input_power_sum = 0
        active_input_time_step_count = 0
        maximum_active_input_power_above_setpoint = None
        minimum_active_input_power_below_setpoint = None
        for time in range(-self.maximum_useful_history_length, 0):
            timestamp = time_horizon_start + pd.Timedelta(minutes=time)
            historic_input_power = (
                dataframe[self.input_power_apis_id][timestamp]
                * self.input_power_coefficient
            )
            historic_supply_water_temperature = dataframe["CT12"][timestamp]
            historic_flow_mode = self.compute_chiller_flow_mode(
                input_power=historic_input_power,
            )
            self.historic_input_powers[time] = historic_input_power
            self.historic_chiller_flow_modes[time] = historic_flow_mode
            if time >= input_power_prediction_start_minute:
                if (
                    self.historic_chiller_flow_modes[time - self.theta_startup]
                    and historic_flow_mode
                ):
                    active_input_power_sum += historic_input_power
                    active_input_time_step_count += 1
                    if (
                        historic_supply_water_temperature
                        > parameters.supply_water_temperature_setpoint
                    ):
                        if maximum_active_input_power_above_setpoint is None:
                            maximum_active_input_power_above_setpoint = (
                                historic_input_power
                            )
                        else:
                            maximum_active_input_power_above_setpoint = max(
                                maximum_active_input_power_above_setpoint,
                                historic_input_power,
                            )
                    elif self.energy_source == EnergySource.ELECTRICITY:
                        if minimum_active_input_power_below_setpoint is None:
                            minimum_active_input_power_below_setpoint = (
                                historic_input_power
                            )
                        else:
                            minimum_active_input_power_below_setpoint = min(
                                minimum_active_input_power_below_setpoint,
                                historic_input_power,
                            )

        # allow some inactive moments
        if active_input_time_step_count >= self.input_power_prediction_minutes - 4:
            self.historic_input_power_average = (
                active_input_power_sum / active_input_time_step_count
            )
        if minimum_active_input_power_below_setpoint is not None:
            self.input_power_low_prediction = minimum_active_input_power_below_setpoint
        if maximum_active_input_power_above_setpoint is not None:
            self.input_power_high_prediction = maximum_active_input_power_above_setpoint
            # ensure that low prediction is lower than or equal to high prediction
            if self.input_power_low_prediction >= self.input_power_high_prediction:
                self.input_power_low_prediction = self.input_power_high_prediction
        self.historic_values_initialized = True

    def compute_chiller_flow_mode(
        self,
        input_power=0,
    ) -> int:
        if input_power > self.maximum_capacity / 100:
            return 1
        return 0


def get_default_chillers() -> list[Chiller]:
    default_chillers = [
        Chiller(
            chiller_type="absorption",
            number=1,
            maximum_capacity=3.3,
            max_volumetric_flow=330,
            theta_output=8,
            theta_startup=2,
            energy_source=EnergySource.HEAT,
            average_coefficient_of_performance=0.6,
            tau_output=8,
            tau_input=2,
            colour="olive",
            model_line_style=(0, (4, 3)),
            effect_apis_id="Q41",
            input_power_apis_id="Q42",
            output_temperature_apis_id="CT411",
            input_temperature_apis_id="CT412",
        ),
        Chiller(
            chiller_type="centrifugal",
            number=2,
            maximum_capacity=2.8,
            max_volumetric_flow=317,
            theta_output=2,
            theta_startup=2,
            energy_source=EnergySource.ELECTRICITY,
            average_coefficient_of_performance=5.7,
            tau_output=0.222,
            tau_input=2,
            colour="purple",
            model_line_style=(0, (4, 4)),
            effect_apis_id="Q51",
            input_power_apis_id="CE51",
            output_temperature_apis_id="CT511",
            input_temperature_apis_id="CT512",
        ),
        Chiller(
            chiller_type="screw",
            number=3,
            maximum_capacity=1.3,
            max_volumetric_flow=140,
            theta_output=0,
            theta_startup=2,
            energy_source=EnergySource.ELECTRICITY,
            average_coefficient_of_performance=4.5,
            tau_output=0.857,
            tau_input=2,
            colour="brown",
            model_line_style=(0, (4, 5)),
            effect_apis_id="Q61",
            input_power_apis_id="CE61",
            output_temperature_apis_id="CT611",
            input_temperature_apis_id="CT612",
        ),
        Chiller(
            chiller_type="absorption",
            number=4,
            maximum_capacity=3.2,
            max_volumetric_flow=329,
            theta_output=8,
            theta_startup=2,
            energy_source=EnergySource.HEAT,
            average_coefficient_of_performance=0.6,
            tau_output=8,
            tau_input=2,
            colour="green",
            model_line_style=(0, (4, 6)),
            effect_apis_id="Q71",
            input_power_apis_id="Q43",
            output_temperature_apis_id="CT711",
            input_temperature_apis_id="CT712",
        ),
    ]
    return default_chillers


class ModelParameters:
    def __init__(
        self,
        fixed_variables,
        time_horizon_start_date=datetime.datetime.utcnow(),
        time_horizon=60,
        time_step=2,
        theta_free=5,
        maximum_supply_water_temperature=10,
        parameters=None,
        chillers=None,
    ):
        self.time_horizon_start_date = time_horizon_start_date
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.theta_free = theta_free
        self.maximum_supply_water_temperature = maximum_supply_water_temperature

        if parameters is None:
            parameters = Parameters()
        self.parameters = parameters

        self.fixed_variables = fixed_variables

        if chillers is None:
            chillers = get_default_chillers()
        self.chillers = chillers

    def get_chiller_count(self):
        return len(self.chillers)

    def get_time_horizon_start_date(self, _model):
        return self.time_horizon_start_date

    def get_time_horizon(self, _model):
        return self.time_horizon

    def get_time_step(self, _model):
        return self.time_step

    def get_theta_free(self, _model):
        return self.theta_free

    def get_maximum_supply_water_temperature(self, _model):
        return self.maximum_supply_water_temperature

    def get_maximum_chiller_volumetric_flow(self, _model, chiller):
        return self.chillers[chiller - 1].max_volumetric_flow

    def get_chiller_energy_cost(self, _model, chiller):
        if self.chillers[chiller - 1].energy_source == EnergySource.HEAT:
            return self.parameters.heat_cost
        elif self.chillers[chiller - 1].energy_source == EnergySource.ELECTRICITY:
            return self.parameters.electricity_cost
        else:
            raise ValueError(
                f"Unknown energy source {self.chillers[chiller - 1].energy_source}"
            )

    def get_maximum_chiller_capacity(self, _model, chiller):
        return self.chillers[chiller - 1].maximum_capacity

    def get_input_output_effect_lag_timesteps(self, _model, chiller):
        return int(self.chillers[chiller - 1].theta_output / self.time_step)

    def get_average_coefficient_of_performance(self, _model, chiller):
        return self.chillers[chiller - 1].average_coefficient_of_performance

    def get_chiller_in_out_discrete_time_constant_coefficient(self, _model, chiller):
        return _model.time_step / (
            _model.time_step + self.chillers[chiller - 1].tau_output
        )

    def get_initial_chiller_mode(self, _model, chiller):
        return self.chillers[chiller - 1].initial_chiller_flow_mode

    def get_initial_chiller_input_power(self, _model, chiller):
        return self.chillers[chiller - 1].initial_chiller_input_power

    def get_initial_chiller_output_power(self, _model, chiller):
        return self.chillers[chiller - 1].initial_chiller_output_power

    def get_initial_chiller_output_water_temperature(self, _model, chiller):
        return self.chillers[chiller - 1].initial_chiller_output_water_temperature

    def get_initial_chiller_input_water_temperature(self, _model, chiller):
        return self.chillers[chiller - 1].initial_chiller_input_water_temperature

    def get_water_density(self, _model):
        return self.parameters.water_density

    def get_water_heat_capacity(self, _model):
        return self.parameters.water_heat_capacity

    def get_cooling_load_demand(self, _model, time):
        return self.fixed_variables.cooling_load_demand[time]

    def get_supply_water_volumetric_flow(self, _model, time):
        return self.fixed_variables.supply_water_volumetric_flow[time]

    def get_active_chillers_volumetric_flow_approximation(self, _model, time, chiller):

        active_initial_flow = 0
        for c in self.chillers:
            active_initial_flow += c.initial_chiller_flow_mode * c.max_volumetric_flow
        if (
            active_initial_flow
            > self.fixed_variables.supply_water_volumetric_flow[time]
        ):
            return active_initial_flow

        if not self.fixed_variables.active_flow_approximation_is_computed:
            self.fixed_variables.compute_active_chillers_volumetric_flow_approximation(
                _model, self.chillers
            )
        return self.fixed_variables.get_active_chillers_volumetric_flow_approximation(
            time, chiller
        )

    def get_initial_supply_water_temperature(self, _model):
        return self.fixed_variables.initial_supply_water_temperature

    def get_startup_flow_timesteps(self, _model, chiller):
        return int(self.chillers[chiller - 1].theta_startup / self.time_step)

    def get_historic_chiller_flow_mode(self, _model, time, chiller):
        if not self.chillers[chiller - 1].historic_values_initialized:
            raise RuntimeError(f"Chiller {chiller} historic values are not initialized")
        return self.chillers[chiller - 1].historic_chiller_flow_modes[time]

    def get_historic_chiller_input_power(self, _model, time, chiller):
        if not self.chillers[chiller - 1].historic_values_initialized:
            raise RuntimeError(f"Chiller {chiller} historic values are not initialized")
        return self.chillers[chiller - 1].historic_input_powers[time]

    def get_supply_water_setpoint_temperature(self, _model, time):
        return self.parameters.supply_water_temperature_setpoint

    def get_chiller_input_power_high_prediction(self, _model, time, chiller):
        return self.chillers[chiller - 1].input_power_high_prediction

    def get_chiller_input_power_low_prediction(self, _model, time, chiller):
        return self.chillers[chiller - 1].input_power_low_prediction

    def get_chiller_input_power_evolution_discrete_time_constant_coefficient(
        self, _model, time, chiller
    ):
        return _model.time_step / (
            _model.time_step + self.chillers[chiller - 1].tau_input
        )

    def get_river_chiller_output_power(self, _model, time):
        return self.fixed_variables.river_water_chiller_output_power[time]

    def get_return_water_temperature(self, _model, time):
        return self.fixed_variables.return_water_temperature[time]

    def get_river_chilled_water_temperature(self, _model, time):
        m = _model
        t = time
        return_heating = (
            self.get_return_water_temperature(m, t)
            * m.supply_water_volumetric_flow[t]
            * m.water_density
            * m.water_heat_capacity
        )
        river_water_chilling = self.get_river_chiller_output_power(m, t)
        river_chilled_water_temperature = (return_heating - river_water_chilling) / (
            m.supply_water_volumetric_flow[t] * m.water_density * m.water_heat_capacity
        )
        if m.supply_water_volumetric_flow[t] == 0:
            river_chilled_water_temperature = 0
        return river_chilled_water_temperature

    def get_supply_water_temperature(self, _model, time):
        try:
            return _model.supply_water_heating[time].value / (
                _model.plant_volumetric_flow[time].value
                * _model.water_density
                * _model.water_heat_capacity
            )
        except TypeError:
            return None

    def get_cooling_load(self, _model, time):
        try:
            return (
                (
                    self.get_return_water_temperature(_model, time)
                    - self.get_supply_water_temperature(_model, time)
                )
                * _model.supply_water_volumetric_flow[time]
                * _model.water_density
                * _model.water_heat_capacity
            )
        except TypeError:
            return None


class Parameters:
    def __init__(
        self,
        water_density=999.7,  # kg/m3 at 10 C, 1 atm
        water_heat_capacity=0.000001164,  # MWh/(kg*K) at 10 C, isochoric
        freezing_point=0,  # C
        boiling_point=100,  # C
        heat_cost=10,  # NOK/MWh
        electricity_cost=1000,  # NOK/MWh
        supply_water_temperature_setpoint=6,  # C
    ):
        self.water_density = water_density
        self.water_heat_capacity = water_heat_capacity
        self.freezing_point = freezing_point
        self.boiling_point = boiling_point
        self.heat_cost = heat_cost
        self.electricity_cost = electricity_cost
        self.supply_water_temperature_setpoint = supply_water_temperature_setpoint


def get_sorted_active_flow_levels(chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    active_flow_levels = []
    binary_combinations = list(itertools.product([0, 1], repeat=len(chillers)))
    for combination in binary_combinations:
        active_flow_level = 0
        for chiller_index, chiller in enumerate(chillers):
            active_flow_level += (
                combination[chiller_index] * chiller.max_volumetric_flow
            )
        active_flow_levels.append(active_flow_level)
    return sorted(active_flow_levels)


def get_minimum_active_chillers_volumetric_flow_for_backflow(
    supply_water_volumetric_flow,
    sorted_active_flow_levels=None,
):
    if sorted_active_flow_levels is None:
        sorted_active_flow_levels = get_sorted_active_flow_levels()
    for active_flow_level in sorted_active_flow_levels:
        if active_flow_level >= supply_water_volumetric_flow:
            return active_flow_level
    return sorted_active_flow_levels[-1]


def get_sorted_active_chiller_flow_levels(chillers, current_chiller_number=None):
    own_chiller_flow = 0
    other_chillers = chillers
    if current_chiller_number is not None:
        own_chiller_flow = chillers[current_chiller_number - 1].max_volumetric_flow
        other_chillers = (
            chillers[: current_chiller_number - 1] + chillers[current_chiller_number:]
        )
    active_flow_levels = []
    binary_combinations = list(itertools.product([0, 1], repeat=len(other_chillers)))
    for combination in binary_combinations:
        active_flow_level = own_chiller_flow
        for chiller_index, chiller in enumerate(other_chillers):
            active_flow_level += (
                combination[chiller_index] * chiller.max_volumetric_flow
            )
        active_flow_levels.append(active_flow_level)
    sorted_active_flow_levels = sorted(active_flow_levels)
    if sorted_active_flow_levels[0] == 0:
        sorted_active_flow_levels = sorted_active_flow_levels[1:]
    return sorted_active_flow_levels


class FixedVariables:
    def __init__(
        self,
        time_horizon_start,
        time_horizon,
        time_step,
        dataframe,
        supply_water_volumetric_flow_is_measured=False,
        return_water_temperature_is_measured=False,
    ):
        initial_supply_water_temperature = dataframe["CT12"][time_horizon_start]
        initial_cooling_load_demand = dataframe["Q11"][time_horizon_start]

        self.cooling_load_demand = [
            initial_cooling_load_demand for _ in range(0, time_horizon + time_step)
        ]

        if supply_water_volumetric_flow_is_measured:
            self.supply_water_volumetric_flow = [
                dataframe["CF11"][time_horizon_start + pd.Timedelta(minutes=time)]
                for time in range(0, time_horizon + 1)
            ]
        else:
            initial_supply_water_volumetric_flow = dataframe["CF11"][time_horizon_start]
            self.supply_water_volumetric_flow = [
                initial_supply_water_volumetric_flow for _ in range(0, time_horizon + 1)
            ]
        self.active_flow_approximation_is_computed = False
        self.active_flow_approximations = {}
        self.sorted_active_flow_levels = get_sorted_active_flow_levels()
        self.active_chillers_volumetric_flow_approximation = [
            get_minimum_active_chillers_volumetric_flow_for_backflow(
                supply_water_volumetric_flow,
                self.sorted_active_flow_levels,
            )
            for supply_water_volumetric_flow in self.supply_water_volumetric_flow
        ]
        self.initial_supply_water_temperature = initial_supply_water_temperature

        if return_water_temperature_is_measured:
            self.return_water_temperature = [
                dataframe["CT11"][time_horizon_start + pd.Timedelta(minutes=time)]
                for time in range(0, time_horizon + 1)
            ]
        else:
            initial_return_water_temperature = dataframe["CT11"][time_horizon_start]
            self.return_water_temperature = [
                initial_return_water_temperature for _ in range(0, time_horizon + 1)
            ]

        initial_river_water_chiller_output_power = dataframe["Q12"][time_horizon_start]
        self.river_water_chiller_output_power = [
            initial_river_water_chiller_output_power for _ in range(0, time_horizon + 1)
        ]

    def compute_active_chillers_volumetric_flow_approximation(
        self,
        model,
        chillers,
    ):
        self.active_flow_approximations = {time: {} for time in model.time}
        for chiller in chillers:
            sorted_active_flow_levels = get_sorted_active_chiller_flow_levels(chillers)
            for time in model.time:
                self.active_flow_approximations[time][chiller.number] = (
                    get_minimum_active_chillers_volumetric_flow_for_backflow(
                        model.supply_water_volumetric_flow[time],
                        sorted_active_flow_levels,
                    )
                )
        self.active_flow_approximation_is_computed = True

    def get_active_chillers_volumetric_flow_approximation(self, time, chiller):
        if not self.active_flow_approximation_is_computed:
            raise RuntimeError("Active flow approximation is not computed")
        return self.active_flow_approximations[time][chiller]


def get_model_parameters(
    optimization_data=None,
    time_step=2,  # minutes
    time_horizon=60,  # minutes
    theta_free=5,  # minutes
    heat_cost=10,  # NOK/MWh
    electricity_cost=1000,  # NOK/MWh
) -> ModelParameters:
    if optimization_data is None:
        optimization_data = get_optimization_data()
    dataframe = optimization_data
    start_time = dataframe.last_valid_index()
    fixed_variables = FixedVariables(
        time_horizon_start=start_time,
        time_horizon=time_horizon,
        time_step=time_step,
        dataframe=dataframe,
    )
    parameters = Parameters(
        heat_cost=heat_cost,
        electricity_cost=electricity_cost,
    )

    initialized_chillers = []
    for chiller in get_default_chillers():
        initialized_chiller = chiller
        initialized_chiller.set_historic_values(
            dataframe=dataframe,
            time_horizon_start=start_time,
        )
        initialized_chillers.append(initialized_chiller)

    model_params = ModelParameters(
        time_horizon_start_date=start_time,
        time_horizon=time_horizon,
        time_step=time_step,
        theta_free=theta_free,
        fixed_variables=fixed_variables,
        parameters=parameters,
        chillers=initialized_chillers,
    )

    return model_params
