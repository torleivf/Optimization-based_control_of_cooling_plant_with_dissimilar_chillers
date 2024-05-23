from src.oks.MPC.model_parameters import get_default_chillers
import enum


class ChillerAction(enum.Enum):
    TURN_OFF = 0
    TURN_ON = 1


class ControllerAction:
    def __init__(self, chiller_number: int, action: ChillerAction):
        self.chiller_number = chiller_number
        self.action = action

    def __str__(self):
        return f"Chiller {self.chiller_number}: {self.action}"


def get_initial_active_chillers(model, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    all_chiller_modes = model.chiller_mode.extract_values()
    initial_active_chillers = []
    initial_timestep = 0
    for chiller in chillers:
        if all_chiller_modes[(initial_timestep, chiller.number)]:
            initial_active_chillers.append(chiller.number)
    return initial_active_chillers


def get_recommended_active_chillers(model, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    all_chiller_modes = model.chiller_mode.extract_values()
    recommended_active_chillers = []
    first_optimization_timestep = list(model.time)[1]
    for chiller in chillers:
        if all_chiller_modes[(first_optimization_timestep, chiller.number)]:
            recommended_active_chillers.append(chiller.number)
    return recommended_active_chillers


def get_controller_actions(model, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    initial_active_chillers = get_initial_active_chillers(model, chillers)
    recommended_active_chillers = get_recommended_active_chillers(model, chillers)

    recommended_actions = []
    for chiller in chillers:
        if (
            chiller.number in recommended_active_chillers
            and chiller.number not in initial_active_chillers
        ):
            recommended_actions.append(
                ControllerAction(chiller.number, ChillerAction.TURN_ON)
            )
        if (
            chiller.number not in recommended_active_chillers
            and chiller.number in initial_active_chillers
        ):
            recommended_actions.append(
                ControllerAction(chiller.number, ChillerAction.TURN_OFF)
            )
    return recommended_actions


def get_chiller_mode_description(model, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    initial_active_chillers = get_initial_active_chillers(model, chillers)

    chiller_mode_description = ""
    if len(initial_active_chillers) == 0:
        chiller_mode_description += "No chillers are currently active. "
    elif len(initial_active_chillers) == 1:
        chiller_mode_description += (
            f"Chiller {initial_active_chillers[0]} is the only chiller "
            f"that is currently active. "
        )
    else:
        chiller_mode_description += (
            f"Chillers {', and '.join(str(c) for c in initial_active_chillers)} "
            f"are currently active. "
        )

    controller_actions = get_controller_actions(model, chillers)
    if len(controller_actions) == 0:
        chiller_mode_description += (
            "No chiller actions are recommended by the optimization result."
        )
        return chiller_mode_description

    for action in controller_actions:
        if action.action == ChillerAction.TURN_ON:
            chiller_mode_description += (
                f"\nThe optimization result indicates that chiller {action.chiller_number} "
                f"should be turned on now."
            )
        if action.action == ChillerAction.TURN_OFF:
            chiller_mode_description += (
                f"\nThe optimization result indicates that chiller {action.chiller_number} "
                f"should be turned off now."
            )

    return chiller_mode_description


def get_active_chiller_numbers(model, chillers=None):
    if chillers is None:
        chillers = get_default_chillers()
    all_chiller_modes = model.chiller_mode.extract_values()
    active_chillers = []
    for chiller in chillers:
        if any([all_chiller_modes[(tau, chiller.number)] for tau in model.time]):
            active_chillers.append(chiller.number)
    return active_chillers
