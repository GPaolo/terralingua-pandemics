"""Action templates for OpenGridWorld."""

from core.environment.artifact import ARTIFACT_TYPE

ACTION_TEXT = {
    "move": {
        "description": "Move of one cell in the specified direction, or stay in the current position",
        "params": {"direction": "One among [right, left, up, down, stay]."},
    },
    "give": {
        "description": "Transfer some of your energy to a being on a cell "
        "adjacent to yours. This is physical contact.",
        "params": {
            "target": "Name of a being on a cell next to yours to give energy to.",
            "amount": "Integer amount of energy to transfer (1 up to your current energy).",
        },
    },
    "bury": {
        "description": "Bury remains lying on the ground next to you, removing "
        "them from the world. Handling remains is risky: you may catch the sickness.",
        "params": {
            "name": "Name of the remains artifact to bury (must be on your cell or an adjacent one).",
        },
    },
    "take": {
        "description": "Steal energy from a being on a cell adjacent to yours. "
        "This is physical contact.",
        "params": {
            "target": "Name of a being on a cell next to yours to steal energy from.",
            "amount": "Integer amount of energy to steal (1 up to target's current energy).",
        },
    },
    "reproduce": {
        "description": "Asexually generate an offspring. It costs {reproduction_cost} energy.",
        "params": {
            "energy": "Integer amount of **additional** energy the parent gifts the child (0 up to <parent_current_energy - {reproduction_cost}>)",
            "name": "Name of the offspring (use **unique** names)",
            "offspring_genome": "A single short sentence (max 15 words) describing the personality of your offspring. Leave empty to pass on your own personality unchanged.",
        },
    },
    "create_artifact": {
        "description": "Creates a new artifact at the being's location.",
        "params": {
            "name": "The name of the artifact (use **unique** names)",
            "type": f"Type of the artifact to create. One among: {list(ARTIFACT_TYPE.keys())}",
            "payload": f"Content of the artifact (e.g. a message, a code snippet, etc.). It depends on the artifact type: {ARTIFACT_TYPE}",
            "lifespan": "How many time steps the artifact will last (in number of steps, integer > 0. If -1 the artifact will never disappear)",
        },
    },
    "pickup_artifact": {
        "description": "Picks up the artifact and puts it in the being's inventory",
        "params": {"name": "Name of the artifact to pick up"},
    },
    "drop_artifact": {
        "description": "Drops artifact from the inventory at the being's current position",
        "params": {"name": "Name of the artifact to drop"},
    },
    "give_artifact": {
        "description": "Gives an artifact from the inventory to a being on a "
        "cell adjacent to yours. This is physical contact.",
        "params": {
            "artifact_name": "The name of the artifact",
            "target_agent": "Name of a being on a cell next to yours to give the artifact to.",
        },
    },
    "set_color": {
        "description": "Change how you appear to other beings by choosing your color",
        "params": {"color": "The color you want to show to other beings"},
    },
}
