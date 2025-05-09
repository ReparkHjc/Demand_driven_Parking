import os
import sys
from gymnasium.envs.registration import register

__version__ = "1.0.0"

try:
    from farama_notifications import notifications

    if "avp_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["avp_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_avp_envs():
    """Import the envs module so that envs register themselves."""

    register(
        id="avp-v0",
        entry_point="avp_env.envs.avp_env:AutonomousParkingEnv",
    )

    register(
        id="avp-v0",
        entry_point="avp_env.envs.avp_env:MetricsEnv",
    )


_register_avp_envs()
