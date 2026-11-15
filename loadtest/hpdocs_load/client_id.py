"""One anonymous client ID per simulated browser, as the Streamlit page makes one."""

import random
import uuid


def new_client_id(rng: random.Random) -> str:
    """A UUID4 from a seeded generator, so a run's users can be reproduced."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))
