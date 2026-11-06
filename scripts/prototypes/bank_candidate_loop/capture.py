"""Reuse the solve tracer while recording actual persistent-cache decisions."""

import logging

from scripts.prototypes.bank_booking_determinism.measure import CacheEvents
from scripts.prototypes.bank_wall_events.capture import main

if __name__ == "__main__":
    logger = logging.getLogger("jax._src.compiler")
    logger.addHandler(CacheEvents())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    main()
