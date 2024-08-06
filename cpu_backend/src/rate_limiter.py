import logging
import time

logger = logging.getLogger(__name__)


def rate_limit(rate, name):
    """Rate limit a function to a certain number of calls per second."""
    count = 0
    start_time = time.monotonic()

    while True:
        count += 1
        elapsed_time = time.monotonic() - start_time
        if elapsed_time >= 1:
            if count > rate:
                logger.warning(
                    "%s rate limit exceeded: %s requests per second", name, count
                )
            count = 0
            start_time = time.monotonic()
        time.sleep(0.01)
