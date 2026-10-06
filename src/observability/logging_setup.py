"""
Process-wide logging: every log line, from our code and from libraries, is one structured record.

Existing `logging.getLogger(__name__)` calls keep working: structlog's ProcessorFormatter renders
them like its own. A request's id is bound to a context variable by the request middleware, so
every line written while serving it, including from worker threads started with asyncio.to_thread,
carries it without any call site passing it along. Output is JSON when stderr is not a terminal
(a hosting platform's log collector) and readable text on a terminal; LOG_FORMAT forces either.
"""
import logging
import sys

import structlog

LOG_FORMATS = ("auto", "json", "console")


def _shared_processors():
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]


def configure_logging(log_format: str = "auto", level: int = logging.INFO, stream=None) -> None:
    """Idempotent: calling it again replaces the previous configuration instead of stacking handlers."""
    stream = stream or sys.stderr
    use_json = log_format == "json" or (log_format == "auto" and not stream.isatty())
    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(colors=stream.isatty())

    structlog.configure(
        processors=[*_shared_processors(), structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers and prints an access line per request; the request
    # middleware writes one structured access line instead, so its logger is silenced.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers[:] = []
    access.propagate = False
