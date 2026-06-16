"""Shared wandb initialisation helper."""

import logging

log = logging.getLogger(__name__)


def init_wandb(cfg: dict, job_type: str, run_name: str = None, extra_config: dict = None):
    """
    Initialise a wandb run if wandb.enabled is True in cfg.

    Returns the wandb run object, or a _NoOpRun stub when disabled.
    """
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        log.info("wandb disabled — skipping init.")
        return _NoOpRun()

    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed — skipping. pip install wandb to enable.")
        return _NoOpRun()

    full_cfg = {**cfg}
    if extra_config:
        full_cfg.update(extra_config)

    run = wandb.init(
        project=wcfg.get("project", "laplace-head-sam"),
        entity=wcfg.get("entity") or None,
        group=wcfg.get("group") or None,
        job_type=job_type,
        name=run_name,
        config=full_cfg,
        reinit=True,
    )
    log.info("wandb run started: %s  project=%s", run.name, wcfg.get("project"))
    return run


class _NoOpRun:
    """Stub that silently absorbs all wandb calls when wandb is disabled."""

    def log(self, *a, **kw):
        pass

    def summary(self):
        return {}

    def __setattr__(self, name, value):
        pass  # absorb summary assignments like run.summary["x"] = y

    def finish(self):
        pass

    def watch(self, *a, **kw):
        pass
