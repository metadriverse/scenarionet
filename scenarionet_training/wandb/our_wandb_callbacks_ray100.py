from multiprocessing import Queue

from ray.tune.integration.wandb import WandbLogger, _clean_log, _set_api_key


class OurWandbLogger(WandbLogger):
    def __init__(self, config, logdir, trial):
        self.exp_name = config["logger_config"]["wandb"].pop("exp_name")
        super(OurWandbLogger, self).__init__(config, logdir, trial)

    def _init(self):

        config = self.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        try:
            if config.get("logger_config", {}).get("wandb"):
                logger_config = config.pop("logger_config")
                wandb_config = logger_config.get("wandb").copy()
            else:
                wandb_config = config.pop("wandb").copy()
        except KeyError:
            raise ValueError(
                "Wandb logger specified but no configuration has been passed. "
                "Make sure to include a `wandb` key in your `config` dict "
                "containing at least a `project` specification.")

        _set_api_key(wandb_config)

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        additional_excludes = wandb_config.pop("excludes", [])
        exclude_results += additional_excludes

        # Log config keys on each result?
        log_config = wandb_config.pop("log_config", False)
        if not log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = self.trial.trial_id if self.trial else None
        trial_name = str(self.trial) if self.trial else None

        # Project name for Wandb
        try:
            wandb_project = wandb_config.pop("project")
        except KeyError:
            raise ValueError(
                "You need to specify a `project` in your wandb `config` dict.")

        # Grouping
        wandb_group = wandb_config.pop(
            "group", self.trial.trainable_name if self.trial else None)

        # remove unpickleable items!
        config = _clean_log(config)

        assert trial_id is not None
        run_name = "{}_{}".format(self.exp_name, trial_id)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=run_name,
            resume=True,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            project=wandb_project,
            config=config)
        wandb_init_kwargs.update(wandb_config)

        self._queue = Queue()
        self._wandb = self._logger_process_cls(
            queue=self._queue,
            exclude=exclude_results,
            to_config=self._config_results,
            **wandb_init_kwargs)
        self._wandb.start()
