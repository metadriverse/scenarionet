from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, Queue


class OurWandbLoggerCallback(WandbLoggerCallback):
    def __init__(self, exp_name, *args, **kwargs):
        super(OurWandbLoggerCallback, self).__init__(*args, **kwargs)
        print("===== Successfully initialize WandbCallback! ===== ")
        self.exp_name = exp_name

    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        # trial_name = str(trial) if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.experiment_dir_name if trial else None

        # remove unpickleable items!
        config = _clean_log(config)
        config = {
            key: value for key, value in config.items() if key not in self.excludes
        }

        assert trial_id is not None
        run_name = "{}_{}".format(self.exp_name, trial_id)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=run_name,
            resume=False,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            project=wandb_project,
            config=config,
        )
        wandb_init_kwargs.update(self.kwargs)

        self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)

    # def __del__(self):
    #     if self._trial_processes:
    #         for v in self._trial_processes.values():
    #             if hasattr(v, "close"):
    #                 v.close()
    #         self._trial_processes.clear()
    #         self._trial_processes = {}
