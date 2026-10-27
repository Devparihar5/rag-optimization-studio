from .models import PipelineConfig


class ExperimentService:
    def __init__(self) -> None:
        self._experiments = []

    def create_experiment(self, name: str, config: PipelineConfig) -> dict:
        payload = {
            "name": name,
            "config": config.model_dump(),
            "status": "queued",
        }
        self._experiments.append(payload)
        return payload

    def list_experiments(self) -> list[dict]:
        return self._experiments
