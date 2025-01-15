import os
from typing import List, Tuple, Type, Literal

from gpt_task import config as gpt_config
from pydantic import BaseModel
from pydantic_settings import (BaseSettings, PydanticBaseSettingsSource,
                               SettingsConfigDict, YamlConfigSettingsSource)
from sd_task import config as sd_config


class ModelsDirConfig(BaseModel):
    huggingface: str
    external: str


class DataDirConfig(BaseModel):
    models: ModelsDirConfig


class ModelConfig(BaseModel):
    id: str
    variant: str | None = "fp16"


class PreloadedModelsConfig(BaseModel):
    sd_base: List[ModelConfig] | None = None
    gpt_base: List[ModelConfig] | None = None
    controlnet: List[ModelConfig] | None = None
    vae: List[ModelConfig] | None = None


class ProxyConfig(BaseModel):
    host: str = ""
    port: int = 8080
    username: str = ""
    password: str = ""


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "FATAL", "CRITICAL"]

class LogConfig(BaseModel):
    dir: str
    level: LogLevel
    filename: str = "crynux-worker.log"


class Config(BaseSettings):
    log: LogConfig

    node_url: str
    data_dir: DataDirConfig = DataDirConfig(
        models=ModelsDirConfig(
            huggingface="models/huggingface", external="models/external"
        )
    )
    preloaded_models: PreloadedModelsConfig | None = None
    proxy: ProxyConfig | None = None

    worker_url: str = "https://dy.relay.crynux.ai/v1/worker"
    output_dir: str = "results"

    pid_file: str = "crynux_worker.pid"

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        yaml_file=os.getenv("CRYNUX_WORKER_CONFIG", "config.yml"),
        env_prefix="cw_",
        extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


_default_config: Config | None = None


def get_config() -> Config:
    global _default_config

    if _default_config is None:
        _default_config = Config()  # type: ignore

    return _default_config


def generate_sd_config(config: Config) -> sd_config.Config:
    base = None
    controlnet = None
    vae = None
    if config.preloaded_models is not None:
        if config.preloaded_models.sd_base is not None:
            base = [
                sd_config.ModelConfig(id=model.id, variant=model.variant)
                for model in config.preloaded_models.sd_base
            ]
        if config.preloaded_models.controlnet is not None:
            controlnet = [
                sd_config.ModelConfig(id=model.id, variant=model.variant)
                for model in config.preloaded_models.controlnet
            ]
        if config.preloaded_models.vae is not None:
            vae = [
                sd_config.ModelConfig(id=model.id, variant=model.variant)
                for model in config.preloaded_models.vae
            ]

    proxy = None
    if config.proxy is not None:
        proxy = sd_config.ProxyConfig(
            host=config.proxy.host,
            port=config.proxy.port,
            username=config.proxy.username,
            password=config.proxy.password,
        )

    return sd_config.Config(
        data_dir=sd_config.DataDirConfig(
            models=sd_config.ModelsDirConfig(
                huggingface=config.data_dir.models.huggingface,
                external=config.data_dir.models.external,
            )
        ),
        preloaded_models=sd_config.PreloadedModelsConfig(
            base=base,
            controlnet=controlnet,
            vae=vae,
        ),
        proxy=proxy,
    )


def generate_gpt_config(config: Config) -> gpt_config.Config:
    base = None
    if config.preloaded_models is not None:
        if config.preloaded_models.gpt_base is not None:
            base = [
                gpt_config.ModelConfig(id=model.id)
                for model in config.preloaded_models.gpt_base
            ]

    proxy = None
    if config.proxy is not None:
        proxy = gpt_config.ProxyConfig(
            host=config.proxy.host,
            port=config.proxy.port,
            username=config.proxy.username,
            password=config.proxy.password,
        )

    return gpt_config.Config(
        data_dir=gpt_config.DataDirConfig(
            models=gpt_config.ModelsDirConfig(
                huggingface=config.data_dir.models.huggingface,
            )
        ),
        preloaded_models=gpt_config.PreloadedModelsConfig(base=base),
        proxy=proxy,
    )
