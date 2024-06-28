import os
from typing import Any, Dict, List, Tuple, Type, Literal

import yaml
from gpt_task import config as gpt_config
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import (BaseSettings, PydanticBaseSettingsSource,
                               SettingsConfigDict)
from sd_task import config as sd_config


class YamlSettingsConfigDict(SettingsConfigDict):
    yaml_file: str | None


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a YAML file

    Note: slightly adapted version of JsonConfigSettingsSource from docs.
    """

    _yaml_data: Dict[str, Any] | None = None

    # def __init__(self, settings_cls: type[BaseSettings]):
    #     super().__init__(settings_cls)

    @property
    def yaml_data(self) -> Dict[str, Any]:
        if self._yaml_data is None:
            yaml_file = self.config.get("yaml_file")
            if yaml_file is not None and os.path.exists(yaml_file):
                with open(yaml_file, mode="r", encoding="utf-8") as f:
                    self._yaml_data = yaml.safe_load(f)
            else:
                self._yaml_data = {}
        return self._yaml_data  # type: ignore

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        field_value = self.yaml_data.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d


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

    model_config = YamlSettingsConfigDict(
        env_nested_delimiter="__",
        yaml_file=os.getenv("CRYNUX_WORKER_CONFIG", "config.yml"),
        env_file=".env",
        env_prefix="cw_",
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
