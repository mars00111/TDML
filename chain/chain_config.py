import multiprocessing

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "Blockchain Backend Service"
    app_version: str = "0.1.1"
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use.",
    )
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )
    host: str = Field(default="0.0.0.0", description="Listen address")
    public_port: int = Field(default=8001, description="Listen public port")
    private_port: int = Field(default=9001, description="Listen private port")

    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )
