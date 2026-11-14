from typing import Literal

from pydantic import BaseModel

ServiceState = Literal["up", "down"]

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, ServiceState]