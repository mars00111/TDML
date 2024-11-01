from pydantic import BaseModel, model_serializer
from typing import List, Dict, Any


class NodeType:
    Trainer = "Trainer"
    PS = "ParameterServer"
    MASTER = "Master"


class GPUInfo(BaseModel):
    gpu_id: int
    score: float

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        return {"gpu_id": self.gpu_id, "score": self.score}


class TrainerInfo(BaseModel):
    name: str
    root: str
    job_id: str
    gpu_info: GPUInfo


class PSInfo(BaseModel):
    name: str
    root: str
    job_id: str


class PSTrainers(BaseModel):
    name: str
    trainers: List[str]


class MasterInfo(BaseModel):
    root: str
    url: str
    port: int


class EpochInfo(BaseModel):
    epoch_n: int
    parameter_url: str


class EpochInfoNode(BaseModel):
    epoch_n: int
    client_id: str
    client_type: str
    parameter_url: str


class EpochVal(BaseModel):
    epoch_n: int
    client_id: str
    accuracy: str


class LossInfo(BaseModel):
    root: str
    epoch_n: int
    loss: float


class JobInfo(BaseModel):
    publisher: str
    id: str
    name: str
    description: str
    rewards: str
