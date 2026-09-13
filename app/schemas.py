from pydantic import BaseModel


class MachineInput(BaseModel):
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float