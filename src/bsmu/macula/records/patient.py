from dataclasses import dataclass


@dataclass
class Patient:
    id: int
    name: str
    sex: str
    year_of_birthday: int
