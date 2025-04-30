from typing import Optional

from pydantic import BaseModel, EmailStr, Field

## field to apply the constraints


class Student(BaseModel):
    name: str = "sanket"  ## default value is Sanket
    age: Optional[int] = None  ## default value of age is None
    ## pydantic can convert string to int format when the input expected is int

    email: EmailStr
    cgpa: float = Field(
        gt=0, lt=10, default=9.37, description="Representing the cgpa of the student"
    )


new_student = {"name": "Sanket", "age": "11", "email": "abc@gmail.com", "cgpa": "5"}
student = Student(**new_student)
print(student)

## convertint to json
print(student.model_dump_json())

## converting to dict
print(dict(student))
