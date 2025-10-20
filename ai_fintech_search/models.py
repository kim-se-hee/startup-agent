from typing import List, Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages

class Company(BaseModel):
    name: str
    description: str = Field(..., alias="desription")  
    domain: str

    model_config = ConfigDict(
        populate_by_name=True, 
        str_strip_whitespace=True,
    )

class Product(BaseModel):
    name: str
    description: str
    strengths: List[str]
    limitations: List[str]

class CompanyDetail(BaseModel):
    company: Company
    products: List[Product]


class StartupHit(BaseModel):
    name: str
    domain: str
    description: str

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

class StartupSearchResult(BaseModel):
    items: List[StartupHit] = []


class SearchState(BaseModel):
    messages: Annotated[list, add_messages] = []
    region: str = "Global"
    limit: int = 12
    language: str = "ko"
    results: List[StartupHit] = []
    
    model_config = ConfigDict(arbitrary_types_allowed=True)