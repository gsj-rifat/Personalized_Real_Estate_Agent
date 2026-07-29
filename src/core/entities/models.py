from pydantic import BaseModel, Field


class BuyerPreferences(BaseModel):
    house_size: str = Field(description="Desired house size and layout")
    priorities: str = Field(description="Top priorities for the property")
    amenities: str = Field(description="Desired amenities")
    transportation: str = Field(description="Transportation preferences")
    neighborhood_type: str = Field(description="Urban vs suburban preference")

    def to_query(self) -> str:
        return (
            f"House size: {self.house_size}. "
            f"Priorities: {self.priorities}. "
            f"Amenities: {self.amenities}. "
            f"Transportation: {self.transportation}. "
            f"Neighborhood: {self.neighborhood_type}."
        )


class HomeListing(BaseModel):
    neighborhood: str
    location: str
    bedrooms: int
    bathrooms: float
    house_size_sqft: int
    price_k_usd: float


class RecommendationResult(BaseModel):
    answer: str
    query: str
