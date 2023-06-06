from typing import List
from customer import Customer


class Route:
    """Represents a route in the Traveling Salesman Problem with Time Windows (TSPTW)."""

    def __init__(self, customers: List[Customer]):
        """
        Initializes a route.

        Args:
            customers: A list of Customer objects to visit.
        """
        self.customers = customers
        self.cost = 0  # Cost of the route.
        self.path = []  # List to hold the path of the route.

    def __str__(self) -> str:
        """Returns a simple string representation of the route."""
        return f"Route: {self.path}"

    def __repr__(self) -> str:
        """Returns a more detailed string representation of the route."""
        return f"Route: {self.path}"
