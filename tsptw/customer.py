from typing import Tuple


class Customer:
    """Represents a customer with id, location point, ready time, due date, and service time."""

    def __init__(self, id: int, point: Tuple[float, float], rdy_time: float, due_date: float, serv_time: float):
        """
        Initializes a customer.

        Args:
            id: Unique identifier of the customer.
            point: Coordinate location of the customer as a tuple (x, y).
            rdy_time: The earliest time the customer is ready for service.
            due_date: The latest time the customer can be served.
            serv_time: The time it takes to serve the customer.
        """
        self.id = id
        self.point = point
        self.rdy_time = rdy_time
        self.due_date = due_date
        self.serv_time = serv_time

    def __str__(self):
        """Returns a simple string representation of the customer."""
        return f"{(self.id, self.point, self.rdy_time, self.due_date, self.serv_time)}"

    def __repr__(self):
        """Returns a more detailed string representation of the customer."""
        return f"{(self.id, self.point, self.rdy_time, self.due_date, self.serv_time)}"
