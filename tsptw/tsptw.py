import re
import copy

from customer import Customer
from route import Route
from optimizer import Optimizer


class TSPTW:
    """Solves the Traveling Salesman Problem with Time Windows (TSPTW)."""

    def __init__(self, iter_max, level_max, file_name, initial_path_type):
        """
        Initializes a TSPTW solver.

        Args:
            iter_max: Maximum number of iterations.
            level_max: Maximum level for the local search.
            file_name: Name of the file from which to load customer data.
            initial_path_type: Type of initial path to construct.
        """
        self.iter_max = iter_max
        self.level_max = level_max
        self.raw_data_file_name = file_name
        self.customers = self.load_customers_from_file(file_name)
        self.best_route = Route(self.customers)
        self.optimizer = Optimizer(level_max, initial_path_type)
        self.optimizer.distance_matrix = self.optimizer.calculate_distance_matrix(self.best_route.customers)

    def solve(self):
        """
        Solves the TSPTW.

        Returns:
            best_route: Best route found.
        """
        iter_count = 0
        best_route = Route(self.customers)
        best_route.path = []

        while iter_count < self.iter_max:
            print(f"Iteration {iter_count + 1} of {self.iter_max}")
            iter_count += 1

            route = self.optimizer.build_feasible_solution(self.customers)
            route = self.optimizer.GVNS(route)

            best_route = Route(route.customers[:])
            best_route.path = copy.deepcopy(self.optimizer.choose_better_path(route.path, self.best_route.path, self.customers))

        return best_route

    @staticmethod
    def load_customers_from_file(filename):
        """
        Loads customers from a given file.

        Args:
            filename: Name of the file to load customers from.

        Returns:
            customers: List of customers loaded from the file.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
            start_line = TSPTW.find_data_start(lines)

            if start_line is None:
                raise ValueError("Could not find the start of the data in the file.")

            customers = []
            for line in lines[start_line:]:
                id, point, rdy_time, due_date, serv_time = TSPTW.parse_data_line(line)
                if id != 999:
                    customers.append(Customer(id, point, rdy_time, due_date, serv_time))
        return customers

    @staticmethod
    def parse_data_line(line):
        """
        Parses a line of customer data.

        Args:
            line: Line of data to parse.

        Returns:
            Parsed data as a tuple.
        """
        data = line.split()
        return int(data[0]), (float(data[1]), float(data[2])), float(data[4]), float(data[5]), float(data[6])

    @staticmethod
    def find_data_start(lines):
        """
        Finds the start of customer data in a list of lines.

        Args:
            lines: Lines to search for the data start.

        Returns:
            Line number of the data start, or None if not found.
        """
        data_start_found = False
        for i, line in enumerate(lines):
            if 'CUST' in line and 'NO.' in line:
                data_start_found = True
            elif data_start_found and TSPTW.is_numeric_line(line):
                return i
        return None

    @staticmethod
    def is_numeric_line(line):
        """
        Checks if a line consists solely of numeric characters.

        Args:
            line: Line to check.

        Returns:
            True if the line is numeric, False otherwise.
        """
        return all(part.replace('.', '', 1).isdigit() for part in re.split(r'\s+', line.strip()))
