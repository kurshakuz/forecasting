import numpy as np
from scipy.spatial import distance
from route import Route
import random
import copy
import math

# ------ HELPER FUNCTION ------

def is_feasible(path, customers, distance_matrix):
    """
    Check the feasibility of a given path.

    Parameters:
    path (list): The list of customer ids forming a path.
    customers (list): The list of Customer objects.
    distance_matrix (2D numpy array): The matrix storing the distance between each pair of customers.

    Returns:
    bool: True if the path is feasible (i.e., all time constraints are met), False otherwise.
    """
    if not path:
        return False
    current_time = 0
    eval_path = path.copy()
    eval_path.append(1)
    for i in range(len(eval_path) - 1):
        travel_time = distance_matrix[eval_path[i]-1][eval_path[i+1]-1]
        arrival_time = current_time + travel_time
        if arrival_time < customers[eval_path[i+1]-1].rdy_time:
            current_time = customers[eval_path[i+1]-1].rdy_time
        else:
            current_time = arrival_time
        if current_time > customers[eval_path[i+1]-1].due_date:
            return False
        current_time += customers[eval_path[i]-1].serv_time
    return True
    
# ------ OPTIMIZER CLASS ------

class Optimizer:
    def __init__(self, level_max, initial_path_type):
        self.level_max = level_max
        self.distance_matrix = []
        self.initial_path_type = initial_path_type

    # ------ INITIALIZATION ------        
    
    def random_solution(self, shuffle_ids):
        """
        Generate a random solution by shuffling customer ids.

        Parameters:
        shuffle_ids (list): List of customer ids.

        Returns:
        list: List of shuffled customer ids with depot at the start.
        """
        shuffle_ids = list(shuffle_ids)
        shuffle_ids.pop(0) # customer 1 is always the 1st
        random.shuffle(shuffle_ids)
        return [1] + shuffle_ids
        
    def build_initial_solution(self, route):
        """
        Construct the initial solution based on the defined 'initial_path_type'.

        Parameters:
        route (Route): Current route object containing customers information.

        Returns:
        list: Initial solution as a list of customer ids.
        """
        if self.initial_path_type == 'random':
            return self.random_solution(list(range(1,len(route.customers)+1)))
        elif self.initial_path_type == 'rdy':
            # Sort customer IDs by ready time in ascending order
            sorted_ids = sorted(range(1, len(route.customers) + 1), key=lambda idx: route.customers[idx - 1].rdy_time)
            return sorted_ids
        elif self.initial_path_type == 'due':
            # Sort customer IDs by due date in __descending__ order
            sorted_ids = sorted(range(1, len(route.customers) + 1), key=lambda idx: -route.customers[idx - 1].due_date)
            return sorted_ids

    def VNS(self, route):
        """
        Perform the Variable Neighborhood Search (VNS) method to find feasible solutions.

        Parameters:
        route (Route): Current route object containing customers information.

        Returns:
        Route: Route object with an improved feasible path.
        """
        solution_found = False
        MAX_ITER = 1000
        while not solution_found:
            iter = 0
            level = 1
            route.path = self.build_initial_solution(route)
            if is_feasible(route.path, route.customers, self.distance_matrix):
                route.path = route.path
                solution_found = True
            while not is_feasible(route.path, route.customers, self.distance_matrix) and level < self.level_max:
                potential_route = self.perform_perturbation(level, route)
                potential_route.path = self.construction_local_shift(potential_route)
                best_path = self.construction_choose_better_path(potential_route.path, route.path, route.customers)
                route.path = best_path.copy()

                if self.construction_calculate_objective(route.path, route.customers) == self.construction_calculate_objective(potential_route.path, potential_route.customers):
                    level = 1
                else:
                    level += 1

                if is_feasible(route.path, route.customers, self.distance_matrix):
                    route.path = route.path
                    solution_found = True
                iter += 1
                if iter > MAX_ITER:
                    print("Solution not found")
                    solution_found = True
                    break
        return route

    def build_feasible_solution(self, customers): # VNS - Constructive phase
        """
        Construct a feasible solution using VNS.

        Parameters:
        customers (list): List of Customer objects.

        Returns:
        Route: Route object with a feasible path.
        """
        feasible_route = Route(customers)
        feasible_route = self.VNS(feasible_route)

        return feasible_route
        
    def calculate_distance_matrix(self, customers):
        """
        Calculate the distance matrix among all customers including the depot.

        Parameters:
        customers (list): List of Customer objects.

        Returns:
        numpy.ndarray: Distance matrix of size n x n, where n is the number of customers.
        """
        ids = [customer.id for customer in customers]
        locations = [customer.point for customer in customers]
        n = len(ids)
        dist_matrix = np.zeros((n, n))
    
        for i in range(n):
            for j in range(i+1, n):  # We only calculate the upper half of the matrix
                dist = distance.euclidean(locations[i], locations[j])
                dist = math.floor(dist)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Mirror the value to the other half of the matrix
        return dist_matrix 

    # ------ SOLUTION GENERATION AND IMPROVEMENT ------
    
    def perform_perturbation(self, level, route):
        """
        Perturb the given route by moving a random customer to a new position.

        Parameters:
        level (int): The number of perturbations to be performed.
        route (Route): Current route object.

        Returns:
        Route: New route object after performing perturbations.
        """
        new_path = route.path.copy()  # Copy the path to avoid modifying the original
        customers = route.customers.copy()
        new_path.pop(0)
        for _ in range(level):
            # Select a random customer to shift
            customer_to_shift = random.choice(new_path)
            # Remove the selected customer from its current position
            new_path.remove(customer_to_shift)
            # Choose a random position to insert the selected customer
            insert_pos = random.randint(0, len(new_path) + 1)
            # Insert the customer at the new position
            new_path.insert(insert_pos, customer_to_shift)
        # Create a new Route with the perturbed path
        new_route = Route(customers)
        new_route.path = [1] + new_path
        return new_route        
       
    def optimization_local_shift(self, path, customers):
        """
        Optimizes a given route path by performing local shift operations.

        Parameters:
        path (list): The current path.
        customers (list): List of Customer objects.

        Returns:
        list: The optimized path.
        """
        optimizable_path = path.copy()
        raw_distance = self.optimization_calculate_objective(optimizable_path, customers)

        for i in range(1, len(optimizable_path)):
            for j in range(1, len(optimizable_path)):
                if i != j:
                    # Generate a neighbor solution by performing a 1-shift
                    neighbor_path = optimizable_path.copy()
                    neighbor_path.insert(j, neighbor_path.pop(i))
                    
                    # Evaluate the neighbor solution
                    neighbor_distance = self.optimization_calculate_objective(neighbor_path, customers)

                    # If the neighbor solution is feasible and better than the current solution, update the current solution
                    if neighbor_distance < raw_distance and is_feasible(neighbor_path, customers, self.distance_matrix):
                        raw_distance = neighbor_distance
                        optimizable_path = neighbor_path

        # Return the best list found
        return optimizable_path
    
    def forward_movements(self, path, customer_id):
        """
        Perform a forward shift of a given customer in the path.

        Parameters:
        path (list): The current path.
        customer_id (int): The ID of the customer to be moved.

        Returns:
        list: Updated path after the forward shift.
        """
        path = path.copy()
        if customer_id not in path:
            return path

        idx = path.index(customer_id)
        if idx < len(path) - 1:
            # swap positions with the next customer in the path
            path[idx], path[idx + 1] = path[idx + 1], path[idx]
        return path

    def backward_movements(self, path, customer_id):
        """
        Perform a backward shift of a given customer in the path.

        Parameters:
        path (list): The current path.
        customer_id (int): The ID of the customer to be moved.

        Returns:
        list: Updated path after the backward shift.
        """
        path = path.copy()
        if customer_id not in path:
            return path

        idx = path.index(customer_id)
        if idx > 0:
            # swap positions with the previous customer in the path
            path[idx], path[idx - 1] = path[idx - 1], path[idx]
        return path
    
    def get_customer_subsets(self, current_path, customers):
        """
        Categorize customers in the current path into violated and non-violated sets based on time constraints.

        Parameters:
        current_path (list): The current path.
        customers (list): List of Customer objects.

        Returns:
        tuple: Two lists of customers - violated and non-violated.
        """
        current_time = 0
        violated_customers = []
        non_violated_customers = []

        for i in range(len(current_path) - 1):
            # Calculate travel time from current customer to next
            travel_time = self.distance_matrix[current_path[i]-1][current_path[i+1]-1]
            # Calculate arrival time at next customer
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the ready time, wait until ready time
            if arrival_time < customers[current_path[i]-1].rdy_time:
                current_time = customers[current_path[i]-1].rdy_time
            else:
                current_time = arrival_time

            # If arrival time is later than the due date, the route is not feasible
            if current_time > customers[current_path[i+1]-1].due_date:
                violated_customers.append(copy.deepcopy(customers[current_path[i+1]-1]))
            else:
                non_violated_customers.append(copy.deepcopy(customers[current_path[i+1]-1]))

            # Add service time to current time
            current_time += customers[current_path[i]-1].serv_time

        return violated_customers, non_violated_customers
        
    def construction_local_shift(self, route):
        """
        Perform local shift operations to construct a feasible solution.

        Parameters:
        route (Route): Current route object.

        Returns:
        list: The path of a feasible solution.
        """
        # Make a copy of current route
        current_path = route.path.copy()
        # Define the current solution
        current_cost = self.construction_calculate_objective(current_path, route.customers)

        # Identify violated and non-violated customers
        violated_customers, non_violated_customers = self.get_customer_subsets(current_path, route.customers)

        current_path.pop(0)
            
        # Perform 1-shift movement for violated customers, moving them backward
        for customer in violated_customers:
            new_path = self.backward_movements(current_path, customer.id)
            full_new_path = [1] + new_path.copy()
            new_cost = self.construction_calculate_objective(full_new_path, route.customers)
            if new_cost < current_cost:
                current_path = new_path.copy()
                current_cost = new_cost

        # Identify violated and non-violated customers
        violated_customers, non_violated_customers = self.get_customer_subsets(current_path, route.customers)
        
        # Perform 1-shift movement for non-violated customers, moving them forward
        for customer in non_violated_customers:
            new_path = self.forward_movements(current_path, customer.id)
            full_new_path = [1] + new_path.copy()
            new_cost = self.construction_calculate_objective(full_new_path, route.customers)
            if new_cost < current_cost:
                current_path = new_path.copy()
                current_cost = new_cost
        
        # Identify violated and non-violated customers
        violated_customers, non_violated_customers = self.get_customer_subsets(current_path, route.customers)
        
        # Perform 1-shift movement for non-violated customers, moving them backward
        for customer in non_violated_customers:
            new_path = self.backward_movements(current_path, customer.id)
            full_new_path = [1] + new_path.copy()
            new_cost = self.construction_calculate_objective(full_new_path, route.customers)
            if new_cost < current_cost:
                current_path = new_path.copy()
                current_cost = new_cost
        
        # Identify violated and non-violated customers
        violated_customers, non_violated_customers = self.get_customer_subsets(current_path, route.customers)
        
        # Perform 1-shift movement for violated customers, moving them forward
        for customer in violated_customers:
            new_path = self.forward_movements(current_path, customer.id)
            full_new_path = [1] + new_path.copy()
            new_cost = self.construction_calculate_objective(full_new_path, route.customers)
            if new_cost < current_cost:
                current_path = new_path.copy()
                current_cost = new_cost
                
        return [1] + current_path
    
    def local_2opt(self, path, customers):
        """
        Perform 2-opt local search to improve the current solution.

        Parameters:
        path (list): The current path.
        customers (list): List of Customer objects.

        Returns:
        list: Optimized path after performing 2-opt operations.
        """
        # Initialize the best_path with the current path
        best_path = path.copy()
        # Flag to keep track if the path was improved in the last iteration
        improved = True

        # Continue the process as long as the path is improved
        while improved:
            # Initially, we assume no improvement is made
            improved = False

            # For each customer in the path (except the last two and the first one)
            for i in range(1, len(path) - 2):
                # For each subsequent customer in the path
                for j in range(i + 1, len(path)):
                    # Skip if the two customers are adjacent
                    if j - i == 1: 
                        continue

                    # Make a copy of the current path
                    new_path = path.copy()

                    # Perform 2-opt swap: reverse the section of the path between customer i and j
                    new_path[i:j] = path[j - 1:i - 1:-1]
                    
                    new_distance = self.optimization_calculate_objective(new_path, customers)
                    best_distance = self.optimization_calculate_objective(best_path, customers)
                    if new_distance < best_distance:
                        best_path = new_path.copy()
                        improved = True

            # Update the current path to the best path found in this iteration
            path = best_path.copy()

        # Return the best path found
        return path
    
    def VND(self, route, customers): # VND (Algo 4)
        """
        Perform Variable Neighborhood Descent (VND) to find the best local optimal solution.

        Parameters:
        route (Route): Current route object.
        customers (list): List of Customer objects.

        Returns:
        list: Optimized path after performing VND.
        """
        new_path = []
        while route.path != new_path:
            route.path = self.optimization_local_shift(route.path, customers)
            new_path = route.path.copy()
            route.path = self.local_2opt(route.path, customers)
        return route.path
        
    # ------ COST AND OBJECTIVE FUNCTION CALCULATION ------    
    
    def calculate_cost(self, path, customers):
        """
        Calculate the total cost of a path, taking into account travel, service and waiting times.
        Returns infinity if the path is not feasible.

        Parameters:
        path (list): The current path.
        customers (list): List of Customer objects.

        Returns:
        float: Total cost of the path.
        """
        # Calculates TOTAL cost with arrival and service time (even though it is always 0 for now)
        total_cost = 0
        current_time = 0
        path_length = len(path)

        if path_length == 0:
            return math.inf

        if not is_feasible(path, customers, self.distance_matrix):
            return math.inf

        eval_path = path.copy()
        eval_path.append(1)

        for i in range(path_length):
            # Calculate travel time from current customer to next
            travel_time = self.distance_matrix[eval_path[i]-1][eval_path[i+1]-1]
            # Calculate arrival time at next customer
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the ready time, add waiting time to cost
            if arrival_time < customers[eval_path[i+1]-1].rdy_time: 
                total_cost += customers[eval_path[i+1]-1].rdy_time - arrival_time
                current_time = customers[eval_path[i+1]-1].rdy_time
            else:
                current_time = arrival_time

            # Add travel time and service time to total cost
            total_cost += travel_time + customers[eval_path[i]-1].serv_time
            current_time += customers[eval_path[i]-1].serv_time

        return total_cost
    
    def optimization_calculate_objective(self, path, customers):
        """
        Calculate the total cost of a path based on the sum of the Euclidean distances.
        Returns infinity if the path is not feasible.

        Parameters:
        path (list): The current path.
        customers (list): List of Customer objects.

        Returns:
        float: Total cost of the path.
        """
        # Calculate the cost of the route based only the some of the euclidean distances
        total_cost = 0
        path_length = len(path)

        if not is_feasible(path, customers, self.distance_matrix):
            return math.inf

        if path_length == 0:
            return math.inf

        eval_path = path.copy()
        eval_path.append(1)

        for i in range(path_length):
            total_cost += self.distance_matrix[eval_path[i]-1][eval_path[i+1]-1]

        return total_cost

    def construction_calculate_objective(self, path, customers):
        """
        Calculate the total cost of a path, taking into account travel, service and late penalties.
        Returns infinity if the path is empty.

        Parameters:
        path (list): The current path.
        customers (list): List of Customer objects.

        Returns:
        float: Total cost of the path.
        """
        # Calculate the cost of the late "penalties" a route accumulates
        total_cost = 0
        current_time = 0
        path_length = len(path)

        if path_length == 0:
            return math.inf

        if not is_feasible(path, customers, self.distance_matrix):
            return math.inf

        eval_path = path.copy()
        eval_path.append(1)

        for i in range(path_length):
            # Calculate travel time from current customer to next
            travel_time = self.distance_matrix[eval_path[i]-1][eval_path[i+1]-1]
            
            # Calculate arrival time at next customer
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the ready time, add waiting time to cost
            if arrival_time < customers[eval_path[i+1]-1].rdy_time: 
                total_cost += customers[eval_path[i+1]-1].rdy_time - arrival_time
                current_time = customers[eval_path[i+1]-1].rdy_time
            else:
                current_time = arrival_time

            # The objective function used in this procedure is the sum of all positive
            # differences between the time to reach each customer and its due date, that is, ∑n i=1 max(0, βi − bi)
            total_cost += max(0, current_time - customers[eval_path[i+1]-1].due_date)
            current_time += customers[eval_path[i]-1].serv_time

        return total_cost    
        
    # ------ PATH SELECTION ------ 
    
    def construction_choose_better_path(self, path1, path2, customers):
        """
        Compare two paths and return the one with the lower cost based on a construction objective function.

        Parameters:
        path1, path2 (list): Paths to compare.
        customers (list): List of Customer objects.

        Returns:
        list: Path with the lower cost.
        """

        # Calculate the costs of both paths
        cost_path1 = self.construction_calculate_objective(path1, customers)
        cost_path2 = self.construction_calculate_objective(path2, customers)

        # Return the path with the lower cost
        if cost_path1 < cost_path2:
            return path1
        else:
            return path2

    def optimization_choose_better_path(self, path1, path2, customers):
        """
        Compare two paths and return the one with the lower cost based on an optimization objective function.

        Parameters:
        path1, path2 (list): Paths to compare.
        customers (list): List of Customer objects.

        Returns:
        list: Path with the lower cost.
        """
        # Calculate the costs of both paths
        total_cost_path1 = self.optimization_calculate_objective(path1, customers)
        total_cost_path2 = self.optimization_calculate_objective(path2, customers)

        # Return the path with the lower cost
        if total_cost_path1 < total_cost_path2:
            return path1
        else:
            return path2
            
    def choose_better_path(self, path1, path2, customers):
        """
        This function chooses the better of two paths based on their total cost.
        
        Parameters:
        path1 (list): The first path.
        path2 (list): The second path.
        customers (list): The customers to be served.
        
        Returns:
        list: The path with the lower total cost.
        """
        # Calculate the costs of both paths
        cost_path1 = self.calculate_cost(path1, customers)
        cost_path2 = self.calculate_cost(path2, customers)
        
        # Return the path with the lower cost
        return path1 if cost_path1 < cost_path2 else path2
            
    # ------ MAIN OPTIMIZATION ALGORITHM ------        
           
    def GVNS(self, route): # GVNS (Algo 3)
        """
        Perform the General Variable Neighborhood Search (GVNS) optimization algorithm.

        Parameters:
        route (Route object): The initial route.

        Returns:
        Route object: The optimized route.
        """
        level = 1
        improvement = True

        while improvement:
            prime_potential_route = copy.deepcopy(route)
            star_potential_route = copy.deepcopy(route)
            prime_potential_route = self.perform_perturbation(level, route)
            star_potential_route.path = self.VND(prime_potential_route, prime_potential_route.customers) #see Algo 4
            star_potential_cost = self.calculate_cost(star_potential_route.path, star_potential_route.customers)
            route_cost = self.calculate_cost(route.path, route.customers)

            if star_potential_cost < route_cost:
                route = copy.deepcopy(star_potential_route)
                level = 1
            elif level < self.level_max:
                level += 1
            else:
                improvement = False
        return route
