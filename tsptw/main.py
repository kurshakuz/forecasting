import numpy as np
from scipy.spatial import distance
import argparse
import time
import copy
import random
import re
import math

random.seed(320)
# ------ CUSTOMER CLASS ------

class Customer:
    def __init__(self, id, point, rdy_time, due_date, serv_time):
        self.id = id
        self.point = point
        self.rdy_time = rdy_time
        self.due_date = due_date
        self.serv_time = serv_time

    def __str__(self):
        return f"cust-{self.id}"

    def __repr__(self):
        return f"cust-{self.id}"

# ------ DATA HANDLING FUNCTIONS ------

def parse_data_line(line):
    data = line.split()
    customer_id = int(data[0])
    coord = (float(data[1]), float(data[2]))
    ready_time = float(data[4])
    due_date = float(data[5])
    service_time = float(data[6])
    #print(f'Parsed data line: id={customer_id}, coord={coord}, ready_time={ready_time}, due_date={due_date}, service_time={service_time}')
    return customer_id, coord, ready_time, due_date, service_time

def find_data_start(lines):
    data_start_found = False
    for i, line in enumerate(lines):
        if 'CUST' in line and 'NO.' in line:
            data_start_found = True
        elif data_start_found and is_numeric_line(line):
            return i
    return None

def is_numeric_line(line):
    parts = re.split(r'\s+', line.strip())
    return all(part.replace('.', '', 1).isdigit() for part in parts)

def load_customers_from_file(filename):
    print(f'Loading customers from file {filename}')
    with open(filename, 'r') as file:
        lines = file.readlines()
        start_line = find_data_start(lines)

        if start_line is None:
            raise ValueError("Could not find the start of the data in the file.")

        previous_customer_id = None
        customers = []

        for line in lines[start_line:]:
            id, point, rdy_time, due_date, serv_time = parse_data_line(line)
            if id != 999:
                customers.append(Customer(id, point, rdy_time, due_date, serv_time))
            else:
                print(f'Found end of data at line {line}')
    
    return customers


# ------ ROUTE CLASS ------

class Route:
    def __init__(self, customers):
        self.customers = customers
        self.cost = 0
        self.path = []

    def __str__(self):
        return f"Route: {self.path}"
    
    def __repr__(self):
        return f"Route: {self.path}"

# ------ Helper functions ------
"""
def calculate_distance(point1, point2):
    return distance.euclidean(point1, point2)
    
    # Calculate Euclidean distance between two points
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.floor((x_diff**2 + y_diff**2)**0.5)
    
"""    
def is_feasible(path, customers, distance_matrix):
    if not path:
        return False
    current_time = 0
    for i in range(len(path) - 1):
        travel_time = distance_matrix[path[i]-1][path[i+1]-1]
        arrival_time = current_time + travel_time
        if arrival_time < customers[path[i+1]-1].rdy_time:
            current_time = customers[path[i+1]-1].rdy_time
        else:
            current_time = arrival_time
        if current_time > customers[path[i+1]-1].due_date:
            # print(f"current_time: {current_time}, arrival_time: {arrival_time}, travel_time: {travel_time}, due_date: {customers[path[i+1]-1].due_date}")
            return False
        current_time += customers[path[i]-1].serv_time
    return True

# ------ OPTIMIZER CLASS ------


def get_distance_matrix(customers):
    ids = [customer.id for customer in customers]
    locations = [customer.point for customer in customers]
    n = len(ids)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):  # We only calculate the upper half of the matrix
            dist = distance.euclidean(locations[i], locations[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Mirror the value to the other half of the matrix
    # print(dist_matrix)
    return dist_matrix

class Optimizer:
    def __init__(self, level_max, initial_path_type):
        self.level_max = level_max
        self.distance_matrix = []
        self.initial_path_type = initial_path_type
        
    def build_initial_solution(self, route):
        if self.initial_path_type == 'random':
            return self.random_solution(list(range(1,len(route.customers)+1)))
        elif self.initial_path_type == 'rdy':
            # Sort customer IDs by ready time in ascending order
            sorted_ids = sorted(range(1, len(route.customers) + 1), key=lambda idx: route.customers[idx - 1].rdy_time)
            return sorted_ids
        elif self.initial_path_type == 'due':
            # Sort customer IDs by due date in __descending__ order
            sorted_ids = sorted(range(1, len(route.customers) + 1), key=lambda idx: -route.customers[idx - 1].due_date)
            sorted_due_dates = [route.customers[idx - 1].due_date for idx in sorted_ids]
            # print(sorted_ids)
            # print(sorted_due_dates)

            return sorted_ids

    def VNS(self, route):
        solution_found = False
        while not solution_found:
            level = 1
            route.path = self.build_initial_solution(route)
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
                    print("Feasible")
                    print(f"feasible path2: {route.path} with cost {self.calculate_cost(route.path, route.customers)}")
                    route.path = route.path
                    solution_found = True
        print(f"FEASIBLE = {is_feasible(route.path, route.customers, self.distance_matrix)}")
        return route
        
    def build_feasible_solution(self, customers): # VNS - Constructive phase
        print("Building feasible solution")
        feasible_route = Route(customers)
        feasible_route = self.VNS(feasible_route)
        
        return feasible_route

    def perform_perturbation(self, level, route):
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
        optimizable_path = path.copy()
        raw_distance = self.optimization_calculate_objective(optimizable_path, customers)
            
        for i in range(1, len(optimizable_path)):
            for j in range(0, len(optimizable_path)):
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

    """
    def optimization_local_shift(self, path, customers):
        # Make a copy of current route
        current_path = path.copy()
        # Calculate the cost of the current path
        current_cost = self.optimization_calculate_objective(current_path, customers)

        # Iterate over all possible positions to shift the customers
        for i in range(len(current_path)):
            # Store the customer to be moved
            customer_to_shift = current_path[i]
            # Remove the customer from the current position
            temp_path = current_path[:i] + current_path[i+1:]
            # Insert the customer at all possible new positions
            for j in range(len(temp_path) + 1):
                new_path = temp_path[:j] + [customer_to_shift] + temp_path[j:]
                # Calculate the cost of the new path
                new_cost = self.optimization_calculate_objective(new_path, customers)
                # Check if the new cost is less than the current cost
                if new_cost < current_cost:
                    # Check if the new path is feasible
                    if is_feasible(new_path, customers, self.distance_matrix):
                        # If the new path is feasible and its cost is less, update the current path and cost
                        current_path = new_path
                        current_cost = new_cost

        return current_path
    """
    def get_customer_subsets(self, current_path, customers):
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
    
    def forward_movements(self, path, customer_id):
        """
        Forward movement shifts a given customer to a later
        position in the sequence. After the movement, the customer 
        will be visited later than in the current solution.
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
        Backward movement shifts a given customer to an earlier
        position in the sequence. After the movement, the customer 
        will be visited earlier than in the current solution.
        """
        path = path.copy()
        if customer_id not in path:
            return path

        idx = path.index(customer_id)
        if idx > 0:
            # swap positions with the previous customer in the path
            path[idx], path[idx - 1] = path[idx - 1], path[idx]
        return path
    
    def construction_local_shift(self, route):
        #print("Construction local Shift")
        # Make a copy of current route
        current_path = route.path.copy()
        # Define the current solution
        current_cost = self.construction_calculate_objective(current_path, route.customers)

        # Identify violated and non-violated customers
        violated_customers, non_violated_customers = self.get_customer_subsets(current_path, route.customers)
        # print(f"Violated customers: {[customer.id for customer in violated_customers]}")
        # print(f"Non-violated customers: {[customer.id for customer in non_violated_customers]}")

        current_path.pop(0)
            
        # Perform 1-shift movement for violated customers, moving them backward
        for customer in violated_customers:
            new_path = self.backward_movements(current_path, customer.id)
            full_new_path = [1] + new_path.copy()
            new_cost = self.construction_calculate_objective(full_new_path, route.customers)
            # print("moving violated")
            if new_cost < current_cost:
                # print("IIIIIIIIIIIIIIIIIIIIII")
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
        
        # print(len(violated_customers))

        return [1] + current_path


    def calculate_cost(self, path, customers):
        # Calculates TOTAL cost with arrival and service time (even though it is always 0 for now)
        total_cost = 0
        current_time = 0
        path_length = len(path)

        if path_length == 0:
            return math.inf

        # check feasibility
        if not is_feasible(path, customers, self.distance_matrix):
            return math.inf

        for i in range(path_length - 1):
            # Calculate travel time from current customer to next
            travel_time = self.distance_matrix[path[i]-1][path[i+1]-1]
            #print(f"{customers[path[i]-1].point} and {customers[path[i+1]-1].point} are {travel_time} apart")
            # Calculate arrival time at next customer
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the ready time, add waiting time to cost
            if arrival_time < customers[path[i+1]-1].rdy_time: 
                total_cost += customers[path[i+1]-1].rdy_time - arrival_time
                current_time = customers[path[i+1]-1].rdy_time
            else:
                current_time = arrival_time

            # Add travel time and service time to total cost
            total_cost += travel_time + customers[path[i]-1].serv_time
            current_time += customers[path[i]-1].serv_time

        return total_cost

    def optimization_calculate_objective(self, path, customers):
        # Calculate the cost of the route based only the some of the euclidean distances
        total_cost = 0
        path_length = len(path)

        # check feasibility
        if not is_feasible(path, customers, self.distance_matrix):
            return math.inf

        if path_length == 0:
            return math.inf

        for i in range(path_length - 1):
            total_cost += self.distance_matrix[path[i]-1][path[i+1]-1]

        return total_cost

    def construction_calculate_objective(self, path, customers):
        # Calculate the cost of the late "penalties" a route accumulates
        total_cost = 0
        current_time = 0
        path_length = len(path)

        if path_length == 0:
            return math.inf

        for i in range(path_length - 1):
            # Calculate travel time from current customer to next
            travel_time = self.distance_matrix[path[i]-1][path[i+1]-1]
            # print(f"{customers[path[i]-1].point} and {customers[path[i+1]-1].point} are {travel_time} apart")
            # Calculate arrival time at next customer
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the ready time, add waiting time to cost
            if arrival_time < customers[path[i+1]-1].rdy_time: 
                total_cost += customers[path[i+1]-1].rdy_time - arrival_time
                current_time = customers[path[i+1]-1].rdy_time
            else:
                current_time = arrival_time

            # The objective function used in this procedure is the sum of all positive
            # differences between the time to reach each customer and its due date, that is, ∑n i=1 max(0, βi − bi)
            # print(f"Customerid: {customers[path[i]-1].id}, Current time: {current_time}, due date: {customers[path[i+1]-1].due_date}")
            total_cost += max(0, current_time - customers[path[i+1]-1].due_date)
            current_time += customers[path[i]-1].serv_time

        return total_cost


    def construction_choose_better_path(self, path1, path2, customers):
        # Calculate the costs of both paths
        cost_path1 = self.construction_calculate_objective(path1, customers)
        cost_path2 = self.construction_calculate_objective(path2, customers)

        # Return the path with the lower cost
        if cost_path1 < cost_path2:
            return path1
        else:
            return path2

    def optimization_choose_better_path(self, path1, path2, customers):
        # Calculate the costs of both paths
        total_cost_path1 = self.optimization_calculate_objective(path1, customers)
        total_cost_path2 = self.optimization_calculate_objective(path2, customers)
        print(f"Path1({is_feasible(path1, customers, self.distance_matrix)}): {path1} with cost {total_cost_path1}")
        print(f"Path2:({is_feasible(path2, customers, self.distance_matrix)}) {path2} with cost {total_cost_path2}")

        # Return the path with the lower cost
        if total_cost_path1 < total_cost_path2:
            return path1
        else:
            return path2

    def random_solution(self, shuffle_ids):
        shuffle_ids = list(shuffle_ids) # convert the range object to a list
        shuffle_ids.pop(0) # remove the depot from the list
        random.shuffle(shuffle_ids)
        return [1] + shuffle_ids
        
    def local_2opt(self, path, customers):
        # Initialize the best_path with the current path
        print("Entering 2opt")
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

                    # If the new path is better than the best path found so far
                    # (i.e., its cost is lower), then update the best path and set improved flag to True
                    
                    # new_distance = self.optimization_calculate_objective(new_path, customers)
                    new_distance = self.calculate_cost(new_path, customers)
                    # best_distance = self.optimization_calculate_objective(best_path, customers)
                    best_distance = self.calculate_cost(best_path, customers)
                    if new_distance < best_distance:
                        #  and is_feasible(new_path, customers, self.distance_matrix):
                        best_path = new_path.copy()
                        improved = True

            # Update the current path to the best path found in this iteration
            path = best_path.copy()

        # Return the best path found
        print(f"2OPT OUT {is_feasible(path, customers, self.distance_matrix)}, {path}")
        return path


    def VND(self, route, customers): # VND (Algo 4)
        new_path = []
        while route.path != new_path:
            route.path = self.optimization_local_shift(route.path, customers)
            new_path = route.path.copy()
            route.path = self.local_2opt(route.path, customers)
        return route.path

    def GVNS(self, route): # GVNS (Algo 3)
        level = 1
        # potential_route = copy.deepcopy(route)
        # potential_route.path = self.VND(potential_route, route.customers) #see Algo 4
        # print(f"Original VND {is_feasible(potential_route.path, route.customers, self.distance_matrix)}: {potential_route.path}")
        improvement = True
        while improvement:
            prime_potential_route = copy.deepcopy(route)
            star_potential_route = copy.deepcopy(route)
            prime_potential_route = self.perform_perturbation(level, route)
            # print(f"Perturbation #{level}")
            # print(prime_potential_route.path)
            # print(f"FEASIBLE GVNS post-perturb = {is_feasible(prime_potential_route.path, prime_potential_route.customers, self.distance_matrix)}")
            star_potential_route.path = self.VND(prime_potential_route, prime_potential_route.customers) #see Algo 4
            # print(f"VND #{level}")
            # print(f"potential_route VND {is_feasible(potential_route.path, route.customers, self.distance_matrix)}: {potential_route.path}")
            # print(f"prime_potential_route path {is_feasible(prime_potential_route.path, prime_potential_route.customers, self.distance_matrix)}: {prime_potential_route.path}")
            star_potential_cost = self.calculate_cost(star_potential_route.path, star_potential_route.customers)
            # prime_potential_cost = self.calculate_cost(prime_potential_route.path, prime_potential_route.customers)
            route_cost = self.calculate_cost(route.path, route.customers)

            if star_potential_cost < route_cost:
                route = copy.deepcopy(star_potential_route)
                level = 1
            elif level < self.level_max:
                level += 1
            else:
                improvement = False
        return route

# ------ TSPTW (Traveling Salesman Problem with Time Windows) CLASS ------

class TSPTW:
    def __init__(self, iter_max, level_max, file_name, initial_path_type):
        self.iter_max = iter_max
        self.level_max = level_max
        self.raw_data_file_name = file_name
        self.customers = load_customers_from_file(file_name)
        self.best_route = Route(self.customers)
        self.optimizer = Optimizer(level_max, initial_path_type)
        self.optimizer.distance_matrix = get_distance_matrix(self.best_route.customers)

    def solve(self): # Two phase heuristic (Algo 1)
        iter_count = 0
        print('Starting to solve...')
        best_route = Route(self.customers)
        best_route.path = []

        while iter_count < self.iter_max:
            print(f'Iteration {iter_count + 1} of {self.iter_max}')
            route = self.optimizer.build_feasible_solution(self.customers) # see Algo 2
            print(f'Initial route: {route.path}')
            
            route = self.optimizer.GVNS(route) #see Algo 3
            # print(f'Route after GVNS: {route.path}')
            print(f'Route after general VNS: {is_feasible(route.path, self.customers, self.optimizer.distance_matrix)} ${self.optimizer.calculate_cost(route.path, self.customers)}, {route.path}')
            # break
            best_route = Route(route.customers[:])
            best_route.path = copy.deepcopy(self.optimizer.optimization_choose_better_path(route.path, self.best_route.path, self.customers))
            iter_count += 1

        return best_route

# ------ MAIN FUNCTION ------

def main(iter_max, level_max, file_name, initial_path_type):
    print('Starting main function...')
    tsptw = TSPTW(iter_max, level_max, file_name, initial_path_type)
    result = tsptw.solve()
    print(f'Best route found: {result.path}')
    print(f'Best route cost: {tsptw.optimizer.calculate_cost(result.path, result.customers)}')
    print(f'Best route feasible: {is_feasible(result.path, result.customers, tsptw.optimizer.distance_matrix)}')

# ------ COMMAND-LINE INTERFACE ------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_max', type=int, default=30, help='Maximum number of iterations')
    parser.add_argument('-f', '--file_name', type=str, default='n20w20.005.txt', help='File name of the input data')
    parser.add_argument('-l', '--level_max', type=int, default=8, help='Range of the local search')
    parser.add_argument('-r', '--rdy', action='store_const', const='rdy', dest='initial_path_type', help='Sets initial path type to "rdy"')
    parser.add_argument('-d', '--due', action='store_const', const='due', dest='initial_path_type', help='Sets initial path type to "due"')
    parser.set_defaults(initial_path_type='random')

    args = parser.parse_args()

    print('Parsed command line arguments:')
    print(f'iter_max={args.iter_max}, file_name={args.file_name}, level_max={args.level_max}, initial_path_type={args.initial_path_type}')

    start_time = time.time()
    main(args.iter_max, args.level_max, args.file_name, args.initial_path_type)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'The process took {elapsed_time:.3f} seconds.')
