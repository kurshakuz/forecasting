import numpy as np
import argparse
import copy
import random
import re
import math

# random.seed(320)
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
    print(f'Parsed data line: id={customer_id}, coord={coord}, ready_time={ready_time}, due_date={due_date}, service_time={service_time}')
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

def calculate_distance(point1, point2):
    # Calculate Euclidean distance between two points
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.floor((x_diff**2 + y_diff**2)**0.5)

def is_feasible(route):
    if not route or not route.path:
        return False
    current_time = 0
    for i in range(len(route.path) - 1):
        travel_time = calculate_distance(route.customers[route.path[i]-1].point, route.customers[route.path[i+1]-1].point)
        arrival_time = current_time + travel_time
        if arrival_time < route.customers[route.path[i+1]-1].rdy_time:
            current_time = route.customers[route.path[i+1]-1].rdy_time
        else:
            current_time = arrival_time
        if current_time > route.customers[route.path[i+1]-1].due_date:
            # print(f"current_time: {current_time}, arrival_time: {arrival_time}, travel_time: {travel_time}, due_date: {route.customers[route.path[i+1]-1].due_date}")
            return False
        current_time += route.customers[route.path[i]-1].serv_time
    return True

# ------ OPTIMIZER CLASS ------

class Optimizer:
    def __init__(self, level_max):
        self.level_max = level_max

    def build_feasible_solution(self, route): # VNS - Constructive phase
        solution_found = False
        feasible_route = Route(route.customers[:])
        while not solution_found:
            level = 1
            print("RANDOM")
            route.path = self.random_solution(list(range(1,len(route.customers)+1)))
            print("looking for feasible solution")
            while not is_feasible(route) and level < self.level_max:
                potential_route = self.perform_perturbation(level, route.path, route.customers)
                potential_route.path = self.construction_local_shift(potential_route)
                best_path = self.construction_choose_better_path(potential_route.path, route.path, route.customers)
                route.path = copy.deepcopy(best_path)

                if self.calculate_objective(route.path, route.customers) == self.calculate_objective(potential_route.path, potential_route.customers):
                    level = 1
                else:
                    level += 1

                if is_feasible(route):
                    print("Feasible")
                    print(f"feasible path2: {route.path} with cost {self.calculate_cost(route.path, route.customers)}")
                    feasible_route.path = route.path
                    solution_found = True
        return feasible_route

    def perform_perturbation(self, level, path, customers):
        new_path = path.copy()  # Copy the path to avoid modifying the original
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
        new_route = Route(customers[:])
        new_route.path = [1] + new_path
        return new_route

    def optimization_local_shift(self, route):
        # Make a copy of current route
        current_path = route.path[:]
        # Calculate the cost of the current path
        current_cost = self.calculate_cost(current_path, route.customers)

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
                new_cost = self.calculate_cost(new_path, route.customers)
                # Check if the new cost is less than the current cost
                if new_cost < current_cost:
                    # Check if the new path is feasible
                    if is_feasible(new_path):
                        # If the new path is feasible and its cost is less, update the current path and cost
                        current_path = new_path
                        current_cost = new_cost

        return current_path
    
    def get_customer_subsets(self, current_path, customers):
        current_time = 0
        violated_customers = []
        non_violated_customers = []

        for i in range(len(current_path) - 1):
            # Calculate travel time from current customer to next
            travel_time = calculate_distance(customers[current_path[i]-1].point, customers[current_path[i+1]-1].point)
            # Calculate arrival time at next customer
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the ready time, wait until ready time
            if arrival_time < customers[current_path[i]-1].rdy_time:
                current_time = customers[current_path[i]-1].rdy_time
            else:
                current_time = arrival_time

            # If arrival time is later than the due date, the route is not feasible
            if current_time > customers[current_path[i+1]-1].due_date:
                violated_customers.append(customers[current_path[i+1]-1])
            else:
                non_violated_customers.append(customers[current_path[i+1]-1])

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
        current_cost = self.calculate_objective(current_path, route.customers)

        # Identify violated and non-violated customers
        violated_customers, non_violated_customers = self.get_customer_subsets(current_path, route.customers)
        # print(f"Violated customers: {[customer.id for customer in violated_customers]}")
        # print(f"Non-violated customers: {[customer.id for customer in non_violated_customers]}")

        # Define the 1-shift movements
        movements = [
            self.backward_movements,
            self.forward_movements,
            self.backward_movements,
            self.forward_movements,
        ]
        # Define the customer subsets for each movement
        customer_subsets = [
            violated_customers,
            non_violated_customers,
            non_violated_customers,
            violated_customers,
        ]
        
        current_path.pop(0)
        # Perform each 1-shift movement
        for movement, customer_subset in zip(movements, customer_subsets):
            # Perform the 1-shift movement for each customer
            for customer in customer_subset:
                # Get the new path
                new_path = movement(current_path, customer.id)
                # Calculate the new solution
                full_new_path = [1] + new_path.copy()
                new_cost = self.calculate_objective(full_new_path, route.customers)
                new_route = Route(route.customers[:])
                new_route.path = full_new_path
                # If the new solution is better, update the current path and solution
                if new_cost < current_cost:
                    current_path = new_path.copy()
                    current_cost = new_cost
        return [1] + current_path

    def calculate_cost(self, path, customers):
        total_cost = 0
        current_time = 0
        path_length = len(path)

        for i in range(path_length - 1):
            # Calculate travel time from current customer to next
            travel_time = calculate_distance(customers[path[i]-1].point, customers[path[i+1]-1].point)
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

    def calculate_objective(self, path, customers):
        total_cost = 0
        current_time = 0
        path_length = len(path)

        for i in range(path_length - 1):
            # Calculate travel time from current customer to next
            travel_time = calculate_distance(customers[path[i]-1].point, customers[path[i+1]-1].point)
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
        cost_path1 = self.calculate_objective(path1, customers)
        cost_path2 = self.calculate_objective(path2, customers)

        # Return the path with the lower cost
        if cost_path1 < cost_path2:
            return path1
        else:
            return path2

    def random_solution(self, shuffle_ids):
        shuffle_ids = list(shuffle_ids) # convert the range object to a list
        shuffle_ids.pop(0) # remove the depot from the list
        random.shuffle(shuffle_ids)
        return [1] + shuffle_ids
        
    def local_2opt(self, path, customers):
        best_path = path[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_path) - 2):
                for j in range(i + 1, len(best_path)):
                    if j - i == 1: continue  # changes nothing, skip then
                    new_path = path[:]
                    new_path[i:j] = path[j - 1:i - 1:-1]  # this is the 2-optSwap
                    if self.calculate_cost(new_path, customers) < self.calculate_cost(best_path, customers):
                        best_path = new_path
                        improved = True
            path = best_path
        return path
    
    def VND(self, path, customers): # VND (Algo 4)
        path = path[:]
        new_path = None
        while path != new_path:
            path = self.perform_local_shift(path)
            new_path = path[:]
            path = self.local_2opt(path, customers)
        return path
        
    def GVNS(self, route): # GVNS (Algo 3)
        level = 1
        path = copy.deepcopy(self.VND(route.path, route.customers)) #see Algo 4
        while level <= self.level_max:
            print(f"VNS level {level}/{self.level_max}")
            
            self.perform_perturbation(level, path, route.customers)
            print(route.path)
            print(potential_route.path)
            potential_route.path = VND(potential_route.path, route.customers) #see Algo 4
            route.path = self.choose_better_path(route.path, potential_route.path, route.customers)
            
            if route.path == potential_route.path:
                level = 1
            else:
                level += 1
        return route
                    
# ------ TSPTW (Traveling Salesman Problem with Time Windows) CLASS ------

class TSPTW:
    def __init__(self, iter_max, level_max, file_name):
        self.iter_max = iter_max
        self.level_max = level_max
        self.raw_data_file_name = file_name
        self.customers = load_customers_from_file(file_name)
        self.best_route = Route(self.customers)
        self.optimizer = Optimizer(level_max)

    def solve(self): # Two phase heuristic (Algo 1)
        iter_count = 0
        print('Starting to solve...')
        route = Route(self.customers)
        self.optimizer.build_feasible_solution(route)
        # while iter_count < self.iter_max:
        #     print(f'Iteration {iter_count + 1} of {self.iter_max}')
        #     self.optimizer.build_feasible_solution(route) # see Algo 2
        #     break
        #     print(f'Initial route: {route.path}')
        #     route = optimizer.GVNS(route) #see Algo 3
        #     print(f'Route after general VNS: {route.path}')
        #     self.best_route = self.choose_better_path(route.path, self.best_route.path)
        #     print(f'Best route so far: {self.best_route.path}')
        #     iter_count += 1

    

# ------ MAIN FUNCTION ------

def main(iter_max, level_max, file_name):
    print('Starting main function...')
    tsptw = TSPTW(iter_max, level_max, file_name)
    tsptw.solve()

# ------ COMMAND-LINE INTERFACE ------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_max', type=int, default=30, help='Maximum number of iterations')
    parser.add_argument('-f', '--file_name', type=str, default='raw.txt', help='File name of the input data')
    parser.add_argument('-l', '--level_max', type=int, default=8, help='Range of the local search')
    args = parser.parse_args()

    print('Parsed command line arguments:')
    print(f'iter_max={args.iter_max}, file_name={args.file_name}, level_max={args.level_max}')

    main(args.iter_max, args.level_max, args.file_name)