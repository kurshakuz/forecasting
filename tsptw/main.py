import argparse
import time
import random
import logging
import os

from tsptw import TSPTW
from customer import Customer
from route import Route
from optimizer import Optimizer, is_feasible

# Set a seed for the random number generator to ensure reproducibility
random.seed(320)

# Set up logging
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Solves the Time-Dependent Traveling Salesman Problem (TSPTW).',
                                     add_help=True)
    parser.add_argument('-i', '--iter_max', type=int, default=30, help='Maximum number of iterations.')
    parser.add_argument('-f', '--file_name', type=str, default='./benchmarks/n20w20.001.txt', help='File name of the input data.')
    parser.add_argument('-l', '--level_max', type=int, default=8, help='Range of the local search.')
    parser.add_argument('-r', '--rdy', action='store_const', const='rdy', dest='initial_path_type', help='Sets initial path type to "rdy".')
    parser.add_argument('-d', '--due', action='store_const', const='due', dest='initial_path_type', help='Sets initial path type to "due".')
    parser.add_argument('-?', action='help', help='Show this message and exit.')

    parser.set_defaults(initial_path_type='random')

    args = parser.parse_args()
    
    # Validate the arguments
    assert os.path.isfile(args.file_name), f"Input file {args.file_name} doesn't exist."
    assert args.iter_max > 0, 'iter_max should be a positive integer.'
    assert args.level_max > 0, 'level_max should be a positive integer.'

    logging.info(f'Parsed command line arguments: iter_max={args.iter_max}, file_name={args.file_name}, level_max={args.level_max}, initial_path_type={args.initial_path_type}')
    
    return args


def main(iter_max, level_max, initial_path_type, file_name):
    """
    Main function that initiates and solves the TSPTW problem.
    """
    logging.info('Starting main function...')
    
    tsptw = TSPTW(iter_max, level_max, initial_path_type, file_name)
    result = tsptw.solve()
    # tsptw = TSPTW(iter_max, level_max, file_name, initial_path_type)
    # customers = tsptw.load_customers_from_file(file_name)
    # result = Route(customers)
    # 1 17 10 20 18 19 11 6 16 2 12 13 7 14 8 3 5 9 21 4 15 
    # result.path = [1, 17, 10, 20, 18, 19, 11, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]
    # 1 20 12 8 19 17 14 9 4 18 3 6 11 5 16 10 15 7 21 2 13 
    # result.path = [1, 20, 12, 8, 19, 17, 14, 9, 4, 18, 3, 6, 11, 5, 16, 10, 15, 7, 21, 2, 13]
    # 1 6 20 14 2 9 17 5 10 4 8 16 15 18 12 3 11 19 21 7 13 
    # result.path = [1, 6, 20, 14, 2, 9, 17, 5, 10, 4, 8, 16, 15, 18, 12, 3, 11, 19, 21, 7, 13]
    # 1 12 4 3 20 8 16 10 9 6 7 11 15 5 13 17 19 14 21 18 2 
    # result.path = [1, 12, 4, 3, 20, 8, 16, 10, 9, 6, 7, 11, 15, 5, 13, 17, 19, 14, 21, 18, 2]

    logging.info(f'Best route found: {result.path}')
    logging.info(f'Best route cost: {tsptw.optimizer.calculate_cost(result.path, result.customers)}')
    logging.info(f'Best route feasible: {is_feasible(result.path, result.customers, tsptw.optimizer.distance_matrix)}')

if __name__ == "__main__":
    args = parse_arguments()

    start_time = time.time()
    main(args.iter_max, args.level_max, args.initial_path_type, args.file_name)
    end_time = time.time()

    elapsed_time = end_time - start_time
    logging.info(f'The process took {elapsed_time:.3f} seconds.')
