import argparse
import logging
import random
import time

import numpy as np

from optimizer import is_feasible
from tsptw import TSPTW

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
    parser.add_argument('-l', '--level_max', type=int, default=8, help='Range of the local search.')
    parser.add_argument('-r', '--rdy', action='store_const', const='rdy', dest='initial_path_type', help='Sets initial path type to "rdy".')
    parser.add_argument('-d', '--due', action='store_const', const='due', dest='initial_path_type', help='Sets initial path type to "due".')
    parser.add_argument('-?', action='help', help='Show this message and exit.')

    parser.set_defaults(initial_path_type='random')

    args = parser.parse_args()

    # Validate the arguments
    assert args.iter_max > 0, 'iter_max should be a positive integer.'
    assert args.level_max > 0, 'level_max should be a positive integer.'

    logging.info(f'Parsed command line arguments: iter_max={args.iter_max}, level_max={args.level_max}, initial_path_type={args.initial_path_type}')

    return args


def main(iter_max, level_max, initial_path_type):
    """
    Main function that initiates and solves the TSPTW problem.
    """
    logging.info('Starting main function...')

    preds = np.array([[158.2681, 204.52983, 203.10486, 206.61021, 153.7534, 204.46507, 209.63512,
                       206.24173, 149.51596, 205.16443, 215.93907, 205.83598, 144.41257, 205.06213,
                       219.77278, 201.92041, 0., 0., 0., 0., 0.]])

    tsptw = TSPTW(iter_max, level_max, initial_path_type, file_name=None, preds=preds)

    for customer in tsptw.customers_list:
        result = tsptw.solve(customer)
        print(result.path)
        # logging.info(f'Best route found: {result.path}')
        # logging.info(f'Best roue cost: {tsptw.optimizer.calculate_cost(result.path, result.customers)}')
        # logging.info(f'Best route feasible: {is_feasible(result.path, result.customers, tsptw.optimizer.distance_matrix)}')
        # print(tsptw.best_route)

if __name__ == "__main__":
    args = parse_arguments()

    start_time = time.time()
    main(args.iter_max, args.level_max, args.initial_path_type)
    end_time = time.time()

    elapsed_time = end_time - start_time
    logging.info(f'The process took {elapsed_time:.3f} seconds.')