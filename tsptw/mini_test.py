from main import load_customers_from_file, TSPTW, is_feasible

# customers = load_customers_from_file('n20w20.001.txt')
# print(customers)

tsptw = TSPTW(30, 8, 'n20w20.001.txt')
tsptw.best_route.path = [1, 17, 10, 20, 18, 19, 11, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]
print(is_feasible(tsptw.best_route))
print(tsptw.best_route.customers)
tsptw.solve()

