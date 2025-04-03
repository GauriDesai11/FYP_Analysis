import statistics

# Example list of numbers
numbers = [23, 21, 21, 20, 22, 22, 23, 21, 23, 21, 22, 25]

# Calculate average (mean)
average = statistics.mean(numbers)

# Calculate standard deviation
std_dev = statistics.stdev(numbers)  # For sample standard deviation
# If you want population std dev, use statistics.pstdev(numbers)

# Print results
print(f"Numbers: {numbers}")
print(f"Average: {average}")
print(f"Standard Deviation: {std_dev}")
