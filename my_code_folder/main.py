"""
UtilityHub: A collection of diverse Python functions for general-purpose tasks.
This file serves as a comprehensive example of Python syntax, docstrings, 
and functional programming.
"""

import math
import random
import json
import time
from datetime import datetime
from functools import wraps

# --- DECORATORS ---

def execution_timer(func):
    """Logs the time it takes for a function to execute."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"[Timer] {func.__name__} took {end_time - start_time:.4f}s")
        return result
    return wrapper

# --- MATHEMATICAL UTILITIES ---

def calculate_bmi(weight_kg, height_m):
    """Calculates Body Mass Index."""
    try:
        return round(weight_kg / (height_m ** 2), 2)
    except ZeroDivisionError:
        return 0.0

def get_fibonacci_sequence(n_terms):
    """Generates a list containing the Fibonacci sequence up to n terms."""
    sequence = [0, 1]
    while len(sequence) < n_terms:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n_terms]

def solve_quadratic(a, b, c):
    """Solves the quadratic equation ax^2 + bx + c = 0."""
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # Complex roots not handled here
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    return (root1, root2)

# --- STRING & TEXT MANIPULATION ---

def reverse_string_manual(text):
    """Reverses a string using slicing."""
    return text[::-1]

def count_vowels(text):
    """Returns the count of vowels in a string."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)

def is_palindrome(text):
    """Checks if a string is a palindrome, ignoring case and spaces."""
    clean_text = "".join(text.split()).lower()
    return clean_text == clean_text[::-1]

def generate_random_password(length=12):
    """Generates a random alphanumeric password."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return "".join(random.choice(chars) for _ in range(length))

# --- LIST & DATA PROCESSING ---

@execution_timer
def bubble_sort(data):
    """A standard bubble sort implementation."""
    arr = data.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def filter_even_numbers(numbers):
    """Returns only even numbers from a list."""
    return [num for num in numbers if num % 2 == 0]

def get_list_statistics(numbers):
    """Returns a dictionary with min, max, average, and sum."""
    if not numbers:
        return {}
    return {
        "min": min(numbers),
        "max": max(numbers),
        "avg": sum(numbers) / len(numbers),
        "total": sum(numbers)
    }

# --- FILE & DATA MOCKING ---

def save_to_json(data, filename="output.json"):
    """Saves a dictionary to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def simulate_api_call(endpoint):
    """Simulates a network request delay and returns mock data."""
    print(f"Fetching data from {endpoint}...")
    time.sleep(1.5)  # Simulate network latency
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "data": {"id": random.randint(100, 999), "active": True}
    }

# --- MAIN EXECUTION BLOCK ---

def main():
    print("--- UtilityHub Dashboard ---")
    
    # Math Test
    print(f"Fibonacci (10): {get_fibonacci_sequence(10)}")
    
    # String Test
    pwd = generate_random_password(16)
    print(f"Secure Password: {pwd}")
    
    # Sorting Test
    unsorted = [64, 34, 25, 12, 22, 11, 90]
    print(f"Sorted List: {bubble_sort(unsorted)}")
    
    # API Mock Test
    response = simulate_api_call("/api/v1/user")
    print(f"Mock Response: {response}")
    
    # Statistics Test
    stats = get_list_statistics(unsorted)
    print(f"List Stats: {stats}")

if __name__ == "__main__":
    main()