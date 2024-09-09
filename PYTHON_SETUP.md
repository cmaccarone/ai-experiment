# Python Environment Setup and Basics

This guide will help you set up your Python environment and get started with the basics of Python programming.

## Setting Up Your Python Environment

1. Install Python:
   - Download and install Python from [python.org](https://www.python.org/downloads/)
   - During installation, make sure to check the box that says "Add Python to PATH"

2. Verify installation:
   Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and type:
   ```
   python --version
   ```
   This should display the installed Python version.

3. Create a virtual environment:
   ```
   python -m venv myenv
   ```
   This creates a new virtual environment named "myenv".

4. Activate the virtual environment:
   - On Windows:
     ```
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source myenv/bin/activate
     ```

5. Install required packages:
   ```
   pip install tensorflow numpy matplotlib
   ```

## Python Basics

1. Running a Python script:
   ```
   python your_script.py
   ```

2. Python interactive mode:
   Type `python` in the terminal to enter interactive mode.

3. Basic Python syntax:
   ```python
   # This is a comment
   
   # Variables
   x = 5
   y = "Hello, World!"

   # Print to console
   print(y)

   # Conditional statements
   if x > 0:
       print("Positive number")
   else:
       print("Non-positive number")

   # Loops
   for i in range(5):
       print(i)

   # Functions
   def greet(name):
       return f"Hello, {name}!"

   print(greet("Alice"))
   ```

4. Importing modules:
   ```python
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt
   ```

5. List comprehension:
   ```python
   squares = [x**2 for x in range(10)]
   ```

6. Working with files:
   ```python
   # Reading a file
   with open('file.txt', 'r') as f:
       content = f.read()

   # Writing to a file
   with open('output.txt', 'w') as f:
       f.write("Hello, World!")
   ```

Virtual environments are crucial for managing project-specific dependencies. Activating a virtual environment isolates your project's Python environment from the system-wide Python installation. Here's what you need to know:

1. Purpose of activation:
   - Isolates project dependencies
   - Prevents conflicts between different projects
   - Ensures reproducibility of your development environment

2. What activation does:
   - Modifies your shell's PATH to prioritize the virtual environment's Python interpreter
   - Changes your shell prompt to indicate the active environment

3. How to activate:
   ```
   source myenv/bin/activate  # On macOS/Linux
   myenv\Scripts\activate     # On Windows
   ```

4. Effects of activation:
   - `python` command now refers to the virtual environment's Python interpreter
   - Packages installed via pip will be isolated to this environment

5. Deactivation:
   When you're done working on your project, deactivate the virtual environment:
   ```
   deactivate
   ```
   This restores your shell to its original state.

By using virtual environments, you ensure a clean, isolated setup for your neural network project, making it easier to manage dependencies and reproduce your development environment across different machines.