import math
import time
import os
import sys
from typing import Union

class TerminalCalculator:
    def __init__(self):
        self.history = []
        self.memory = 0
        self.loading_frames = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def animate_loading(self, duration=0.5):
        for _ in range(int(duration * 8)):
            for frame in self.loading_frames:
                sys.stdout.write(f'\rCalculating {frame}')
                sys.stdout.flush()
                time.sleep(0.0625)
        print("\n")
        
    def display_welcome_animation(self):
        calculator_art = """
        ╔══════════════════════════════════════╗
        ║           Super Calculator           ║
        ║          Terminal Edition            ║
        ╚══════════════════════════════════════╝
        """
        for line in calculator_art.split('\n'):
            print(line)
            time.sleep(0.1)

    def show_help(self):
        help_text = """
Available Operations:
═══════════════════
Basic Operations:
  +  Addition         : 5 + 3
  -  Subtraction     : 10 - 4
  *  Multiplication  : 6 * 2
  /  Division        : 15 / 3
  ** Power           : 2 ** 3

Scientific Functions:
  sin(x)   : Sine of x (in radians)
  cos(x)   : Cosine of x (in radians)
  tan(x)   : Tangent of x (in radians)
  sqrt(x)  : Square root of x
  log(x)   : Natural logarithm of x
  log10(x) : Base-10 logarithm of x
  abs(x)   : Absolute value of x
  
Constants:
  pi : Mathematical constant π (3.14159...)
  e  : Mathematical constant e (2.71828...)

Memory Operations:
  m+  : Add to memory
  m-  : Subtract from memory
  mr  : Recall memory
  mc  : Clear memory

Special Commands:
  help    : Show this help message
  history : Show calculation history
  clear   : Clear screen
  q/quit  : Exit calculator
"""
        print(help_text)

    def evaluate_safe(self, expression: str) -> Union[float, str]:
        """Safely evaluate mathematical expressions."""
        # Handle memory operations
        if expression.strip().lower() == 'mr':
            return f"Memory: {self.memory}"
        elif expression.strip().lower() == 'mc':
            self.memory = 0
            return "Memory cleared"
        elif expression.strip().lower().startswith('m+'):
            try:
                value = float(self.evaluate_safe(expression[2:].strip()))
                self.memory += value
                return f"Added {value} to memory. Memory now: {self.memory}"
            except:
                return "Error: Invalid memory operation"
        elif expression.strip().lower().startswith('m-'):
            try:
                value = float(self.evaluate_safe(expression[2:].strip()))
                self.memory -= value
                return f"Subtracted {value} from memory. Memory now: {self.memory}"
            except:
                return "Error: Invalid memory operation"

        # Define safe functions and constants
        safe_dict = {
            'abs': abs,
            'round': round,
            'pow': pow,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'pi': math.pi,
            'e': math.e,
            'log': math.log,
            'log10': math.log10
        }
        
        try:
            # Replace scientific function names with safe versions
            for func in safe_dict:
                if func in expression:
                    expression = expression.replace(func, f"safe_dict['{func}']")
                    
            # Evaluate the expression
            result = eval(expression, {"__builtins__": None, "safe_dict": safe_dict})
            return float(result)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def format_result(self, result):
        """Format the result for display"""
        if isinstance(result, float):
            # Format float to reasonable precision
            if result.is_integer():
                return str(int(result))
            return f"{result:.6f}".rstrip('0').rstrip('.')
        return str(result)

    def run(self):
        self.clear_screen()
        self.display_welcome_animation()
        print("\nWelcome to Super Calculator! Type 'help' for commands.")
        
        while True:
            try:
                user_input = input("\n► ").strip().lower()
                
                if user_input in ['q', 'quit']:
                    print("\nThank you for using Super Calculator!")
                    break
                    
                elif user_input == 'help':
                    self.show_help()
                    continue
                    
                elif user_input == 'history':
                    if not self.history:
                        print("No calculations in history.")
                    else:
                        print("\nCalculation History:")
                        for i, calc in enumerate(self.history, 1):
                            print(f"{i}. {calc}")
                    continue
                    
                elif user_input == 'clear':
                    self.clear_screen()
                    continue
                    
                elif not user_input:
                    continue
                
                self.animate_loading()
                result = self.evaluate_safe(user_input)
                formatted_result = self.format_result(result)
                
                if not str(result).startswith('Error'):
                    self.history.append(f"{user_input} = {formatted_result}")
                
                print(f"Result: {formatted_result}")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                continue
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

def main():
    calc = TerminalCalculator()
    calc.run()

if __name__ == "__main__":
    main()