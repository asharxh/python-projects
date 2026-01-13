class CoffeeMachine:
    def __init__(self):
        # Initial resources in the machine
        self.resources = {
            'water': 1000,  # in ml
            'milk': 500,    # in ml
            'coffee_beans': 300,  # in grams
            'money': 0      # in dollars
        }
        
        # Coffee recipes (amount of ingredients for each type)
        self.recipes = {
            'espresso': {'water': 50, 'milk': 0, 'coffee_beans': 18, 'price': 1.5},
            'latte': {'water': 200, 'milk': 150, 'coffee_beans': 24, 'price': 2.5},
            'cappuccino': {'water': 250, 'milk': 100, 'coffee_beans': 24, 'price': 3.0}
        }

    def report(self):
        """Print the current resource status."""
        print("\n--- Coffee Machine Resources ---")
        for resource, amount in self.resources.items():
            unit = 'ml' if resource in ['water', 'milk'] else 'grams' if resource == 'coffee_beans' else 'dollars'
            print(f"{resource.capitalize()}: {amount} {unit}")
    
    def check_resources(self, coffee_type):
        """Check if there are enough resources to make the selected coffee."""
        recipe = self.recipes[coffee_type]
        for item, amount in recipe.items():
            if item != 'price' and self.resources[item] < amount:
                print(f"Sorry, not enough {item}!")
                return False
        return True
    
    def process_payment(self, price):
        """Process the payment and check if it's enough."""
        print(f"The cost is ${price}. Please insert money.")
        total_inserted = float(input("Insert money: $"))
        if total_inserted >= price:
            change = round(total_inserted - price, 2)
            print(f"Payment successful! Your change is ${change}.")
            self.resources['money'] += price
            return True
        else:
            print("Sorry, not enough money. Refunding money.")
            return False
    
    def make_coffee(self, coffee_type):
        """Make the coffee by subtracting the resources."""
        recipe = self.recipes[coffee_type]
        for item, amount in recipe.items():
            if item != 'price':
                self.resources[item] -= amount
        print(f"Here is your {coffee_type}! Enjoy!")
    
    def refill_resources(self):
        """Refill the coffee machine resources (for maintenance)."""
        print("Refilling resources...")
        self.resources['water'] = 1000
        self.resources['milk'] = 500
        self.resources['coffee_beans'] = 300
        print("Resources refilled!")
    
    def start(self):
        """Start the coffee machine and handle user input."""
        while True:
            print("\nOptions: [espresso/latte/cappuccino] or [report/refill/exit]")
            choice = input("What would you like to do? ").lower()
            
            if choice == 'report':
                self.report()
            elif choice == 'refill':
                self.refill_resources()
            elif choice == 'exit':
                print("Turning off the coffee machine.")
                break
            elif choice in self.recipes:
                if self.check_resources(choice):
                    if self.process_payment(self.recipes[choice]['price']):
                        self.make_coffee(choice)
            else:
                print("Invalid option. Please try again.")

# Create a CoffeeMachine instance and start the machine
coffee_machine = CoffeeMachine()
coffee_machine.start()
