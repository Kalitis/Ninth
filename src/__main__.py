"""
Main entry point for Ninth VM
"""
from .vm import NinthVM


def main():
    """Main function to run the Ninth VM"""
    vm = NinthVM()
    print("Ninth v0.5.1 Initialized.")
    # Example test
    vm.execute('[2 4] [RANDN] [PEEK]')


if __name__ == "__main__":
    main()