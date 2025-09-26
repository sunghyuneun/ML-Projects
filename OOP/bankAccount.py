class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self,amount):
        if (isinstance(amount, (int, float)) and amount >= 0):
            self.balance += amount
        else:
            raise ValueError("Invalid amount to deposit")
        
    def withdraw(self,amount):
        if (isinstance(amount, (int, float)) and amount >= 0 and self.balance - amount >= 0):
            self.balance -= amount
        else:
            raise ValueError("Invalid amount to withdraw")
        
    def display_balance(self):
        string = f"${self.balance}"
        return string

    @property
    def owner(self):
        return self._owner
    
    @owner.setter
    def owner(self,owner):
        self._owner = owner

    @property
    def balance(self):
        return self._balance
    
    @balance.setter
    def balance(self,balance):
        self._balance = balance


def main():
    name = input("Name: ")
    bankaccount = BankAccount(name)
    bankaccount.deposit(10)
    print(bankaccount.display_balance())


if __name__ == "__main__":
    main()