class Car:
    def __init__(self,make,model,year):
        self._make = make
        self._model = model
        self._year = year

    def get_description(self):
        string = f"This is a {self.year} {self.make} {self.model}"
        return string

    def get_age(self):
        return (2024 - self.year)


    @property
    def make(self):
        return self._make
    
    @make.setter
    def make(self,make):
        self._make = make

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self,model):
        self._model = model

    @property
    def year(self):
        return self._year
    
    @year.setter
    def year(self,year):
        self._year = year


def main():
    car = Car("Honda", "Odyssey", "2005")
    print(car.get_description())

if __name__ == "__main__":
    main()