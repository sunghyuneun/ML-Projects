class Vector:
    def __init__(self,components):
        if len(components) != 0:
            self.components = components
        else: 
            raise ValueError("Do not give an empty vector")

    def __repr__(self):
        return f"Vector({self.components})"
    
    def __add__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Must be same dimensions!")
        else:
            returnList = []
            for dim in range(len(self.components)):
                returnList.append(self.components[dim] + other.components[dim])
            returnVector = Vector(returnList)
            return returnVector
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            returnList = []
            for dim in range(len(self.components)):
                returnList.append(self.components[dim] * other)
            returnVector = Vector(returnList)
            return returnVector
        else:
            if len(self.components) != len(other.components):
                raise ValueError("Must be same dimensions!")
            else:
                dotproduct = 0
                for dim in range(len(self.components)):
                    dotproduct += (self.components[dim] * other.components[dim])
                return dotproduct
    
    def __eq__(self,other):
        if not isinstance(other, Vector) or len(self.components) != len(other.components):
            return False
        else:
            return self.components == other.components
        
    def dimension(self):
        return len(self.components)

def main():
    vector = [1,2,3]
    vector1 = Vector([1,2,3])
    vector2 = Vector([4,5,6])
    vector3 = vector1 + vector2
    print(repr(vector1))
    print(repr(vector2))
    print(repr(vector3))


if __name__ == "__main__":
    main()