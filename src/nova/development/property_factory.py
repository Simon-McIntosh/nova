
def quantity(storage_name):   

    def qty_getter(instance):  # B   
        return instance.__dict__[storage_name]   

    def qty_setter(instance, value):   
        if value > 0:
            instance.__dict__[storage_name] = value   
        else:
            raise ValueError('value must be > 0')

    return property(qty_getter, qty_setter)      

class LineItem:
    
    weight = quantity('weight')   # A
    price = quantity('price')   

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight   
        self.price = price

    def subtotal(self):
        return self.weight * self.price
    
if __name__ == '__main__':
    
    line = LineItem('s', 10, 50)
    