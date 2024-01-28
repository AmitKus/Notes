# Interview 3

## Questions

### Difference between staticmethod and classmethod

Staticmethod:
- A staticmethod is a method that is bound to a class rather than an instance of the class.
- It does not have access to the instance or class itself.

Classmethod:
- A classmethod is a method bound to the class and takes the class itself as its first argument (conventionally named cls).


```python
class MyClass:
    class_variable = 10

    @classmethod
    def my_class_method(cls, arg1, arg2):
        return cls.class_variable + arg1 + arg2

result = MyClass.my_class_method(3, 5)
```

```python
class MathOperations:
    @staticmethod
    def add(x, y):
        return x + y

result = MathOperations.add(3, 5)
```
