import math
import operator

import numpy as np


def binary_value(operation, *, reflected=False):
    def apply(self, other):
        left = self.value
        right = other.value if isinstance(other, ScalarOperations) else other
        return operation(right, left) if reflected else operation(left, right)

    return apply


def unary_value(operation):
    def apply(self):
        return operation(self.value)

    return apply


class ScalarOperations:
    __hash__ = None
    __array_priority__ = 1000

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def __format__(self, specification):
        return format(self.value, specification)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("Converting a sample value to a NumPy array requires a copy")
        return np.array(self.value, dtype=dtype)

    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    def __pow__(self, other, modulo=None):
        other = other.value if isinstance(other, ScalarOperations) else other
        return pow(self.value, other, modulo)

    def __rpow__(self, other):
        other = other.value if isinstance(other, ScalarOperations) else other
        return pow(other, self.value)

    __bool__ = unary_value(bool)
    __float__ = unary_value(float)
    __int__ = unary_value(int)
    __complex__ = unary_value(complex)
    __index__ = unary_value(operator.index)
    __floor__ = unary_value(math.floor)
    __ceil__ = unary_value(math.ceil)
    __trunc__ = unary_value(math.trunc)
    __neg__ = unary_value(operator.neg)
    __pos__ = unary_value(operator.pos)
    __abs__ = unary_value(abs)
    __invert__ = unary_value(operator.invert)

    __add__ = binary_value(operator.add)
    __radd__ = binary_value(operator.add, reflected=True)
    __sub__ = binary_value(operator.sub)
    __rsub__ = binary_value(operator.sub, reflected=True)
    __mul__ = binary_value(operator.mul)
    __rmul__ = binary_value(operator.mul, reflected=True)
    __truediv__ = binary_value(operator.truediv)
    __rtruediv__ = binary_value(operator.truediv, reflected=True)
    __floordiv__ = binary_value(operator.floordiv)
    __rfloordiv__ = binary_value(operator.floordiv, reflected=True)
    __mod__ = binary_value(operator.mod)
    __rmod__ = binary_value(operator.mod, reflected=True)
    __divmod__ = binary_value(divmod)
    __rdivmod__ = binary_value(divmod, reflected=True)
    __and__ = binary_value(operator.and_)
    __rand__ = binary_value(operator.and_, reflected=True)
    __or__ = binary_value(operator.or_)
    __ror__ = binary_value(operator.or_, reflected=True)
    __xor__ = binary_value(operator.xor)
    __rxor__ = binary_value(operator.xor, reflected=True)
    __lshift__ = binary_value(operator.lshift)
    __rlshift__ = binary_value(operator.lshift, reflected=True)
    __rshift__ = binary_value(operator.rshift)
    __rrshift__ = binary_value(operator.rshift, reflected=True)
    __eq__ = binary_value(operator.eq)
    __ne__ = binary_value(operator.ne)
    __lt__ = binary_value(operator.lt)
    __le__ = binary_value(operator.le)
    __gt__ = binary_value(operator.gt)
    __ge__ = binary_value(operator.ge)
