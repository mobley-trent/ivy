import ivy
import numpy as np
import scipy
import ivy.functional.frontends.numpy as ivy_np


class Vector:
    def toArray(self):
        raise NotImplementedError

    def asML(self):
        raise NotImplementedError


class DenseVector(Vector):
    def __init__(self, ar):
        if isinstance(ar, bytes):
            ar = ivy.frombuffer(ar, dtype=ivy.float64)
        elif not isinstance(ar, ivy.Array):
            ar = ivy.array(ar, dtype=ivy.float64)
        if ar.dtype != ivy.float64:
            ar = ivy.astype(ar, ivy.float64)
        self.array = ar

    @staticmethod
    def parse(s):
        start = s.find("[")
        if start == -1:
            raise ValueError("Array should start with '['.")
        end = s.find("]")
        if end == -1:
            raise ValueError("Array should end with ']'.")
        s = s[start + 1 : end]

        try:
            values = [float(val) for val in s.split(",") if val]
        except ValueError:
            raise ValueError("Undable to parse values from %s" % s)
        return DenseVector(values)

    def __reduce__(self):
        return DenseVector, (self.array.tostring(),)

    def numNonzeros(self):
        return ivy.count_nonzero(self.array)

    def norm(self, p):
        return ivy_np.linalg.norm(self.array, p)

    def dot(self, other):
        if type(other) == np.ndarray:
            if other.ndim > 1:
                assert len(self) == other.shape[0], "dimension mismatch"
            return ivy.dot(self.array, other)
        elif _have_scipy and scipy.sparse.issparse(other):
            assert len(self) == other.shape[0], "dimension mismatch"
            return other.transpose().dot(self.toArray())
        else:
            assert len(self) == _vector_size(other), "dimension mismatch"
            if isinstance(other, SparseVector):
                return other.dot(self)
            elif isinstance(other, Vector):
                return ivy.dot(self.toArray(), other.toArray())
            else:
                return ivy.dot(self.toArray(), other)

    def squared_distance(self, other):
        assert len(self) == _vector_size(other), "dimension mismatch"
        if isinstance(other, SparseVector):
            return other.squared_distance(self)
        elif _have_scipy and scipy.sparse.issparse(other):
            return _convert_to_vector(other).squared_distance(self)

        if isinstance(other, Vector):
            other = other.toArray()
        elif not isinstance(other, ivy.Array):
            other = ivy.array(other)
        diff = self.toArray() - other
        return ivy.dot(diff, diff)

    def toArray(self):
        return self.array

    def asML(self):
        return NotImplementedError
