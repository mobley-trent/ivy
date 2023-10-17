import array
import struct
import numpy as np
import ivy
import scipy

from typing import Iterable, Tuple, cast

try:
    import scipy.sparse

    _have_scipy = True
except:
    _have_scipy = False


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
        return np.linalg.norm(self.array, p)

    def dot(self, other):
        if type(other) == ivy.Array:
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
        return DenseVector(self.array)

    @property
    def values(self):
        return self.array

    def __getitem__(self, item):
        return self.array[item]

    def __len__(self):
        return len(self.array)

    def __str__(self):
        return "[" + ",".join([str(v) for v in self.array]) + "]"

    def __repr__(self):
        return "DenseVector([%s])" % ", ".join(_format_float(i) for i in self.array)

    def __eq__(self, other):
        if isinstance(other, DenseVector):
            return ivy.array_equal(self.array, other.array)
        elif isinstance(other, SparseVector):
            if len(self) != other.size:
                return False
            return Vectors._equals(
                list(range(len(self))), self.array, other.indices, other.values
            )
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        size = len(self)
        result = 31 + size
        nnz = 0
        i = 0
        while i < size and nnz < 128:
            if self.array[i] != 0:
                result = 31 * result + i
                bits = _double_to_long_bits(self.array[i])
                result = 31 * result + (bits ^ (bits >> 32))
                nnz += 1
            i += 1
        return result

    def __getattr__(self, item):
        return getattr(self.array, item)

    def __neg__(self):
        return DenseVector(-self.array)

    def _delegate(op):
        def func(self, other):
            if isinstance(other, DenseVector):
                other = other.array
            return DenseVector(getattr(self.array, op)(other))

        return func

    __add__ = _delegate("__add__")
    __sub__ = _delegate("__sub__")
    __mul__ = _delegate("__mul__")
    __div__ = _delegate("__div__")
    __truediv__ = _delegate("__truediv__")
    __mod__ = _delegate("__mod__")
    __radd__ = _delegate("__radd__")
    __rsub__ = _delegate("__rsub__")
    __rmul__ = _delegate("__rmul__")
    __rdiv__ = _delegate("__rdiv__")
    __rtruediv__ = _delegate("__rtruediv__")
    __rmod__ = _delegate("__rmod__")


class SparseVector(Vector):
    def __init__(self, size, *args):
        self.size = int(size)
        assert 1 <= len(args) <= 2, "must pass either 2 or 3 arguments"
        if len(args) == 1:
            pairs = args[0]
            if type(pairs) == dict:
                pairs = pairs.items()
            pairs = cast(Iterable[Tuple[int, float]], sorted(pairs))
            self.indices = ivy.array([p[0] for p in pairs], dtype=ivy.int32)
            self.values = ivy.array([p[1] for p in pairs], dtype=ivy.float64)
        else:
            if isinstance(args[0], bytes):
                assert isinstance(args[1], bytes), "Values should be string too"
                if args[0]:
                    self.indices = ivy.frombuffer(args[0], dtype=ivy.int32)
                    self.values = ivy.frombuffer(args[1], dtype=ivy.float64)
                else:
                    self.indices = ivy.array([], dtype=ivy.int32)
                    self.values = ivy.array([], dtype=ivy.float64)
            else:
                self.indices = ivy.array(args[0], dtype=ivy.int32)
                self.values = ivy.array(args[1], dtype=ivy.float64)
            assert len(self.indices) == len(
                self.values
            ), "indices and values arrays must have the same length"
            for i in range(len(self.indices) - 1):
                if self.indices[i] >= self.indices[i + 1]:
                    raise TypeError(
                        "Indices %s and %s are not strictly increasing"
                        % (self.indices[i], self.indices[i + 1])
                    )

        if self.indices.size > 0:
            assert (
                ivy.max(self.indices) < self.size
            ), "Index %d is out of the size of vector with size=%d" % (
                ivy.max(self.indices),
                self.size,
            )
            assert ivy.min(self.indices) >= 0, "Contains negative index %d" % (
                ivy.min(self.indices)
            )

    def numNonzeros(self):
        return ivy.count_nonzero(self.values)

    def norm(self, p):
        return np.linalg.norm(self.values, p)

    def __reduce__(self):
        return (
            SparseVector,
            (self.size, self.indices.tobytes(), self.values.tobytes()),
        )

    def dot(self, other):
        if isinstance(other, ivy.Array):
            if other.ndim not in [2, 1]:
                raise ValueError(
                    "Cannot call dot with %d-dimensional array" % other.ndim
                )
            assert len(self) == other.shape[0], "dimension mismatch"
            return ivy.dot(self.values, other[self.indices])

        assert len(self) == _vector_size(other), "dimension mismatch"

        if isinstance(other, DenseVector):
            return ivy.dot(other.array[self.indices], self.values)

        elif isinstance(other, SparseVector):
            # TODO: implement ivy.in1d
            self_cmind = np.in1d(self.indices, other.indices, assume_unique=True)
            self_values = self.values[self_cmind]
            if self_values.size == 0:
                return ivy.float64(0)
            else:
                other_cmind = np.in1d(other.indices, self.indices, assume_unique=True)
                return ivy.dot(self_values, other.values[other_cmind])

        else:
            return self.dot(_convert_to_vector(other))

    def squared_distance(self, other):
        assert len(self) == _vector_size(other), "dimension mismatch"

        if isinstance(other, DenseVector) or isinstance(other, ivy.Array):
            if isinstance(other, ivy.Array) and ivy.get_num_dims(other) != 1:
                raise ValueError(
                    "Cannot call squared_distance with %d-dimensional array"
                    % ivy.get_num_dims(other)
                )
            if isinstance(other, DenseVector):
                other = other.array
            sparse_ind = ivy.zeros(other.size, dtype=ivy.bool)
            sparse_ind[self.indices] = True
            dist = other[sparse_ind] - self.values
            result = ivy.dot(dist, dist)

            other_ind = other[~sparse_ind]
            result += ivy.dot(other_ind, other_ind)
            return result

        elif isinstance(other, SparseVector):
            result = 0.0
            i, j = 0, 0
            while i < len(self.indices) and j < len(other.indices):
                if self.indices[i] == other.indices[j]:
                    diff = self.values[i] - other.values[j]
                    result += diff * diff
                    i += 1
                    j += 1
                elif self.indices[i] < other.indices[j]:
                    result += self.values[i] * self.values[i]
                    i += 1
                else:
                    result += other.values[j] * other.values[j]
                    j += 1
            while i < len(self.indices):
                result += self.values[i] * self.values[i]
                i += 1
            while j < len(other.indices):
                result += other.values[j] * other.values[j]
                j += 1
            return result
        else:
            return self.squared_distance(_convert_to_vector(other))

    def toArray(self):
        arr = ivy.zeros((self.size,), dtype=ivy.float64)
        arr[self.indices] = self.values
        return arr

    def asML(self):
        return SparseVector(self.size, self.indices, self.values)

    def __len__(self):
        return self.size

    def __str__(self):
        inds = "[" + ",".join([str(i) for i in self.indices]) + "]"
        vals = "[" + ",".join([str(v) for v in self.values]) + "]"
        return "(" + ",".join((str(self.size), inds, vals)) + ")"

    def __repr__(self):
        inds = self.indices
        vals = self.values
        entries = ", ".join(
            [
                "{0}: {1}".format(inds[i], _format_float(vals[i]))
                for i in range(len(inds))
            ]
        )
        return "SparseVector({0}, {{{1}}})".format(self.size, entries)

    def __eq__(self, other):
        if isinstance(other, SparseVector):
            return (
                other.size == self.size
                and ivy.array_equal(other.indices, self.indices)
                and ivy.array_equal(other.values, self.values)
            )
        elif isinstance(other, DenseVector):
            if self.size != len(other):
                return False
            return Vectors._equals(
                self.indices, self.values, list(range(len(other))), other.array
            )
        return False

    def __getitem__(self, index):
        self.indices
        self.values
        if not isinstance(index, int):
            raise TypeError(
                "Indices must be of type integer, got type %s" % type(index)
            )

        if index >= self.size or index < -self.size:
            raise IndexError("Index %d out of bounds." % index)
        if index < 0:
            index += self.size


# --- Helpers --- #
# --------------- #


def _convert_to_vector(l):
    if isinstance(l, Vector):
        return l
    elif type(l) in (array.array, ivy.Array, list, tuple, range):
        return DenseVector(l)
    elif _have_scipy and scipy.sparse.issparse(l):
        assert l.shape[1] == 1, "Expected column vector"
        csc = l.tocsc()
        if not csc.has_sorted_indices:
            csc.sort_indices()
        return SparseVector(l.shape[0], csc.indices, csc.data)
    else:
        raise TypeError("Cannot convert type %s into Vector" % type(l))


def _double_to_long_bits(d):
    if ivy.isnan(value):
        value = float("nan")
    return struct.unpack("Q", struct.pack("d", value))[0]


def _format_float(f, digits=4):
    s = str(round(f, digits))
    if "." in s:
        s = s[: s.index(".") + 1 + digits]
    return s


def _vector_size(v):
    if isinstance(v, Vector):
        return len(v)
    elif type(v) in (array.array, list, tuple, range):
        return len(v)
    elif type(v) == ivy.Array:
        if v.ndim == 1 or (v.ndim == 2 and v.shape[1] == 1):
            return len(v)
        else:
            raise ValueError(
                "Cannot treat an array of shape %s as a Vector" % str(v.shape)
            )
    elif _have_scipy and scipy.sparse.issparse(v):
        assert v.shape[1] == 1, "Expected column vector"
        return v.shape[0]
    else:
        raise TypeError("Cannot treat type %s as a Vector" % type(v))
