from __future__ import print_function, division
import numpy
import os

class NORX_F(object):
    """Funkcja rundowa NORX
       Użycie:
         f = NORX_F(32)
         state = f.new(nonce, key)
         for i in range(nrounds):
           f(state)
    """

    def __init__(self, w):
        """Utwórz funkcję rundową.  w wartość 32 lub 64"""
        assert (w in (32, 64))

        if w == 32:
            self.dtype = numpy.uint32
            self.mask = self.dtype(0x7fffffff)
            self.r = numpy.array([8, 11, 16, 31], dtype=self.dtype)
            self.u = numpy.array([
                [0x243f6a88,           0,          0,  0x85a308d3],
                [         0,           0,          0,           0],
                [0x13198a2e,  0x03707344, 0x254f537a,  0x38531d48],
                [0x839c6e83,  0xf97a3ae5, 0x8c91d88c,  0x11eafb59]
            ], dtype=self.dtype)
        elif w == 64:
            self.dtype = numpy.uint64
            self.mask = self.dtype(0x7fffffffffffffff)
            self.r = numpy.array([8, 19, 40, 63], dtype=self.dtype)
            self.u = numpy.array([
                [0x243f6a8885a308d3,                  0,                  0, 0x13198a2e03707344],
                [                 0,                  0,                  0,                  0],
                [0xa4093822299f31d0, 0x082efa98ec4e6c89, 0xae8858dc339325a1, 0x670a134ee52d7fa6],
                [0xc4316d80cd967541, 0xd21dfbf8b630b762, 0x375a18d261e7f892, 0x343d1f187d92285b]
            ], dtype=self.dtype)
        self.one = self.dtype(1)
        self.nbits = self.dtype(w)

    def new(self, nonce, key):
        """Zwróć nowy, zainicjowany stan.
            nonce: obiekt bajtowy o rozmiarze 4 lub a
                   numpy tablica kształtu (2,) typu dtype uint32 lub uint64 odpowiednio.
            klucz: albo obiekt bajtowy o rozmiarze w/2, albo a
                   numpy tablica kształtu (4,) typu dtype uint32 lub uint64, jeśli to konieczne."""
        state = self.u.copy()

        if isinstance(nonce, bytes):
            assert (len(nonce) == self.nbits // 4)
            state[0, 1:3] = numpy.fromstring(nonce, dtype=self.dtype)
        elif isinstance(nonce, self.dtype):
            assert (nonce.shape == (2,))
            state[0, 1:3] = nonce.copy()
        else:
            assert (False)

        if isinstance(key, bytes):
            assert (len(key) == self.nbits // 2)
            state[1, :] = numpy.fromstring(key, dtype=self.dtype)
        elif isinstance(key, self.dtype):
            assert (key.shape == (4,))
            state[1, :] = key.copy()
        else:
            assert (False)

        return state

    def __call__(self, state):
        """Wykonaj jedną rundę Norxa. Dotyczy to funkcji G
            na pionach, a następnie po przekątnych. To modyfikuje stan
            w miejscu."""
        assert (state.shape == (4, 4))

        # Columns
        for i in range(4):
            self.G(state[:, i])

        # Diagonals
        (xixs, yixs) = numpy.diag_indices(4)
        for i in range(4):
            index = (xixs, (yixs + i) % 4)
            s = state[index]
            self.G(s)
            state[index] = s

    def reverse(self, state):
        """Odwrotność funkcji F"""
        assert (state.shape == (4, 4))

        # Diagonals
        (xixs, yixs) = numpy.diag_indices(4)
        for i in range(4):
            index = (xixs, (yixs + i) % 4)
            s = state[index]
            self.G_reverse(s)
            state[index] = s

        # Columns
        for i in range(4):
            self.G_reverse(state[:, i])

    def nonlinear(self, x, y):
        return (x ^ y) ^ ((x & y) << self.one)

    def reverse_nonlinear(self, z, y):
        """Odwrotność funkcji nieliniowej()"""
        # Recover bit 0 of x
        prevx = (z ^ y) & self.one
        ret = prevx

        for i in numpy.arange(1, self.nbits, dtype=self.dtype):
            mask = self.one << i
            prevx = (z ^ y ^ ((prevx & y) << self.one)) & mask
            ret |= prevx
        return ret

    def G(self, row):
        """Zastosuj funkcję G (opartą na Cha-Cha) do rzędu
            4 elementy. To jest napisane, aby, miejmy nadzieję, zapewnić
            maksymalna przejrzystość - można to zrobić szybciej, nawet w
            pyton."""

        for i in range(2):
            row[0] = self.nonlinear(row[0], row[1])
            row[3] = self.rotr(row[3] ^ row[0], self.r[i * 2])

            row[2] = self.nonlinear(row[2], row[3])
            row[1] = self.rotr(row[1] ^ row[2], self.r[(i * 2) + 1])

    def G_reverse(self, row):
        """Odwrotność funkcji G"""
        for i in range(2):
            row[1] = self.rotl(row[1], self.r[3 - (i * 2)]) ^ row[2]
            row[2] = self.reverse_nonlinear(row[2], row[3])

            row[3] = self.rotl(row[3], self.r[3 - ((i * 2) + 1)]) ^ row[0]
            row[0] = self.reverse_nonlinear(row[0], row[1])

    def rotr(self, x, n):
        """Bitowy obrót w prawo, biorąc pod uwagę
            skonfigurowany rozmiar słowa."""
        mask = (self.one << n) - self.one
        return (x >> n) | ((x & mask) << (self.nbits - n))

    def rotl(self, x, n):
        """Bitowy obrót w lewo, biorąc pod uwagę
            skonfigurowany rozmiar słowa."""
        mask = (self.one << (self.nbits - n)) - self.one
        return (x >> (self.nbits - n)) | ((x & mask) << n)


def _bitcount(n):
    """Zwraca liczbę bitów w zbiorze n"""
    count = 0
    while n != 0:
        count += n & 1
        n >>= 1
    return count


# Zmień liczbę bitów na ufunc, umożliwiając bezpośrednie zastosowanie jej do tablicy numpy
bitcount = numpy.frompyfunc(_bitcount, 1, 1)


def runtests():

    n = NORX_F(32)
    numpy.set_printoptions(formatter={"int": "0x{0:08x}".format})

    nonce = os.urandom(8)
    key = os.urandom(16)
    state = n.new(nonce, key)

    print("Utworzono nowy stan:")
    print(state)
    print()

    permuted = state.copy()
    n(permuted)
    print("Po rundzie:")
    print(permuted)
    print()

    unpermuted = permuted.copy()
    n.reverse(unpermuted)
    print("Odwrócony:")
    print(unpermuted)
    assert ((state == unpermuted).all())
    print()

    print("Bity zmienione:")
    print(bitcount(permuted ^ unpermuted))
    print()

    print("Średnia dyfuzja rundowa dla funkcji G:")
    print("64:", test_g_diffusion_64())
    print("32:", test_g_diffusion_32())

    print("Średnia dyfuzja rundowa dla funkcji F:")
    print("64:", test_f_diffusion_64())
    print("32:", test_f_diffusion_32())


def test_g_diffusion_64():
    """Sprawdza, ile rund potrzeba, aby z jednego bitu ustawić >= 31 bitów"""
    F = NORX_F(64)
    total = 0

    for i in range(4):
        for j in range(64):
            row = numpy.fromstring(os.urandom(32), dtype=numpy.uint64)
            row[0:2] = numpy.zeros(2, dtype=numpy.uint64)
            orig = row.copy()
            row[i] = numpy.uint64(1 << j)
            bits = 1
            rounds = 0
            while bits < 128:
                F.G(row)
                bits = bitcount(row ^ orig).sum()
                rounds += 1
            total += rounds
    return total / (4 * 64)


def test_g_diffusion_32():
    """Sprawdź, ile rund potrzeba, aby z jednego bitu ustawić >= 16 bitów"""
    F = NORX_F(32)
    total = 0

    for i in range(4):
        for j in range(32):
            row = numpy.fromstring(os.urandom(16), dtype=numpy.uint32)
            row[0:2] = numpy.zeros(2, dtype=numpy.uint32)
            orig = row.copy()
            row[i] = numpy.uint32(1 << j)
            bits = 1
            rounds = 0
            while bits < 64:
                F.G(row)
                bits = bitcount(row ^ orig).sum()
                rounds += 1
            total += rounds
    return total / (4 * 32)


def test_f_diffusion_64():
    """Sprawdź, ile rund potrzeba, aby ustawić >= 32 bity z pojedynczego bitu w obszarze /c/"""
    F = NORX_F(64)
    total = 0

    for i in range(4):
        for j in range(2):
            for k in range(64):
                state = numpy.fromstring(os.urandom(128), dtype=numpy.uint64).reshape((4, 4))
                state[0:2, :] = numpy.zeros((2, 4), dtype=numpy.uint64)  # Attacker controlled
                orig = state.copy()
                state[i, j + 2] ^= numpy.uint64(1 << k)
                bits = 1
                rounds = 0
                while bits < 512:
                    F(state)
                    bits = bitcount(state ^ orig).sum()
                    rounds += 1
                total += rounds
    return total / (2 * 4 * 64)


def test_f_diffusion_32():
    """Sprawdź, ile rund potrzeba, aby ustawić >= 16 bitów z pojedynczego bitu w obszarze /c/"""
    F = NORX_F(32)
    total = 0

    for i in range(4):
        for j in range(2):
            for k in range(32):
                state = numpy.fromstring(os.urandom(64), dtype=numpy.uint32).reshape((4, 4))
                state[0:2, :] = numpy.zeros((2, 4), dtype=numpy.uint64)  # Attacker controlled
                orig = state.copy()
                state[i, j + 2] ^= numpy.uint32(1 << k)
                bits = 1
                rounds = 0
                while bits < 256:
                    F(state)
                    bits = bitcount(orig ^ state).sum()
                    rounds += 1
                total += rounds
    return total / (2 * 4 * 32)


if __name__ == "__main__":
    runtests()