import norx

# Core permutation F
# state is initialized by key and a nonce
# each round of F applies a 4w-bit permutation G
    # column step: apply G in each of 4 columns
    # diagonal step: apply G in each of 4 diagonals
# F outputs the state after 4 (6) rounds

# NORX, a very fast authenticated encryption algorithm with associated data (AEAD). NORX is a 3rd-round candidate to CAESAR.
# default NORX 64-4-1 variant consists of state 16 64-bit words, four rounds, no parallel execution, key and nonce are 256 bits - see https://github.com/norx/resources
# NORX32, a variant of NORX intended for smaller architectures (32-bit and less). Key and nonce are 128 bits.



F = norx.NORX_F(64)
state = F.new(nonce, key)
F(state)

norx.runtests()
print('a')