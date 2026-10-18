"""Scalar fields: callables f(P (k,3)) -> (k,). Used by displace() on meshes and displace_sdf() on SDFs.
noise() is deterministic everywhere: lattice gradients come from an integer hash of (cell, seed), not an RNG."""
import numpy as np

_GRAD = np.array([(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                  (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1), (1, 1, 0), (-1, 1, 0), (0, -1, 1), (0, -1, -1)],
                 dtype=float)


def _hash(ix, iy, iz, seed):
    """32-bit integer hash of lattice coordinates (murmur3 finalizer); wraps identically on every platform."""
    u = np.uint32
    h = (ix.astype(u) * u(0x8DA6B343)) ^ (iy.astype(u) * u(0xD8163841)) ^ (iz.astype(u) * u(0xCB1AB31F))
    h ^= u((seed * 0x9E3779B9) & 0xFFFFFFFF)
    h ^= h >> u(16)
    h *= u(0x85EBCA6B)
    h ^= h >> u(13)
    h *= u(0xC2B2AE35)
    h ^= h >> u(16)
    return h


def gradient_noise(P, seed=0):
    """Single-octave 3D gradient (Perlin-style) noise at points P (k,3); about [-1, 1], 0 at lattice points."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    i0 = np.floor(P).astype(np.int64)
    f = P - i0
    u = f * f * f * (f * (f * 6 - 15) + 10)
    out = np.zeros(len(P))
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1 - u[:, 2]
                h = _hash(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
                g = _GRAD[(h & np.uint32(15)).astype(np.int64)]
                out += wx * wy * wz * (g * (f - (dx, dy, dz))).sum(axis=1)
    return out


def noise(seed=0, scale=1.0, strength=1.0, octaves=4, lacunarity=2.0, gain=0.5):
    """Seeded fbm field f(P) -> values in about [-strength, strength]. scale = feature size in world units;
    each octave is lacunarity x finer and gain x weaker. f.amplitude = strength and f.lip (max gradient)
    are bounds used by SDF ops."""
    amps = [gain ** o for o in range(int(octaves))]
    norm = sum(amps)

    def f(P):
        P = np.asarray(P, dtype=float)
        Q = P.reshape(-1, 3) / scale
        out = np.zeros(len(Q))
        for o, a in enumerate(amps):
            out += a * gradient_noise(Q * lacunarity ** o + 17.0 * o, seed + 7919 * o)
        out *= strength / norm
        return out if P.ndim > 1 else out[0]

    f.amplitude = float(abs(strength))
    f.lip = 3.0 * abs(strength) / norm * sum(a * lacunarity ** o for o, a in enumerate(amps)) / scale  # |grad| bound
    return f
