"""Small statistics helpers."""


def mean(values):
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def clamp(value, low, high):
    return max(low, min(high, value))


def bucket(values, width):
    counts = {}
    for v in values:
        key = int(v // width)
        counts[key] = counts.get(key, 0) + 1
    return counts


def top_n(counts, n):
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
