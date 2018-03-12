import numpy as np


def count_entropy(cnt):
    return -sum((p * np.log(p) for p in cnt / sum(cnt) if p != 0))


def count_gini(cnt):
    return 1 - sum((cnt / sum(cnt))**2)


def count_criterion_generator(count_criterion):
    def generator(y):
        cnt_l = np.zeros(max(y) + 1)
        cnt_r = np.zeros(max(y) + 1)
        for value in y:
            cnt_r[value] += 1
        for value in y[:-1]:
            cnt_l[value] += 1
            cnt_r[value] -= 1
            yield (count_criterion(cnt_l), count_criterion(cnt_r))
    return generator


def variance_generator(y):
    sum_l = 0
    sum_r = sum(y)
    sum2_l = 0
    sum2_r = sum(y**2)

    def var(s, s2, n):
        return s2 / n - (s / n)**2

    cnt = 0
    for value in y[:-1]:
        sum_l += value
        sum2_l += value**2
        sum_r -= value
        sum2_r -= value**2
        cnt += 1
        yield (var(sum_l, sum2_l, cnt), var(sum_r, sum2_r, len(y) - cnt))


# TODO: implement by heap
def mad_median_generator(y):
    left = []
    right = list(y)
    for i in range(y.shape[0]):
        left.append(right.pop())
        med_l, med_r = np.median(left), np.median(right)
        score_l = sum((np.abs(val - med_l) for val in left))
        score_r = sum((np.abs(val - med_r) for val in right))
        yield score_l, score_r


criterion_generators = {
    "gini": count_criterion_generator(count_gini),
    "entropy": count_criterion_generator(count_entropy),
    "variance": variance_generator,
    "mad_median": mad_median_generator
}
