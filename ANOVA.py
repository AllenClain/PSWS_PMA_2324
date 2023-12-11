import numpy as np
import pandas as pd
from scipy import stats


def f_test(s1, s2):
    sample1, sample2 = np.array(s1), np.array(s2)
    var1, var2 = sample1.var(ddof=1), sample2.var(ddof=1)
    f = var1 / var2
    if f > 1: # Use f statistic that is larger than 1 as an agreement
        p = stats.f.sf(f, len(s1) - 1, len(s2) - 1)
    else:
        p = stats.f.cdf(f, len(s1) - 1, len(s2) - 1)
    print(f'F Statistic = {f:.4f}, P Value = {p:.4f}')
    

class oneway_ANOVA:
    def __init__(self, sample=[]):
        self.sample = sample
        self.alldata = np.concatenate(sample)
        self.groupnumber = len(sample)
        self.result = self.__f()

    def __ss(self, sample): # calculate 'sum of squares' of a given sample
        nsample = np.array(sample)
        __mean = nsample.mean()
        return ((nsample - __mean) ** 2).sum()

    def __f(self): # core ANOVA process
        ssw = sum([self.__ss(s) for s in self.sample])
        sst = self.__ss(self.alldata)
        ssg = sst - ssw
        d1 = self.groupnumber - 1
        d2 = len(self.alldata) - self.groupnumber
        f = ssg  * d2 / d1 / ssw
        prob = stats.f.sf(f, d1, d2)
        return {
            'Source of Variation': ['Between Groups', 'Within Groups', 'Total'],
            'Sums of Squares': [ssg, ssw, sst],
            'Degrees of Freedom': [d1, d2, d1 + d2],
            'Mean Squares': [ssg / d1, ssw / d2],
            'F Ratio': [f],
            'P Value': [prob]
        }

    def showresult(self): # present result dictionary in commonly used table
        __result_clean = self.result.copy()
        __result_clean['Mean Squares'] += [None]
        __result_clean['F Ratio'] += [None] * 2
        __result_clean['P Value'] += [None] * 2
        res = pd.DataFrame(__result_clean).round(4).fillna('')#.astype('str')

        # prepare table in lines
        title = 'ANOVA Result Table'
        label = '   '.join(res.columns)
        topline = '_' * len(label)
        titleline = '-' * len(label)
        bottomline = '_' * len(label)
        dataset = ['', '', '']
        for j in range(3):
            for i, block in enumerate(res.iloc[j]): # I love f-string!
                dataset[j] += f'{block:<{len(res.columns[i])}}' if type(block) is str else f'   {block:>{len(res.columns[i])}}'

        # send to standard output
        print(title)
        print(topline)
        print(label)
        print(titleline)
        print('\n'.join(dataset))
        print(bottomline)


class twoway_ANOVA:
    def __init__(self, sample=[]): # sample must be 3-dimension
        self.sample = np.array(sample)
        self.alldata = np.concatenate(sample, axis=None)
        self.blocknumber = self.sample.shape[0]
        self.groupnumber = self.sample.shape[1]
        self.result = self.__f()

    def __ss(self, sample): # calculate 'sum of squares' of a given sample
        nsample = np.array(sample)
        __mean = nsample.mean()
        return ((nsample - __mean) ** 2).sum()

    def __f(self): # core ANOVA process
        # replace raw data with series' sum and size
        mat = np.array([[[group.sum(), group.shape[0]] for group in block] for block in self.sample])
        # X_a = \sum_{i=1}^a\left(\frac{\sum_{j=1}^{b}S_{ij}}{\sum_{j=1}^{b}t_{ij}}\right)
        xa = np.array([block[0] / block[1] for block in mat.sum(axis=1)])
        # X_b = \sum_{j=1}^b\left(\frac{\sum_{i=1}^{a}S_{ij}}{\sum_{i=1}^{a}t_{ij}}\right)
        xb = np.array([group[0] / group[1] for group in mat.sum(axis=0)])
        # \bar\bar{X} = \frac{\sum_{i=1}^a\sum_{j=1}^{b}S_{ij}}{}
        xx = mat.sum(axis=(0, 1))[0] / mat.sum(axis=(0, 1))[1]
        ssa = ((xa - xx)**2 * np.array([block[1] for block in mat.sum(axis=1)])).sum()
        ssb = ((xb - xx)**2 * np.array([group[1] for group in mat.sum(axis=0)])).sum()
        ssab = ((np.array([[group[0] / group[1] for group in block] for block in mat]) - np.array([xa]).T - np.array([xb]) + xx)**2 * np.array([[group[1] for group in block] for block in mat])).sum(axis=(0, 1))
        sst = self.__ss(self.alldata)
        sse = sst - ssa - ssb - ssab
        
        dn = len(self.alldata) - 1
        da = self.blocknumber - 1
        db = self.groupnumber - 1
        dab = da * db
        de = dn - da - db - dab
        fa = ssa * de / da / sse
        fb = ssb * de / db / sse
        fab = ssab * de / dab / sse
        pa = stats.f.sf(fa, da, de)
        pb = stats.f.sf(fb, db, de)
        pab = stats.f.sf(fab, dab, de)
        
        return {
            'Source of Variation': ['Between Groups', 'Between Blocks', 'Interaction', 'Error', 'Total'],
            'Sums of Squares': [ssb, ssa, ssab, sse, sst],
            'Degrees of Freedom': [db, da, dab, de, dn],
            'Mean Squares': [ssb / db, ssa / da, ssab / dab, sse / de],
            'F Ratio': [fb, fa, fab],
            'P Value': [pb, pa, pab]
        }

    def showresult(self): # present result dictionary in commonly used table
        __result_clean = self.result.copy()
        __result_clean['Mean Squares'] += [None]
        __result_clean['F Ratio'] += [None] * 2
        __result_clean['P Value'] += [None] * 2
        res = pd.DataFrame(__result_clean).round(4).fillna('')#.astype('str')

        # prepare table in lines
        title = 'ANOVA Result Table'
        label = '   '.join(res.columns)
        topline = '_' * len(label)
        titleline = '-' * len(label)
        bottomline = '_' * len(label)
        dataset = ['', '', '', '', '']
        for j in range(5):
            for i, block in enumerate(res.iloc[j]): # I love f-string!
                dataset[j] += f'{block:<{len(res.columns[i])}}' if type(block) is str else f'   {block:>{len(res.columns[i])}}'

        # send to standard output
        print(title)
        print(topline)
        print(label)
        print(titleline)
        print('\n'.join(dataset))
        print(bottomline)