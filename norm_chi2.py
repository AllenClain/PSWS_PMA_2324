import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# testing normality using chi^2 method
class norm_chi2:
    def __init__(self, sample, title='', xlabel=''):
        self.sample = sample
        self.samp_title = title
        self.xlabel = xlabel
        self.dscb = self.__describe()
        self.chi2_result = self.chi2()

    def chi2(self): # core chi_square calculation
        # Observed statistics
        k_float = 1 + np.log2(self.dscb['count'])
        k = round(1 + np.log2(self.dscb['count']) + 0.5)
        hist, bin_edges = np.histogram(self.sample, bins=k)
        class_range = np.ptp(self.sample) / k

        # Expected statistics
        cdf = stats.norm.cdf(bin_edges + class_range, self.dscb['mean'], self.dscb['std'])[:-1]
        bin_freq = cdf - np.insert(cdf, 0, 0)[:-1]
        exp_val = bin_freq * self.dscb['count']
        exp_freq_norm = exp_val * hist.sum() / exp_val.sum()

        # chi_square goodness
        chi2 = (hist - exp_val) ** 2 / exp_val
        chi2_stat, p_value = stats.chisquare(hist, f_exp=exp_freq_norm, ddof=2)
        
        return {
            'obsv': {
                'k_f': k_float,
                'k': k,
                'hist': hist,
                'bin': bin_edges,
                'class': class_range
            },
            'exp': {
                'cdf': cdf,
                'bin freq': bin_freq,
                'exp val': exp_val
            },
            'chi2': {
                'chi2': chi2,
                'chi2 stat': chi2_stat,
                'p': p_value
            }
        }

    
    def showresult(self):
        print(f'**Normality Test Result using Chi^2 method for {self.samp_title}**')
        print('-' * 128)
        
        # Descriptive statistic
        descriptive_statistic_table = self.__table(
            title = 'Descriptive statistics',
            labels = f'  {self.xlabel}  ',
            dataset = [
                ('Count', self.dscb['count']),
                ('Mean', self.dscb['mean']),
                ('Std', self.dscb['std']),
                ('Min', self.dscb['min']),
                ('Max', self.dscb['max'])
            ]
        )

        # Quantiles and outliers
        quantile_outlier_table = self.__table(
            title = '',
            labels = '     Quantiles and Outliers     ',
            dataset = [
                ('Quantile 1 (25%)', self.dscb['25%']),
                ('Quantile 2 (50%, Median)', self.dscb['50%']),
                ('Quantile 3 (75%)', self.dscb['75%']),
                ('IQR (Q3 - Q1)', self.dscb['75%'] - self.dscb['25%']),
                ('Extreme min', 2.5 * self.dscb['25%'] - 1.5 * self.dscb['75%']),
                ('Extreme max', 2.5 * self.dscb['75%'] - 1.5 * self.dscb['25%'])
            ]
        )

        # Class range
        class_range_table = self.__table(
            title = '',
            labels = '   Classes description   ',
            dataset = [
                ('Number of Classes', self.chi2_result['obsv']['k_f']),
                ('Rounded number', self.chi2_result['obsv']['k']),
                ('Class Range', self.chi2_result['obsv']['class'])
            ]
        )

        # Observation and expectation comparison
        res = pd.DataFrame(
            {
                'Intervals': self.chi2_result['obsv']['bin'][:-1] + self.chi2_result['obsv']['class'],
                'Frequency': self.chi2_result['obsv']['hist'],
                'CDF Values': self.chi2_result['exp']['cdf'],
                'Bin Frequency': self.chi2_result['exp']['bin freq'],
                'Exp Value': self.chi2_result['exp']['exp val'],
                'Chi Square': self.chi2_result['chi2']['chi2']
            }
        )
        res.loc['sum'] = res.sum(axis=0)
        res.loc['sum', 'Intervals'], res.loc['sum', 'CDF Values'] = None, None
        res.fillna('')
        
        obsv_exp_comp_table = self.__table(
            title = 'Final Chi^2 Table',
            labels = 'Index   ' + '   '.join(res.columns),
            dataset = [
                (
                    f'{res.index[j]:>5}', 
                    '   ' + '   '.join([
                        f'{block:>{len(res.columns[i])}.2f}' if i < len(res.columns) - 1 else f'{block:>{len(res.columns[i])}.4f}' 
                        for i, block in enumerate(res.iloc[j])
                    ])
                )
                for j in range(res.shape[0])
            ]
        )

        # send to standard output
        descriptive_statistic_table.show()
        quantile_outlier_table.show()
        class_range_table.show()
        self.plot()    
        obsv_exp_comp_table.show()
        print(f"Chi^2 Statistic: {res.loc['sum', 'Chi Square']:.4f}")
        print(f"P Value: {self.chi2_result['chi2']['p']:.4f}")
        print('-' * 128, end='\n\n')

    class __table:
        def __init__(self, title, labels, dataset):
            self.title = title
            self.labels = labels
            self.dataset = self.build(dataset, len(labels))

        def show(self):
            topline = '_' * len(self.labels)
            labelline = '-' * len(self.labels)
            bottomline = '_' * len(self.labels)
                
            print(self.title)
            print(topline)
            print(self.labels)
            print(labelline)
            print('\n'.join(self.dataset))
            print(bottomline)

        def build(self, dataset, outlier):
            lines = [f'{data[0]}' for data in (dataset)]
            for i in range(len(lines)):
                if type(dataset[i][1]) is str:
                    lines[i] += dataset[i][1]
                elif type(dataset[i][1]) is int:
                    lines[i] += f'{dataset[i][1]:>{outlier - len(lines[i])}}'
                else:
                    lines[i] += f'{dataset[i][1]:>{outlier - len(lines[i])}.2f}'
            return lines

    def plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        self.__box_plot(ax[0])
        self.__histogram(ax[1])
        fig.suptitle(f'Plots of Breakdown intervals of {self.samp_title}')
        # plt.savefig(f'img/normality_test_result_{self.samp_title}.png', 
        #             transparent=True, 
        #             dpi=300
        #            )
        plt.show()
        
    def __describe(self):
        nsamp = np.array(self.sample)
        return {
            'count': nsamp.shape[0],
            'mean': nsamp.mean(),
            'std': nsamp.std(ddof=1), # sample out of population, whereas freedom should be n - 1
            'min': nsamp.min(),
            '25%': np.quantile(nsamp, 0.25),
            '50%': np.quantile(nsamp, 0.5),
            '75%': np.quantile(nsamp, 0.75),
            'max': nsamp.max()
        }
        
    def __box_plot(self, subplot):
        subplot.boxplot(self.sample, vert=False, patch_artist=True)
        subplot.axvline(self.dscb['mean'], color='red', linestyle='dashed', linewidth=1, label='Mean')
        subplot.set_xlabel(f'{self.xlabel}')
        subplot.set_yticks([1], [f'{self.samp_title}'])
        subplot.legend()

    def __histogram(self, subplot):
        subplot.hist(self.sample, bins=self.chi2_result['obsv']['bin'], alpha=0.7, color='blue', label='Observed')
        subplot.axvline(self.dscb['mean'], color='red', linestyle='dashed', linewidth=1, label='Mean')
        subplot.set_xlabel(f'{self.xlabel}')
        subplot.set_ylabel('Frequency')
        subplot.legend()
