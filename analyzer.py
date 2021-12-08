import pandas as pd
from helper.utils import get_config
from matplotlib import pyplot as plt
from sklearn import preprocessing
import random

def start_analysis():
    show_class_samples_count()

def show_class_samples_count():
    print('\nTotal no of samples: ', train.shape[0], '\n')
    print('Class label sample distribution:')
    for label in labels:
        print(f'\tNo of samples with label {label}: ', train[(train['type'] == label)].shape[0])

def build_features():
    print('Building features...')
    train['word_count'] = train['tweet'].apply(lambda x: len(x.split()))
    train['char_count'] = train['tweet'].apply(lambda x: len(x.replace(" ", "")))
    train['word_density'] = train['word_count'] / (train['char_count'] + 1)
    train['punc_count'] = train['tweet'].apply(lambda x: len([a for a in x if a in punc]))
    train['total_length'] = train['tweet'].apply(len)
    train['capitals'] = train['tweet'].apply(lambda x: sum(1 for c in x if c.isupper()))
    train['caps_vs_length'] = train.apply(lambda x: (float(x['capitals'])/float(x['total_length'])), axis = 1)
    train['percentage_exclamation_marks'] = train.apply(lambda x: (float(x['tweet'].count('!')) * 100 / float(x['total_length'])), axis = 1)
    train['percentage_question_marks'] = train.apply(lambda x: (float(x['tweet'].count('?')) * 100 / float(x['total_length'])), axis = 1)
    train['num_ellipsis'] = train['tweet'].apply(lambda x: x.count('...'))
    train['num_symbols'] = train['tweet'].apply(lambda x: sum(x.count(w) for w in '*&$%=3#'))
    train['num_unique_words'] = train['tweet'].apply(lambda x: len(set(w for w in x.split())))
    train['words_vs_unique'] = train['num_unique_words']/train['word_count']
    train['unique_word_percentage'] = train['num_unique_words'] * 100 / train['word_count']

def visualize_features():
    print('Plotting features...')
    standard_scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    plt.figure(figsize=(24, 20))
    plt.hist(train['type'])
    plt.savefig(f'{visualization_path}/bar_plot_for_target.png')
    plt.clf()
    new_features=['word_count','char_count','word_density','punc_count','total_length','capitals','caps_vs_length','percentage_exclamation_marks','percentage_question_marks','num_ellipsis',
    'num_symbols','num_unique_words','words_vs_unique','unique_word_percentage']
    train_scaled = train[new_features]
    train_scaled['type'] = train['type']
    #train_scaled[new_features] = min_max_scaler.fit_transform(train_scaled[new_features])

    for feature in new_features:
        print(f'\tPlotting scatter plot for {feature}')
        x = train_scaled['type']
        y = train_scaled[feature]
        color_map={'Harmful_Traditional_practice': '#eb4034', 'Physical_violence': '#3443eb', 'economic_violence': '#0b9c2f', 'emotional_violence': '#e5f50f', 'sexual_violence': '#583b75'}
        cdf = pd.DataFrame(x)
        cdf['color'] = cdf['type'].apply(lambda x: color_map[x])
        plt.scatter(x, y, c=cdf['color'].to_list())
        plt.savefig(f'{visualization_path}/scatter_plot_for_{feature}.png')
        plt.clf()
    
    feature_colors=["#"+''.join([random.choice('1234567890ABCDEF') for j in range(6)]) for i in range(5)]
    for i, label in enumerate(labels):
        fig, ax = plt.subplots(5, 3, figsize=(50, 50))
        print(f'\tPrinting scatter plot for label {label}')
        for j, feature in enumerate(new_features):
            x = train_scaled[train_scaled['type'] == label]['type']
            y = train_scaled[train_scaled['type'] == label][feature]
            k = int(j/3)
            r = int(j%3)
            #fig.add_subplot(gs[r, k], xlabel=feature).scatter(x, y, c=feature_colors[i])
            ax[k][r].scatter(x, y)
            ax[k][r].set_xlabel(feature)
        plt.savefig(f'{visualization_path}/scatter_plot_for_{label}')
        plt.clf()


config = None
test = None
train = None
labels = None

input_path = "C:\\Users\\vimurali\\ML\\Gender Based Violence Text Classification\\input"
visualization_path = "C:\\Users\\vimurali\\ML\\Gender Based Violence Text Classification\\visualizations"
train = pd.read_csv(f'{input_path}/Train.csv')
test = pd.read_csv(f'{input_path}/Test.csv')
labels = train['type'].unique()
punc = [',','.',';',':','-','!','?','\'','""','...']
start_analysis()
build_features()
visualize_features()