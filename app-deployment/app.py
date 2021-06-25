import flask
import pickle
import re
from collections import Counter

# Use pickle to load in the pre-trained model.
with open(f'model/CLFmultinomialNB.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'model/train_X_vect.pkl', 'rb') as f:
    vectorizer_trained = pickle.load(f)
with open(f'model/TfidfTransformer.pkl', 'rb') as f:
    tfidfTransformer = pickle.load(f)

outputDict = {0: 'Barack Obama', 1: 'Elon Musk', 2: 'Justin Bieber', 3: 'Donald Trump'}

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # get inputtweet from form
        inputTweet = flask.request.form['inputTweet']

        # clean the inputtweet to match training data
        inputTweet = cleanTweet(inputTweet)

        # Prepare inputtweet for model input
        test_X_counts = vectorizer_trained.transform([inputTweet])
        test_X_tfidf = tfidfTransformer.transform(test_X_counts)

        # make prediction with trained model
        prediction = model.predict(test_X_tfidf)[0]

        # transform numeric output to readable string output
        outputName = outputDict.get(prediction)

        return flask.render_template('main.html', original_input={'Input tweet': inputTweet}, result=outputName)


def cleanTweet(tweet):
    tweet = re.sub(r'@', '', tweet)
    tweet = re.sub(r'http.?://[^\s]+[\s]?', '', tweet)
    tweet = re.sub('[^a-zA-Z\s]', '', tweet)
    tweet = re.sub('\\n', '', tweet)
    tweet = tweet.lstrip()
    tweet = tweet.rstrip()
    tweet = tweet.lower()

    tweet = ' '.join(map(correction, tweet.split()))
    return tweet


# Spelling checker
def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('D:/Development/Techonony/Intake exercise/Data/big.txt').read()))


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


if __name__ == '__main__':
    app.run()
