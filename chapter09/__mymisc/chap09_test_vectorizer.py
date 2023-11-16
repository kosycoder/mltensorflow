import pickle
import re
import os
from nltk.corpus import stopwords
# from sklearn.linear_model import SGDClassifier

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
stop = stopwords.words('english')
# clf = SGDClassifier(loss='log', random_state=1)

# pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'),'wb'),protocol=4)
# pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'),'wb'),protocol=4)

from vectorizer import vect
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

import numpy as np
label = {0:'negative', 1:'positive'}
example = ["I love this movie. It's amazing."]
X = vect.transform(example)
print('Prediction:, %s\nPProbability: %.2f%%' %\
      (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
