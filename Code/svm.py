## This module trains the svm model based on given labelled encodings
# Saket 6/04/18

from p2 import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration options
TEST_SET_PERCENT = 0.2

# Input
#bins = []       # Bins contain sets of face_ids
#faces = []      # List of face class items

# Prepare data to be fed
data = []
for ind in range(len(bins)):
    data += [(faces[fid].enc,ind) for fid in bins[ind]]
X,Y = map(np.array, zip(*data)) ## Is this the original data?

# Prepare test, training sets
# N-fold validation?
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,TEST_SET_PERCENT)

# Train classifier - code taken from scikit site, kernel options yet to be configured
clf = SVC()
clf.fit(X,Y)
predicted = clf.predict(X_test)
acc = accuracy_score(Y_test, predicted)
print(acc)
