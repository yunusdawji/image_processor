import process as p
from sklearn import datasets, svm, metrics, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


dataset = p.load_data("./pot.csv","./targets.csv")

for depth in xrange(1,10,2):
    #clf = svm.SVC()
    clf = RandomForestClassifier(max_depth=depth, random_state=0, )

    #clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
    print("100% of the data is {}.".format(len(dataset.data)))

    # Get 4/5
    split_index = len(dataset.data)//5*4
    print("80% of the data is {}.".format(split_index))

    train_data = dataset.data[:split_index]
    test_data = dataset.data[split_index:]

    train_target = dataset.target[:split_index]
    test_target = dataset.target[split_index:]

    clf.fit(train_data, train_target)

    out = clf.predict(test_data[0:])

    mismatchcount  = 0.0
    for i, j in zip(out, test_target):
        if i != j :
            mismatchcount = mismatchcount + 1

    print mismatchcount / out.shape[0]