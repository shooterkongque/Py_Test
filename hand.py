import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from embedded_window import Window
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from micromlgen import port

if __name__ == '__main__':
    # assume you saved your recordings into a "data" folder
    LA = pd.read_excel('data/A.xlsx', header=1)
    LB = pd.read_excel('data/B.xlsx', header=1)
    LC = pd.read_excel('data/C.xlsx', header=1)
    LD = pd.read_excel('data/D.xlsx', header=1)
    LE = pd.read_excel('data/E.xlsx', header=1)
    LF = pd.read_excel('data/F.xlsx', header=1)
    LG = pd.read_excel('data/G.xlsx', header=1)
    LH = pd.read_excel('data/H.xlsx', header=1)
    LI = pd.read_excel('data/I.xlsx', header=1)
    LJ = pd.read_excel('data/J.xlsx', header=1)
    LK = pd.read_excel('data/K.xlsx', header=1)
    LL = pd.read_excel('data/L.xlsx', header=1)
    LM = pd.read_excel('data/M.xlsx', header=1)
    LN = pd.read_excel('data/N.xlsx', header=1)
    LO = pd.read_excel('data/O.xlsx', header=1)
    LP = pd.read_excel('data/P.xlsx', header=1)
    LQ = pd.read_excel('data/Q.xlsx', header=1)
    LR = pd.read_excel('data/R.xlsx', header=1)
    LS = pd.read_excel('data/S.xlsx', header=1)
    LT = pd.read_excel('data/T.xlsx', header=1)
    LU = pd.read_excel('data/U.xlsx', header=1)
    LV = pd.read_excel('data/V.xlsx', header=1)
    LW = pd.read_excel('data/W.xlsx', header=1)
    LX = pd.read_excel('data/X.xlsx', header=1)
    LY = pd.read_excel('data/Y.xlsx', header=1)
    LZ = pd.read_excel('data/Z.xlsx', header=1)


    #LA.plot(title='A')
    #LB.plot(title='B')
    #LC.plot(title='C')
    #LD.plot(title='D')
    #LI.plot(title='I')
    #LJ.plot(title='J')
    #LZ.plot(title='Z')


   # plt.show() #plt.show()要放在程序的最后，因为它会阻止程序向下运行

    #print("rest' shape:", rest.shape)
    #print("x' shape:", vert.shape)
    #print("y' shape:", hori.shape)
    #print("z' shape:", circ.shape)
    #print("circle' shape:", shak.shape)

    # X is the array of features to train the model on
    # y is the array of labels
    X = np.vstack([
        LA.to_numpy(),
        LB.to_numpy(),
        LC.to_numpy(),
        LD.to_numpy(),
        LE.to_numpy(),
        LF.to_numpy(),
        LG.to_numpy(),
        LH.to_numpy(),
        LI.to_numpy(),
        LJ.to_numpy(),
        LK.to_numpy(),
        LL.to_numpy(),
        LM.to_numpy(),
        LN.to_numpy(),
        LO.to_numpy(),
        LP.to_numpy(),
        LQ.to_numpy(),
        LR.to_numpy(),
        LS.to_numpy(),
        LT.to_numpy(),
        LU.to_numpy(),
        LV.to_numpy(),
        LW.to_numpy(),
        LX.to_numpy(),
        LY.to_numpy(),
        LZ.to_numpy()
    ])

    y = np.concatenate([
        0 * np.ones(len(LA)),
        1 * np.ones(len(LB)),
        2 * np.ones(len(LC)),
        3 * np.ones(len(LD)),
        4 * np.ones(len(LE)),
        5 * np.ones(len(LF)),
        6 * np.ones(len(LG)),
        7 * np.ones(len(LH)),
        8 * np.ones(len(LI)),
        9 * np.ones(len(LJ)),
        10 * np.ones(len(LK)),
        11 * np.ones(len(LL)),
        12 * np.ones(len(LM)),
        13 * np.ones(len(LN)),
        14 * np.ones(len(LO)),
        15 * np.ones(len(LP)),
        16 * np.ones(len(LQ)),
        17 * np.ones(len(LR)),
        18 * np.ones(len(LS)),
        19 * np.ones(len(LT)),
        20 * np.ones(len(LU)),
        21 * np.ones(len(LV)),
        22 * np.ones(len(LW)),
        23 * np.ones(len(LX)),
        24 * np.ones(len(LY)),
        25 * np.ones(len(LZ))
    ])

    print("X.shape", X.shape)
    print("y.shape", y.shape)


    SAMPLING_RATE = 104
    # this is in milliseconds
    WINDOW_DURATION = 300

   # window = Window(length=SAMPLING_RATE * WINDOW_DURATION // 1000, shift=WINDOW_DURATION // 20)
    window = Window(length=50, shift=15)

    # X_windows holds the input arranged in windows
    # features holds the extracted features for each window (min/max/mean/std...)
    # y_windows holds the most frequent label inside each window
    X_windows, features, y_windows = window.fit_transform(X, y)

    print('X_windows.shape', X_windows.shape)
    print('features.shape', features.shape)
    print('y_windows.shape', y_windows.shape)
    np.set_printoptions(threshold=sys.maxsize)
   # with open('X_windows.txt','w') as f1:
   #     f1.write(str(X_windows))

    # use half data for training, half for testing
    X_train, X_test, y_train, y_test = train_test_split(features, y_windows, test_size=0.5, random_state=0)
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0).fit(X_train, y_train)
    #clf = DecisionTreeClassifier(max_depth=20, random_state=0).fit(X_train, y_train)
    #print('X_train.shape', X_train.shape)
    print('y_windows.shape', y_windows.shape)
    print('len of x=0 in y_windows',len([x for x in y_windows if x==0]))


    #with open('X_train.txt', 'w') as f1: f1.write(str(X_train))
   # plot_confusion_matrix(clf, X_test, y_test, normalize='none', display_labels=['rest', 'vert', 'hori', 'circ', 'shak'])
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test,
    display_labels=['A', 'B', 'C', 'D', 'E',
                    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                    'V', 'W', 'X', 'Y', 'Z'])
    plt.show()

  #window.transform

# write the window.h and classifier.h to be included in Arduino, the second line in Classifier.h should be deleted
 #   with open("Window2.h", "w") as file:
 #        file.write(window.port())
 #   with open("Classifier2.h", "w") as file:
 #        file.write(port(clf, classname="Classifier", classmap={0: "A", 1: " B  ", 2: " C  ", 3: " D  ", 4: "E",
 #                                                               5: "F", 6: " G  ", 7: " H  ", 8: " I  ", 9: "J",
 #                                                               10: "K", 11: " L  ", 12: " M  ", 13: " N  ", 14: "O",
 #                                                               15: "P", 16: " Q  ", 17: " R  ", 18: " S  ", 19: "T",
 #                                                               20: "U", 21: " V ", 22: " W  ", 23: " X  ", 24: "Y",
 #                                                               25: "Z" }))
 # when meet



