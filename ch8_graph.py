import os
from sklearn import tree
from matplotlib.pyplot import imshow
from IPython.display import Image, display
def graph(dt,X,y):
    tree.export_graphviz(dt.fit(X,y),out_file='tree.dot',filled=True, rounded=True,
                special_characters=True,feature_names = X.columns)
    os.system("dot -Tpng tree.dot -o tree.png")
    display(Image(filename='tree.png')) 
