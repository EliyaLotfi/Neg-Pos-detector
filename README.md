# Neg-Pos-detector
Detecting the lips' poses by machine learning algorithms for figuring out how he or she is. Smiling :) or not :( 

Actually this algorithm resize the picture into their lips by using MTCNN.
It uses the coordinates of the lips , nose and the parameters of box.
then for making it an array, I reshaped that produced picture in (5,5) by PCA , flatten method and a little art of creativity of combining them together. it helpes the spead of processing. Maybe some of you think by this action I will
lose a lot of information. that's true! but if I want to raise that my system won't handle that :)
However the accuracy of it is 71%! Give me some break!
and in the end it splited the data and train our algorithm. I chose RandomForest becaues of high accuracy and I didn't care about raising processig because
it won't bother my algorithm :)
