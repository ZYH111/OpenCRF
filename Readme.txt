1. implementation
OpenCRF.exe [-est\-estc] -niter [maximum_iteration] -gradientstep [gradient step] -trainfile [train file name] -dstmodel [model file to save] srcmodel [model file to load, only for -estc mode].
----------
example:
OpenCRF.exe -est -niter 100 -gradientstep 0.001 -trainfile example.txt -dstmodel model.txt
OpenCRF.exe -estc -niter 100 -gradientstep 0.001 -trainfile example.txt -dstmodel model_1.txt -srcmodel model.txt
----------
-est : Estimation mode. OpenCRF will read the train file and train the model from the first iteration.
-estc: Estimation continue mode. OpenCRF will read the train file and construct the model according to srcmodel file. Based on the model read, continue the training process.

2. data preparation
First n line describes n variable nodes and their attribute factors in the factor graph mode. 
A line starts with character '?' or '+', where '?' indicates the label of this node is UNKNOWN, while '+' implies a known label. The following character is the label of this node. Note that even if we assume the label is unknown for the model, you can still give the label here, if it is actually available. The model will ignore the label of '?' sample during the training process, but it can be used in the evaluation. Separated by space, the rest tokens in a line are organized as "[feature name]:[feature value]", depicting the attrubutes of the node. The feature value can be a real number, and the feature name should be a string without space. 
-----------
example:
+1 feature_cnt_ratio_1:1 paper_cover_ratio_11:1 coauthor_paper_cnt_6:1 conference_cover_ratio_11:1
**This line depicts a node with a known label '1'. Four features are given: "feature_cnt_ratio_1", "paper_cover_ratio_11", "coauthor_paper_cnt_6" and "conference_cover_ratio_11". The values of all these four features are 1, while the other features appearing in other lines but not this line, has the value 0. 
-----------
The following m lines establish the factors between nodes. The factors can only be established in two nodes. A line started by "#edge", followed by two positive integers i and j, indicates a factor between node i and node j. (Node described in the i-th line of this file is node i) At the end of this line is a string, indicating the name of this factor. Factors with the same name share the same form of function g(yi, yj). 
-----------
example:
#edge 143 4289 same-advisor
**Nodes in the 143-rd line and the 4289-th line are correlated with a "same-advisor" factor. 
-----------

3. output
The accuracy, precision, recall, and F1-score are calculated in each iteration, over both the sets of All nodes and Unlabeled nodes. Note that the result is only valid for binary classification. OpenCRF will take the label of the first line as "positive" label. 
"pred.txt" gives the prediction result of all the nodes (including known nodes). The result is presented by the label id. 0 represents the first kind of label detected in the train file, 1 represents the second kind of label encountered in the file, 2 for the third kind of label, etc. 
"marginal.txt" gives the marginal probability of each label for every node. Each line, the marginal probability is presented in the order of label id (p(y=label0), p(y=labe1), p(y=label2)....) 





