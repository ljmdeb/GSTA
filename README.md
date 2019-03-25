# GSTA
## Replacing a Black-box model by a Global Single Tree Approximation

Laurent Deborde
2019 March the 24th

### Content : 
* discussion of the subject (here and in the .doc and .odt documents). 
* 3 Jupyter notebooks were the approach is applied on public toy datasets : 1 classification (Breast Cancer), 2 regression (Wine and Crime datasets). On all datasets the "black box models" are Random Forest and Gradient Boosting. In the first 2 cases both gaussian and determinist (Munge-like) supplemental data creation is exemplified. In the third case PCA is exemplified. 
All is usable under Creative Commons License (in the unlikely event one would like to use it). 

### Discussion : 
In supervised learning applications, explainability is frequently considered important by end users. They often are willing to trade efficiency for it. 
There even are cases when, for regulatory, practical or commercial reasons, the use of the best machine learning models, if they be black boxes, is forbidden. Only single trees or similar simple models are deemed acceptable then. 
Do we have in those cases to revert to ordinary trees, cart-built on the initial training set? 
Not necessarily so. Methods have been developed to explain complex models. Some of them consist of approximating globally the black box with a single tree. 
Why not use that proxy tree as a replacement model when black boxes are forbidden, then?

If I'm not mistaken, those ideas can be applied quite simply, with the usual tools of machine learning (e.g. Python and Scikit-learn). I tried them on a couple of well-known datasets and found the results to be quite better than the usual Cart Trees (see code). Those tries haven't been extensive though, and I hadn't the opportunity to discuss that with other users. So I still wonder if such model replacements are frequently practiced, on what kind of problems it can be applied succesfully, or even whether the satisfying performances obtained here are just the result of chance/mistake. I would appreciate your feedback. 

The general idea is the following : the performance of a tree built by the Cart algorithm generally increase with the number of data on which it is trained. More training data would let us build a better tree. We'll use the black box model to do just that : generate new labeled data. More precisely, let's take some unlabelled supplemental data (either from your real-life problem or generated  by some procedure based on the existing training set), and use the black box model as an oracle to predict the label of those supplemental data. Then we will train a single tree on the resulting labelled data. Please note that there is no leakage from the test set since is not used in the process. 


The procedure used here has been quite simple: 
fit a Tree with Cart on the training set (choosing parameters with cross-validation), then test : observe mediocre results
fit a more complex model (typically Gradient Boosting and/or Random Forest, with default parameters or cross-validation), then test: observe better result.
Generate many new unlabeled data (drawing the new data at random in a multivariate gaussian process, with mean and variance/covariance estimated on the training set) [alternatively, generate the data by deterministic procedure; and in the case of many dimensions, use ACP before generating new data] 
Compute label predicted by the complex model on the newly generated data.
Fit a tree (with Cart) on those new data (choosing depth with cross-validation; or limiting ex ante the tree depth to 7 or 8 to insure ease of interpretation by end user)
Compute prediction of this new « enhanced » tree on the (original) test set: observe quite good results, typically somewhere in between the original single tree and the complex model
(in real world application:) keep that « enhanced » tree as model

Articles mentioned as references use smart strategies to avoid creating too many points: they typically draw points as needed, as they build the tree step by step (albeit frequently the covariance of variables is ignored). On the opposite, here, for the sake of simplicity, I mainly use a (simplistic!) « brute force » strategy where all new points are drawn first and then the entire tree is built (although using the covariance structure of the training set).

A drawback is that a quite large number of virtual data is needed to properly represent (or map) the space on which the complex model is to be replicated. It leads to lengthy computations when cart-fitting a tree to those data. It's an effect of the « curse of dimensionality. » To give an example, let's imagine that we want to deterministically build new data by taking all combinations of a mere 2 possible values for each variable (for instance max and min of values observed on the real data set, or average +/- standard deviation): for n variables, that's 2^n data points. Way too much for my computer in the quite standard case of some 30 variables. But using much less points will result in mapping too small a part of the data space. 

In professional context, I've used a PCA to reduce dimensionality and randomly draw the supplemental data in the reduced space, to improve the number of data point / number of variables ratio. An additional benefit is that PCA axis being orthogonal, one doesn't have to bother about covariance in the randomly drawn set. It did work in that case. An exemple of this procedure is included here (on the “ViolentCrime” dataset).

Finally, rather than drawing points at random, deterministic approach can be used : for instance the combination of average +/- standard deviation described above (only after ACP, because it doesn't take into account covariances) ; or (inspired by [6]) by taking the exact average of each pair of « real » point in the training set (which produce a total of n^2 points if n is the number of points in the original training set). Variants with more data by pair of real points could produce more points.


My intuition is that this simple way of generating global single tree approximations are appropriate when the underlying problem is not too complex (more specifically, we are able this way to do much better than a initial tree of low depth (for instance the classification case here, Breast Cancer). Then a reasonable number of artificial points let us build a more complex, and more efficient, tree. If the underlying problem is too complex (e.g.we would need a very deep tree to reach acceptable levels of accuracy), the needed number of artificial points gets high and the computing time increases unacceptably. Obviously quality of tree fit to generated additionnal data predict quality of prediction. Additionaly, in the 3 exemples here I observe that single tree approximation fit more closely random forest predicted label than they fit gradient boosting predicted label. Have you experienced the same ? What of other complex models (NN, SVM) ?

##### Thanks in advance for your feedback. 


### Some references: 

The approximation techniques I refer to may be called after [1] « global reverse engineering ». See [1] p. 93:20 sq. for references (some of which are reproduced here) and classification and description of XAI techniques. [2] is an early one. Some of those approaches (using the same source, we may call them « Global Random Single Tree Approximations »), use randomly artificially generated data, labelled by the Black box to be replicated. The explanatory tree is fitted to those data. The idea stem from [3]. [4] applies it from to a real-life medical case where the complexity of the model is limited by the ability to question the patient (and obtain values for the model). More recently see [5]. Additional data, rather than being randomly drawn, can be built following a variety of procedure, such as interpolation between points of the original training set, imitating the “MUNGE” method that [6] introduced in a slightly different context.  


[1] Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Franco Turini, Fosca Giannotti, and Dino Pedreschi. 2018. A Survey of Methods for Explaining Black Box Models. ACM Comput. Surv. 51, 5, Article 93 (August 2018).  

[2] Mark Craven and JudeW. Shavlik. 1996. Extracting tree-structured representations of trained networks. In Proceedings of the Conference on Advances in Neural Information Processing Systems. 24–30.

[3] Domingos, Pedro. "Knowledge Acquisition from Examples Via Multiple Models." In Proceedings of the Fourteenth International Conference on Machine Learning. 1997. 

[4] Robert D Gibbons, Giles Hooker, Matthew D Finkelman, David J Weiss, Paul A Pilkonis, Ellen Frank, Tara Moore, and David J Kupfer. The computerized adaptive diagnostic test for major depressive disorder (cad-mdd): a screening tool for depression. The Journal of clinical psychiatry, 74(7):1{478, 2013.

[5] Bastani, Osbert & Kim, Carolyn & Bastani, Hamsa. (2017). Interpreting Blackbox Models via Model Extraction. 

[6] C. Bucilua, R. Caruana, and A. Niculescu-Mizil, “Model compression,” in Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD ’06, 2006, pp. 535–541.

![https://zenodo.org/badge/177329765.svg]


