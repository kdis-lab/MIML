# MIML: A Java library for MIML learning
* [What is MIML library?](https://github.com/kdis-lab/MIML/blob/master/README.md#what-is-miml-library)
* [Installation, tutorials and documentation](https://github.com/kdis-lab/MIML/blob/master/README.md#installation-tutorials-and-documentation)
* [Methods included](https://github.com/kdis-lab/MIML/blob/master/README.md#methods-included)
* [References](https://github.com/kdis-lab/MIML/blob/master/README.md#references)
* [Citation](https://github.com/kdis-lab/MIML/blob/master/README.md#citation)
* [License](https://github.com/kdis-lab/MIML/blob/master/README.md#license)

## What is MIML library?
MIML is a modular Java library whose aim is to ease the development, testing and comparison of classification algorithms for multi-instance multi-label learning (MIML). It includes three different approaches for solving a MIML problem: transforming the problem to multi-instance, transforming the problem to multi-label problem, and solving directly the MIML problem. Besides, it provides holdout and cross-validation procedures, standard metrics for performance evaluation as well as report generation. Algorithms can be executed by means of *xml* configuration files. It is platform-independent, extensible, free and open source.

## Installation, tutorials and documentation
The documentattion can be found in the doc folder and includes:
* Installation guide
* Tutorials
* A guide about ....
* An example about how to add a new classification method to MIML


## Methods included

|  MIML to MI problem  |                     |
|:--------------------:|---------------------|
| Label Transformation | MI Algorithm (Weka) |
|                      | CitationKNN         |
|                      | MDD                 |
|                      | MIDD                |
|                      | MIBoost             |
|                      | MILR                |
|                      | MIOptimalBall       |
|          BR          | MIRI                |
|                      | MISMO               |
|                      | MISVM               |
|                      | MITI                |
|                      | MIWrapper           |
|                      | SimpleMI            |
|                      | CitationKNN         |
|          LP          | MIWrapper           |
|                      | SimpleMI            |


|  MIML to ML problem  |                     |
|:--------------------:|---------------------|
| Bag Transformation | ML Algorithm (Mulan) |
|                      | BR        |
|                      | LP                 |
|                      | RPC                |
|                      | CLR             |
|                      | BRkNN                |
|          Arithmetic            | DMLkNN       |
|          Geometric          | IBLR                |
|          Min-Max            | MLkNN               |
|                      | HOMER               |
|                      | RAkEL               |
|                      | PS           |
|                      | EPS            |
|                      | CC         |
|                    | ECC           |
|                      | MLStacking            |


| MIML method |  |
| ------------- | ------------- |
| Bagging  | Contenido de la celda  |
| MIMLkNN  | Contenido de la celda  |
| MLkNN  | Contenido de la celda  |


## MIML library data format

The format of data is based on the Weka's format for MI learning and on the Mulan's format for ML learning. Concretely, each data set is represented by two files: 

* An *xml* file based on Mulan's format containing the description of labels. Its aim is to identify those attributes in the *arff* file representing labels. Note that the class attributes do not need to be the last attributes in the *arff* file and also their order in both at the *arff* and the *xml* file does not matter. A hierarchy of labels can be represented by nesting the label tags. The following is an example of *xml* file with 4 labels:

```xml
<?xml version="1.0" encoding="utf-8"?>
<labels xmlns="http://mulan.sourceforge.net/labels">
  <label name="label1"></label>
  <label name="label2"></label>
  <label name="label3"></label>
  <label name="label4"></label>
</labels>
```
    
The following is an example of *xml*  file with a hierarchy of labels:
    
```xml
<?xml version="1.0" encoding="utf-8"?>
<labels xmlns="http://mulan.sourceforge.net/labels">
    <label name="sports"> 
        <label name="football"></label>
        <label name="basketball"></label> 
    </label>
    <label name="arts"> 
        <label name="sculpture"></label>
        <label name="photography"></label> 
    </label>
</labels>
```
    
    
* An *arff* (*Attribute-Relation File Format*) file based on Weka's multi-instance format containing the data. This file is organized in two parts: header and data. 
  * *Header*: it contains the name of the relation and a list with the attributes and their data types.        
     The first line of the file contains the *@relation* relation-name sentence, which defines the name of the dataset. This is a string and it must be quoted if the relation-name includes spaces.
            Next, on the first level, there are only two attributes and the attributes corresponding to the labels.
            
                *bag-id*. Nominal attribute. Unique bag identifier for each bag.
                *bag. Relational attribute. Contains instances attributes.                
                *labels*. One binary attribute for each label (nominal with 0 or 1 value).          
            

         Attributes are defined with \textit{@attribute \textless attribute-name\textgreater\textless data-type\textgreater} sentences. There is a line per attribute.
            \begin{itemize}
                \item Numeric attributes are specified by \emph{numeric}.
                \item In case of nominal attributes, the list of values must be specified with curly brackets and separated by commas: \textit{\{value$_1$, value$_2$, ..., value$_N$\}}.
            \end{itemize}
            
     *Data*: it begins with *@data* and describes each example (*bag*) in a line. The order of attributes in each line must be the same in which they were defined in the previous header. Each attribute value is separated by comma (,) and all lines must have the same number of attributes. Decimal position is marked with a dot (.). The data of the relational attribute is surrounded by single (') or double (") quotes, Weka recognizes both formats, and the single instances inside the bag are separated by line-feeds (\n). 
        
    Next, an example of *arff* file is showed. In the example, each bag contains instances described by 3 numeric attributes and there are 4 labels. The dataset has two bags, the first one with 3 instances and the second one with 2 instances.

```       
@relation toy
@attribute id {bag1,bag2}
@attribute bag relational
  @attribute f1 numeric 
  @attribute f2 numeric 
  @attribute f3 numeric 
@end bag
@attribute label1 {0,1}
@attribute label2 {0,1}
@attribute label3 {0,1}
@attribute label4 {0,1}
@data
bag1,"42,-198,-109\n42.9,-191,-142\n3,4,6",1,0,0,1
bag2,"12,-98,10\n42.5,-19,-12",0,1,1,0 
```   

## Citation
This work has been performed by A. Belmonte, A. Zafra and E. Gibaja and is currently in a reviewing process.

## License
MIML library is released under the GNU General Public License [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).

## References

Stuart Andrews, Ioannis Tsochantaridis, and Thomas Hofmann. Support vector machines for multiple-instance learning. In Advances in Neural Information Processing Systems
15, pages 561–568. MIT Press, 2003.

Peter Auer and Ronald Ortner. A boosting approach to multiple instance learning. In 15th European Conference on Machine Learning, pages 63–74. Springer, 2004. LNAI 3201.

Luke Bjerring and Eibe Frank. Beyond trees: Adopting miti to learn rules and ensemble classifiers for multi-instance data. In Proceedings of the Australasian Joint Conference
on Artificial Intelligence. Springer, 2011.

Hendrik Blockeel, David Page, and Ashwin Srinivasan. Multi-instance tree learning. In Proceedings of the International Conference on Machine Learning, pages 57–64. ACM,
2005.

Leo Breiman. Bagging predictors. Machine learning, 24(2):123–140, 1996. 

Weiwei Cheng and Eyke H¨ullermeier. Combining instance-based learning and logistic regression for multilabel classification. Machine Learning, 76(2-3):211–225, 2009.

Thomas G Dietterich, Richard H Lathrop, and Tom´as Lozano-P´erez. Solving the multiple instance problem with axis-parallel rectangles. Artificial intelligence, 89(1-2):31–71, 1997.

Lin Dong. A comparison of multi-instance learning algorithms. Master’s thesis, The University of Waikato, 2006. 

E. T. Frank and X. Xu. Applying propositional learning algorithms to multi-instance data. Technical report, University of Waikato, Department of Computer Science, University of
Waikato, Hamilton, NZ, 06 2003.

Johannes F¨urnkranz, Eyke H¨ullermeier, Eneldo Loza menc´ıa, and Klaus Brinker. Multilabel classification via calibrated label ranking. Machine Learning, 73(2):133 – 153, 2008.

Thomas G¨artner, Peter A Flach, Adam Kowalczyk, and Alex J Smola. Multi-instance kernels. In ICML, volume 2, pages 179–186, 2002.

Eva Gibaja and Sebasti´an Ventura. A tutorial on multilabel learning. ACM Computing Surveys (CSUR), 47(3):52, 2015.

Mark Hall, Eibe Frank, Geoffrey Holmes, Bernhard Pfahringer, Peter Reutemann, and Ian H. Witten. The weka data mining software: an update. ACM SIGKDD explorations
newsletter, 11(1):10–18, 2009.

F. Herrera, S. Ventura., R. Bello, C. Cornelis, A. Zafra, D. S´anchez-Tarrag´o, and S. Vluymans. Multiple Instance Learning. Foundations and Algorithms. Springer, 2016.

Eyke H¨ullermeier, Johannes F¨urnkranz, Weiwei Cheng, and Klaus Brinker. Label ranking by learning pairwise preferences. Artificial Intelligence, 172:1897–1916, 2008.

LAMDA. Lamda learning and mining from data. http://www.lamda.nju.edu.cn/Data.ashx, 2019. Accessed: 2020-07-19.

Oded Maron and Tom´as Lozano-P´erez. A framework for multiple-instance learning. In Proceedings of the 1997 Conference on Advances in Neural Information Processing Systems
10, NIPS ’97, pages 570–576, Cambridge, MA, USA, 1998. MIT Press. ISBN 0-262-10076-2. URL http://dl.acm.org/citation.cfm?id=302528.302753.

Soumya Ray and Mark Craven. Supervised versus multiple instance learning: An empirical comparison. In Proceedings of the 22nd international conference on Machine learning,
pages 697–704, 2005.

J. Read, P. Reutemann, B. Pfahringer, and G. Holmes. Meka: a multi-label/multi-target extension to weka. Journal of Machine Learning Research, 17(1):667–671, 2016.

Jesse Read, Bernhard Pfahringer, and Geoff Holmes. Multi-label classification using ensembles of pruned sets. In Data Mining, 2008. ICDM’08. Eighth IEEE International
Conference on, pages 995–1000. IEEE, 2008.

Jesse Read, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank. Classifier chains for multi-label classification. Machine learning, 85(3):333, 2011.

E. Spyromitros, G. Tsoumakas, and I.Vlahavas. An empirical study of lazy multilabel classification algorithms. In Proc. 5th Hellenic Conference on Artificial Intelligence (SETN
2008), 2008.

G. Tsoumakas, E. Spyromitros-Xioufis, Jozef Vilcek, and I. Vlahavas. Mulan: A java library for multi-label learning. Journal Machine Learning Research, 12:2411–2414, 2011a. ISSN 1532-4435.

Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas. Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008
Workshop on Mining Multidimensional Data (MMD’08), 2008.

Grigorios Tsoumakas, Anastasios Dimou, Eleftherios Spyromitros-Xioufis, V. Mezaris, Ioannis Kompatsiaris, and I. Vlahavas. Correlation-based pruning of stacked binary relevance
models for multi-label learning. In Proceedings of the 1st International Workshop on Learning from Multi-Label Data, pages 101–116, 01 2009a.

Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas. Mining multi-label data. In Data mining and knowledge discovery handbook, pages 667–685. Springer, 2009b.

Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas. Random k-labelsets for multilabel classification. IEEE Transactions on Knowledge and Data Engineering, 23(7):
1079–1089, 2011b.

Jun Wang and Jean-Daniel Zucker. Solving the multiple-instance problem: A lazy learning approach. In Pat Langley, editor, Proceedings of the 17th International Conference on
Machine Learning, ICML ’00, pages 1119–1126, San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc. ISBN 1-55860-707-2. URL http://dl.acm.org/citation.
cfm?id=645529.757771.

Xin Xu and Eibe Frank. Logistic regression and boosting for labeled bags of instances. In Advances in knowledge discovery and data mining, pages 272–281. Springer, 2004.

Z. Younes, F. Abdallah, and T. Denoeux. Multi-label classification algorithm derived from k-nearest neighbor rule with label dependencies. In 2008 16th European Signal Processing Conference, pages 1–5, 2008.

Min-Ling Zhang and Zhi-Hua Zhou. Ml-knn: A lazy learning approach to multi-label learning. Pattern recognition, 40(7):2038–2048, 2007.

M.L. Zhang. A k-nearest neighbor based multi-instance multi-label learning algorithm. In Proceedings of the 22nd International Conference on Tools with Artificial Intelligence,
volume 2, pages 207–212, 2010.
