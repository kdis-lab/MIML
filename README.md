# MIML: A Java library for MIML learning
* [What is MIML library?](https://github.com/kdis-lab/MIML/blob/master/README.md#what-is-miml-library)
* [Description of the project](https://github.com/kdis-lab/MIML/blob/master/README.md#description-of-the-project)
* [Getting the library](https://github.com/kdis-lab/MIML/blob/master/README.md#getting-the-library)
* [Tutorials and documentation](https://github.com/kdis-lab/MIML/blob/master/README.md#tutorials-and-documentation)
* [Methods included](https://github.com/kdis-lab/MIML/blob/master/README.md#methods-included)
* [MIML library data format](https://github.com/kdis-lab/MIML/blob/master/README.md#miml-library-data-format)
* [Citation](https://github.com/kdis-lab/MIML/blob/master/README.md#citation)
* [License](https://github.com/kdis-lab/MIML/blob/master/README.md#license)
* [Reporting bugs](https://github.com/kdis-lab/MIML/blob/master/README.md#reporting-bugs)


## What is MIML library?
MIML is a modular Java library whose aim is to ease the development, testing and comparison of classification algorithms for multi-instance multi-label learning (MIML). It includes 32 classificaton algorithms for solving MIML problems. Algorithms are included on three different approaches for solving a MIML problem: transforming the problem to multi-instance, transforming the problem to multi-label problem and solving directly the MIML problem. Besides, it provides holdout and cross-validation procedures, standard metrics for performance evaluation as well as report generation. Algorithms can be executed by means of *xml* configuration files. It is platform-independent, extensible, free and open source. It is based in both [Weka](https://www.cs.waikato.ac.nz/ml/weka/) and [Mulan](https://github.com/tsoumakas/mulan) frameworks.

## Description of the project
In this Github project you can find the following directories:
* [*main folder*](https://github.com/kdis-lab/MIML/tree/master/). Contains all the project as wekll as a copy of the [user's manual](https://github.com/kdis-lab/MIML/blob/master/MIML-UserManual.pdf).
* [*apidoc*](https://github.com/kdis-lab/MIML/tree/master/apidoc). Contains the javadoc documentation of the API in html format.
* [*configurations*](https://github.com/kdis-lab/MIML/tree/master/configurations). Contains an example of *xml* configuration file for each of the algorithms included in the library.
* [*data*](https://github.com/kdis-lab/MIML/tree/master/data). Contains some examples of datasets. Particularly, birds dataset in included with the distribution of the librery.
* [*dist*](https://github.com/kdis-lab/MIML/tree/master/dist). Contains the distribution files.
* [*results*](https://github.com/kdis-lab/MIML/tree/master/results). Contains the file reports generated by each example in configurations folder.
* [*src*](https://github.com/kdis-lab/MIML/tree/master/src). Contains de source code of the distribution.
* [*userManual*](https://github.com/kdis-lab/MIML/tree/master/userManual). Contains the user's manual.

## Getting the library

Before downloading the library, it is necessary to have Java Development Kit version 8 or higher installed. 

The project can be download using maven tool, java project or from a compiled version (jar file). Detailed information about getting and running the library can be consulted in [user's manual](https://github.com/kdis-lab/MIML/blob/master/userManual/MIML-UserManual.pdf).


## Tutorials and documentation
[MIML user's manual](https://github.com/kdis-lab/MIML/blob/master/userManual/MIML-UserManual.pdf) can be found in the userManual folder and includes:
* Detailed steps for getting and running the library.
* A description of MIML library architecture.
* Examples for managing MIML data.
* Examples for running each classification MIML algorithm included in the library.
* Examples for developing new classification MIML algorithm in the library.

[MIML API](https://github.com/kdis-lab/MIML/tree/master/apidoc) can be found in apidoc folder.

Moreover, the source code includes the tutorial folder where there are java examples of different functionalities of MIML library. Finally, in configuration folder, it can be found configuration files to run any library algorithm. These files can be used as a basis to modify any configuration parameter in the experimentation of these algorithms.


## Methods included

MIML includes a set of algorithms according to the following three approaches:

1 **MIML to MI approach**. This aprroach transforms the MIML problem to MI and then uses any MI algorithm to solve the problem. To this end, a transformation at labels level is applied. Currently two label transformations have been included: BR and LP.
  * BR transformation.
    * CitationkNN, MDD, MIDD, MIBoost, MILR, MIOptimalBall, MIRI, MISMO, MISVM, MITI, MIWrapper, SimpleMI.
  * LP transformation.
    * CitationkNN, MIWrapper, SimpleMI.
      * Note that if MDD, MIDD, MIBoost, MILR, MIOptimalBall, MIRI, MISMO, MISVM or MITI are run with LP transformation the following execution error is raised *Cannot handle multi-valued nominalclass!*. This is due to the philosophy of the LP method which obtains one multi-class dataset andthese algorithms are only able to deal with binary class data. 

2 **MIML to ML approach**. This approach transforms the MIML problem to ML and then uses any ML algorithm to solve the problem. To this end, a transformation at bag level is performed. Currently, three bag transformations have been included: arithmetic, geometric and min-max. The following are the ML algorithms considered by the library.
  * BR, LP, RPC, CLR, BRkNN, DMLkNN, IBLR, MLkNN, HOMER, RAkEL, PS, EPS, CC, ECC, MLStacking.

3 **MIML solving without transformation**. Currently the following algorithms have been included.
  * Bagging, MIMLkNN
  
  
## MIML library data format

The format of data is based on the Weka's format for MI learning and on the Mulan's format for ML learning. Concretely, each data set is represented by two files: 

* An *xml* file based on [Mulan's format](http://mulan.sourceforge.net/format.html) containing the description of labels. Its aim is to identify those attributes in the *arff* file representing labels. Note that the class attributes do not need to be the last attributes in the *arff* file and also their order in both at the *arff* and the *xml* file does not matter. A hierarchy of labels can be represented by nesting the label tags. The following is an example of *xml* file with 4 labels:

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
    
* An *arff* (*Attribute-Relation File Format*) file based on [Weka's multi-instance format](https://weka.wikispaces.com/Multi-instance+classification) containing the data. This file is organized in two parts: header and data. 
  * *Header*: it contains the name of the relation and a list with the attributes and their data types.        
     * The first line of the file contains the *@relation &lt;relation-name>* sentence, which defines the name of the dataset. This is a string and it must be quoted if the relation-name includes spaces.
     * Next, on the first level, there are defined only two attributes and the attributes corresponding to the labels.            
       * *&lt;bag-id>*. Nominal attribute. Unique bag identifier for each bag.
       * *&lt;bag>*. Relational attribute. Contains instances attributes.                
       * *&lt;labels>*. One binary attribute for each label (nominal with 0 or 1 value).                   
     
    Attributes are defined with *@attribute &lt;attribute-name> &lt;data-type>* sentences. There is a line per attribute.    
       * Numeric attributes are specified by *numeric*.
       * In case of nominal attributes, the list of values must be specified with curly brackets and separated by commas: *{value_1, value_2, ..., value_N}*.
            
   * *Data*: it begins with *@data* and describes each example (*bag*) in a line. The order of attributes in each line must be the same in which they were defined in the previous header. Each attribute value is separated by comma (,) and all lines must have the same number of attributes. Decimal position is marked with a dot (.). The data of the relational attribute is surrounded by single (') or double (") quotes, Weka recognizes both formats, and the single instances inside the bag are separated by line-feeds ('\n'). 
        
    Next, an example of *arff* file is shown. In the example, each bag contains instances described by 3 numeric attributes and there are 4 labels. The dataset has two bags, the first one with 3 instances and the second one with 2 instances.

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

## Reporting bugs
Feel free to open an [issue](https://github.com/kdis-lab/MIML/issues) at Github if anything is not working as expected. Merge request are also encouraged, it will be carfully reviewed and merged if everything is all right.
