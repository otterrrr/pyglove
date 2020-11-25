# pyglove

Pure python module of the original standford **GloVe** word embeddings (https://github.com/stanfordnlp/GloVe)

### Motivation

Especially for small-sized corpus, I need a testbed to revise the existing implementation for my own corpus. Computation only on python wasn't that burdensome to my dataset so **pyglove** make it possible for me to extend the existing implementation and see the feedback from the result quickly

In my viewpoint, it was also beneficial for me to implement from the scratch and the actual python code is quite short to understand. Thus, it would be helpful to those trying to know how GloVe works

### Example

```python
import pyglove

sentences = [
    ['english', 'is', 'language'],
    ['korean', 'is', 'language'],
    ['apple', 'is', 'fruit'],
    ['orange', 'is', 'fruit']
]
glove = pyglove.Glove(sentences, 5)
glove.fit(num_iteration=10, verbose=True)
# training parameters = {'verbose': True, 'num_iteration': 10, 'force_initialize': False, 'x_max': 100, 'self': <pyglove.Glove object at 0x0000018D7634E8D0>, 'num_procs': 8, 'learning_rate': 0.05, 'alpha': 0.75}
# iteration # 0 ... loss = 0.000157
# iteration # 1 ... loss = 0.000136
# iteration # 2 ... loss = 0.000121
# iteration # 3 ... loss = 0.000109
# iteration # 4 ... loss = 0.000099
# iteration # 5 ... loss = 0.000091
# iteration # 6 ... loss = 0.000084
# iteration # 7 ... loss = 0.000078
# iteration # 8 ... loss = 0.000074
# iteration # 9 ... loss = 0.000069
# {'loss': [0.0001574291529957612, 0.0001361206120754304, 0.00012088565389266719, 0.0001085654500573538, 9.887222186502342e-05, 9.134157011817536e-05, 8.411223090092884e-05, 7.821567358495936e-05, 7.397871627005785e-05, 6.903190417689806e-05]}
glove.word_vector
# array([[ 0.03283078, -0.09509491,  0.01144493, -0.0792147 , -0.00604362],
#       [-0.0914974 , -0.00968328,  0.0106788 ,  0.07975675,  0.06333399],
#       [-0.12997769,  0.01405516,  0.00665576, -0.12605855,  0.10085336],
#       [-0.05772575, -0.0987888 ,  0.04216925,  0.03932409, -0.11117414],
#       [ 0.05258524,  0.12941625,  0.00424711,  0.14634097,  0.1428281 ],
#       [ 0.04981236,  0.12080045, -0.00747386,  0.1580294 ,  0.16541023],
#       [-0.11051757, -0.00053117,  0.02030614,  0.03771172,  0.03350186]]#)
glove.word_to_wid
# {'is': 0, 'fruit': 1, 'language': 2, 'apple': 3, 'english': 4, 'korean': 5, 'orange': 6}
glove.wid_to_word
# {0: 'is', 1: 'fruit', 2: 'language', 3: 'apple', 4: 'english', 5: 'korean', 6: 'orange'}
glove.word_vector[glove.word_to_wid['language']]
# array([-0.12997769,  0.01405516,  0.00665576, -0.12605855,  0.10085336])
```

### Use case

#### Stock embeddings of korean stock markets (after processing TSNE dimension reduction)

![stock embeddings](https://user-images.githubusercontent.com/1368591/100223133-6bfb6180-2f5e-11eb-883e-a8e1951decf5.png)

### Limitation

Features the original stanford **Glove** supports but **pyglove** doesn't
* memory-bound execution
  * The original **GloVe** implementation has memory-bound execution logic. In other words, it flushes out intermediate result over memory threshold
  * However, **pyglove** works assuming that system memory is sufficient to contain all the corpus and intermediate results
  * Hence, please make sure your system can provide enough memory
* some parameters fixed in function body
  * cooccurence_count.symmetric (fixed as True)
  * glove.word_vector.model (fixed as 3)
    1. result word_vectors consisting of target and context vectors including biases
    1. result word_vectors consisting of target vectors without biases
    1. result word_vectors consisting of target and context vectors without biases

Besides, **pyglove** is much slower than the original **Glove**
* parallelization using python.multiprocessing boost its learning speed but...
  * x10 slower so it's another reason this's appicable only to small-sized corpus

### Future items

* Performance improvement?
* Enhancement in counting cooccurrence in ranged window
