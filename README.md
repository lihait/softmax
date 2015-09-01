# Softmax
Custom MLR python 2 implementaion

## Development

I developed this when I was looking for python softmax regression implementations are found none that illuminated the internals of the algorithm. I decided to tune my own based off other C++ implmentations I found
Its not perfect, but I've moved on to other algorithms

Useful for binary classification with regularized data

The files I used are the Wisconson Breast Cancer dataset

## Usage

Various linear algebra methods are implemented by hand but feel free to get around them with numpy
```
def vector_to_matrix(vec):

    rows = len(vec[0])
    cols = len(vec)
    mat = np.zeros(shape=(rows, cols))
    for i in xrange(rows):
        for j in xrange(cols):
            mat[i][j] = vec[j][i]

    return mat
```
Becomes 
```
def vector_to_matrix(vec):
    return np.mat(vec)
```

Methods like these can be eliminated, but occasionally the numpy matrix type can be useful. 
Such as if you need to do lots of matrix exponentiation and multiplication. The ndarray methods for this are more verbose. 


## Support

Please [open an issue](https://github.com/neale/softmax/issues/new) for support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/neale/softmax/compare/).
