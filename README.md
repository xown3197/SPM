# Spatial Pyramid Matching (Bag of Visual Word)

## SPM main
```
his = build(train_imgs, codebook) 

x_data = []
for data in his:
  py = pyramid(data, 2)
  x_data.append(py)

his_test = build(test_imgs, codebook)

x_test = []
for data in his_test:
  py = pyramid(data, 2)
  x_test.append(py)

x_train = np.asarray(x_data, dtype=np.float32)
x_test = np.asarray(x_test, dtype=np.float32)

x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_train, y_train)

result = HI_SVM(x_train_t, x_test_t, y_train_t)
#result = LSVM(x_train_t, x_test_t, y_train_t)

print(classification_report(y_test_t, result))
```
> DataSet : Caltech 101
> Scaler : Srandard 적용
> 각 train 30장을 75:25 비율로 split해서 테스트를 진행


|Level|CodeBook_Size|Kernel|Accuracy|
|:---:|:-----------:|:----:|:------:|
| 0   |  256 | Linear|0.31|
| 0   |  256 | HI    |0.41|
| 1 | 256 | Linear |0.44| 
| 1 | 256 | HI | 0.49|
| 2 | 256 | Linear| |
| 2 | 256 | HI | |
| 0   |  512 | Linear|0.31|
| 0   |  512 | HI    |0.41|
| 1 | 512 | Linear |0.44| 
| 1 | 512 | HI | 0.49|
| 2 | 512 | Linear| |
| 2 | 512 | HI | |
| 0   |  1024 | Linear|0.31|
| 0   |  1024 | HI    |0.41|
| 1 | 1024 | Linear |0.44| 
| 1 | 1024 | HI | 0.49|
| 2 | 1024 | Linear| |
| 2 | 1024 | HI | |
| 0   |  2048 | Linear|0.31|
| 0   |  2048 | HI    |0.41|
| 1 | 2048 | Linear |0.44| 
| 1 | 2048 | HI | 0.49|
| 2 | 2048 | Linear| |
| 2 | 2048 | HI | |


> 코드북 사이즈를 늘리는 방식으로 튜닝을 결정하게된 

[출처](http://www.robots.ox.ac.uk/)

## Histogram Intersection(HI) Kernel

```
def HistogramIntersection(X, Y):
    x = X.shape[0]
    y = Y.shape[0]

    result = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            temp = np.sum(np.minimum(X[i], Y[j]))
            result[i][j] = temp
    return result
```
## HI Kermel SVM

```
def HI_SVM(x_train, x_test, y_train):

  gramMatrix = HistogramIntersection(x_train, x_train)
  clf = SVC(kernel='precomputed')
  clf.fit(gramMatrix, y_train)

  predictMatrix = HistogramIntersection(x_test, x_train)
  SVMResults = clf.predict(predictMatrix)

  return SVMResults
```

## Make Hitogram + Histogram Recombination
> 기술자를 ((이미지 높이 h) / Step size(dense sift), (이미지 높이 h) / Step size(dense sift), 128)로 resize
> resize된 기술자를 16등분
> 16등분된 Histogram을 Pyramid 레벨에 맞게 재조합
```
def build(imgs, codebook):
  data = []

  for img in tqdm(imgs):
    _, des = dense_sift_each(img)
    des = np.asarray(des)
    des = np.resize(des, (32, 32, 128))
    des = cutted(des, 2)
    des = np.asarray(des)
    l, _, _, _ = des.shape

    hist = []

    for i in range(0,l):
      tmp = np.resize(des[i], (64 ,128))
      his = histogram(tmp, codebook)
      hist.append(his)

    data.append(hist)

  return data
```
> 16등분을 위한 함수 (이전 kaggle Discussion에서 공개한 피라미드)
```
def cutted(src, level=2):
  h_end, w_end, _ = src.shape
  cutted_img = []
  w_start = 0
  h_start = 0
  w = w_end // (2**level)
  h = h_end // (2**level)

  if level != 0:
    for j in range(2 ** level):
      for i in range(2 ** level):
        img = src[h_start:h_start + h, w_start:w_start + w]
        cutted_img.append(img)
        w_start += h
        if (w_start+h == w_end+w):
          w_start = 0
          h_start += h
  else: cutted_img = src

  return cutted_img
```

```
def pyramid(his, level):
  his1 = np.array([[his[0] + his[1] + his[4] + his[5]], [his[2] + his[3] + his[6] + his[7]], [his[8] + his[9] + his[12] + his[13]], [his[10] + his[11] + his[14] + his[15]]])
  his0 = his1[0] + his1[1] + his1[2] + his1[3]
  his0 = np.ravel(his0, order='C')
  his1 = np.ravel(his1, order='C')
  his2 = his
  his2 = np.ravel(his2, order='C')

  if level == 0:
    return np.asarray(his0)

  py = np.hstack((his0 * 0.25, his1 * 0.25))

  if level == 1:
    return py

  if level == 2:
    return np.hstack((py, his2 * 0.5))
```
