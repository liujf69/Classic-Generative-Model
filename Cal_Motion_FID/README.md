# Calculate Motion FID
```
1. 利用InceptionV3获取source data和generated data的特征；
2. 利用InceptionV3输出的特征，计算均值和标准差；
3. 利用两种data的均值和标准差计算FID；
```
# Run
```python
python gen_test_data.py # randomly generate source data and generated data
python gan_fid.py # Calculate FID
```
