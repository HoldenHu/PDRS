# /home/wing.nus/.pyenv/versions/3.7.0/bin/python3

# Baseline
## ItemPop
nohup python3 -u main_itempop.py  > out/course/log/course.itempop.warm.random-multik.log 2>&1 &
## ItemKNN
nohup python3 -u main_itemknn.py  > out/course/log/itemknn.warm.random-multik.log 2>&1 &
nohup python3 -u main_itemknn.py  > out/movie/log/itemknn.warm.random.log 2>&1 &
## MF
nohup python3 -u main_mf.py  > out/course/log/mf.cold.random.log 2>&1 &
nohup python3 -u main_mf.py  > out/movie/log/mf.cold.random.log 2>&1 &
## MLP
nohup ~/.pyenv/versions/3.7.0/bin/python -u main_mlp.py > log/mlp-bz.log 2>&1 &

# Model
## KMLP
nohup python3 -u main_kmlp.py  > out/course/log/kmlp.warm.random.log 2>&1 &


# Test
nohup /home/wing.nus/.pyenv/versions/3.7.0/bin/python3 -u temp.py  > out.log 2>&1 &