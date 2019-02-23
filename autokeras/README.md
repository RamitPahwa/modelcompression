`pip install autokeras`

python = 3
`https://www.simonwenkel.com/2018/09/02/autokeras-cifar10_100.html`
Usage

SELECT the Desired Subset for cifar 10 in  `main.py ` and cifar100 in `main_cifar100.py`

run  `python main.py` or `python main_cifar100.py`

chnage the time for running the Neural architecture search in `main.py` line 17

```
# select the time you want to run the search for in hrs 
time_array =[1,2,3,4,5,6,7,8,9,10]
# change index to change time for search
time =  time_array[0]
```