# T2FGAN-pytorch
 pytorch version of T2FGAN

## Using the code
1. Please download the trained model for generating underwater images by using the [link](https://1drv.ms/u/s!AoUeNbw3LPG5axwi7r-AyyXSonc?e=DbT485). 
2. Please download the pretrained model of classifier by using the [link](https://drive.google.com/open?id=1XvG8tjBf8prGOte9d5KIiQ-6CUpCWHR1).
3. To train, run adv_train200324.py by using the command line below:
```python
    python3 adv_train200324 --MGpath "file path in step1" --Cpath "file path in step 2"
```
