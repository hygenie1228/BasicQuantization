Basic Quantizaiton Framework
======================
  
This code is the basic framework for DNN quantization. 

## Directory  

```
Basic_Quantization
|-- assets
|-- nets
|-- quantize
|-- |-- func
|-- |-- layers
|-- utils
|-- base.py
|-- main.py
|-- train_fp.py
```

* `nets` contains various networks which quantization is applied.
* `quantize` contains several functions for quantization. Also manages the quantization schedule.
* `utils` contains transform, logging, visualization, etc.
* `base.py` includes network train & test code.
* `main.py` is the code to apply quantization.
* `train_fp.py` is the code to train full precision model.


## Run  
  
### Train full precision model
```
cd ${ROOT}
python ./train_fp.py --gpu 0
```

### Quantization
```
cd ${ROOT}
python ./main.py --gpu 0
```