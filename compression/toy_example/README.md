# Usage

- simply execute ```python test.py```
- [toy_model.py](toy_model.py) shows the model structure
- [train.py](train.py) and [config.py](config.py) are used for training.
- Parameters
    * ```--image-path``` uncompressed input image path
    * ```--output-path``` compressed output image path (png file)
    * ```--string-path``` compressed file path (binary file)
    * ```--param-path``` model parameters path
    * ```--device``` device for pytorch (cpu or cuda or cuda:N)