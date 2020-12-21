# Usage

- simply execute ```python test.py```
- [toy_model.py](toy_model.py) shows the model structure.
- you do not need to train the model. 
- default parameters is what I use to test the model.
- model parameters download: [link](https://drive.google.com/drive/folders/1nq9v937jLySbK6wINHYQ3PdZ08hBEUn5?usp=sharing)
- The only required extra package needs to be installed manually: [Tutorial](https://interdigitalinc.github.io/CompressAI/tutorial_installation.html)

- Parameters
    * ```--image-path``` uncompressed input image path
    * ```--output-path``` compressed output image path (png file)
    * ```--string-path``` compressed file path (binary file)
    * ```--param-path``` model parameters path
    * ```--device``` device for pytorch (cpu or cuda or cuda:N)