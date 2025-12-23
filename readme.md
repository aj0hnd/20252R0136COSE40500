This is my personal respository for the record of the book "Deeplearning from Scratch I" in self-educating.
    -> Link for detail slides. https://www.notion.so/1-286d3503835780638c0cdff3edf0acf1?source=copy_link
    -> ‼️ For personal study and review purposes, there's a lot of overlapping code. ‼️

📚 contents
1. Hello, Python
2. Perceptron
    - 'BasicPerceptron' module in perceptron.py
3. Neural Network
    - basic activation in activaton.py
    - basic 3-dim nn in basic_nn.py
    - mnist pipeline using 3-dim nn in mnist_nn.py (not training just inference)
4. Neural Network Training
    - numerical graident practive in diff.py(one-dim function) and gradient.py(multi-dim function)
    - 'TwoLayerNet' module in two_layer_net.py
    - mnist training pipeline using numerical gradient in mnist_with_numerical.py
5. Backpropagation
    - 'Layer' module with backprop in basic_layer.py and layer.py
    - mnist training pipeline using backpropagation in mnist_with_backprop.py
6. Training Options
    - famous optimizer modules in optimizer.py
    - compare mnist result per each optimizer in compare_optimizer_with_mnist.py
    - activation histogram per weight initialization in weight_init_histogram.py
    - compare mnist result per each initalization in compare_init_with_mnist.py
    - compare mnist result with batch normalization in compare_batchnorm_with_mnist.py
    - hyperparameter optimize in hyper_optim_with_mnist.py