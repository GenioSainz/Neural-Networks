# Gradient Descent Algorithms

- Batch Gradient Descent.
- Stochastic Gradient Descent.
- Mini-Bach Gradient Descent.

<img src="MatlabScripts/imgs/GradientDescentAlgorithms.png"  width="100%">

# Derivatives on Computational Graphs

$\frac{\partial{Z}}{\partial{X}} \rightarrow$ Sum over all possible paths between node $X$ and node $Z$, multiplying the derivatives on each edge of the path together.

<img src="MatlabScripts/imgs/chainRule0.png"  width="100%">
<img src="MatlabScripts/imgs/chainRule1.png"  width="100%">


## *Net feed forward Matrix Dimensions*

<img src="MatlabScripts/imgs/net535_0.png"  width="100%">
<img src="MatlabScripts/imgs/net535_1.png"  width="100%">
<!-- <img src="MatlabScripts/imgs/net535_2.png"  width="100%"> -->


## *Feed forward activations: Single Neuron*

$$a^{l}_j = \sigma \bigg( \sum_k w^l _{jk} a^{l-1}_k + b^l_j \bigg)$$



$w_{jk}$ denote the weight for the connection from the $k \textrm{  } neuron$ in the $(l−1)$ layer to the $j \textrm{  } neuron$ in the layer $l$.

## *Feed forward activation: Vectorized form*

$$a^l = \sigma \bigg( W^la^{l-1}+b^l \bigg)$$


## *Cost function* $MSE$

$$C(w,b) = \frac{1}{n} \sum_x || y(x) - a^L||^2 $$


## *Gradient vector of the cost function*

$$TrainingInputs=x_1, x_2, \ldots,x_n$$

$$MiniBatches=[X_1, X_2, \ldots,X_m],[X_1, X_2, \ldots,X_m]...$$



$$\nabla C =  \frac{1}{n}\sum_{x=1}^n\nabla C_x \approx \frac{1}{m} \sum_{j=1}^m \nabla C_{X_{j}} $$

$$\nabla C \approx \frac{1}{m} \sum_{j=1}^m \nabla C_{X_{j}}$$


## *Update W,b using Gradient Descent*

$$w_k \rightarrow  w_k-\eta \frac{\partial C}{\partial w_k}$$

$$b_l \rightarrow  b_l-\eta \frac{\partial C}{\partial b_l}$$


## *Update W,b using Stochastic Gradient Descent with Mini-Batches*

$$w_k \rightarrow  w_k-\frac{\eta}{m}\sum_j \frac{\partial C_{X_j}}{\partial w_k}$$

$$b_l \rightarrow b_l-\frac{\eta}{m}\sum_j \frac{\partial C_{X_j}}{\partial b_l}$$


## *Backpropagation*

Backpropagation compute:
- The partial derivatives $\partial C_x/ \partial W^l$ and $\partial C_x/ \partial b^l$ for a single training input. We then recover $\partial C/ \partial W^l$ and $\partial C/ \partial b^l$ averaging training examples on the mini bach.

- $Error$ $\delta^l$ and then will relate  $\delta^l$ to $\partial C/ \partial W^l$ and $\partial C/ \partial b^l$.

- Weight and Biases will learn slowly if:
  - The input neuron is low-activation $\rightarrow a^{l-1}_k$.
  - The output neuron has saturated $\rightarrow \sigma'(z^l)$

<!-- $$ \frac{\partial C}{\partial w^L_{jk}} =  
                                              \underbrace{\frac{\partial C}{\partial a^L_{j}} 
                                              \frac{\partial a^L_{j}}{\partial z^L_{j}}}_{\delta^L_j}
                                              \frac{\partial z^L_{j}}{\partial w^L_{jk}}
                                              =\delta^L_j a^{L-1}_{k}
                                              $$

$$ \frac{\partial C}{\partial b^L_{j}} =  
                                          \underbrace{\frac{\partial C}{\partial a^L_{j}} 
                                          \frac{\partial a^L_{j}}{\partial z^L_{j}}}_{\delta^L_j}
                                          \frac{\partial z^L_{j}}{\partial b^L_{j}}
                                          =\delta^L_j
                                          $$ -->

- Backpropagation Equations:

$$
\begin{align}
    & \delta^L = \frac{\partial C}{\partial z^L}\\
    & \\  
    & \delta^L = \nabla_{a^L} C \odot \sigma'(z^L) \\
    & \\
    & \delta^l   = ([W^{l+1}]^T \delta^{l+1}) \odot \sigma'(z^l) \\
    & \\
    & \frac{\partial C}{\partial W^l} = \delta^l [a^{l-1}]^T\\
    & \\
    & \frac{\partial C}{\partial b^l} =\delta^l \\
\end{align}
$$


<img src="MatlabScripts/imgs/netBackProp.png"  width="100%">

## *Gradient Descent Combined with Backpropagation*

$$
\begin{align}
 W^l & \rightarrow W^l -\frac{\eta}{m} \sum^m_x \delta^l_x [a^{l-1}_x]^T\\
 & \\
 b^l & \rightarrow b^l -\frac{\eta}{m} \sum^m_x \delta^l_x
\end{align}
$$

## *References*

[Deriving the Backpropagation Equations from Scratch (Part 1)](https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a)

[Deriving the Backpropagation Equations from Scratch (Part 2)](https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-2-693d4162e779)

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

[Backpropagation from the beginning](https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d)

[Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus)

