# Gradient Descent Algorithms

- Batch Gradient Descent.
- Stochastic Gradient Descent.
- Mini-Bach Gradient Descent.

<img src="MatlabScripts/imgs/GradientDescentAlgorithms.png"  width="100%">

# Derivatives on Computational Graphs

$\frac{\partial{Z}}{\partial{X}} \rightarrow$ Sum over all possible paths between node $X$ and node $Z$, multiplying the derivatives on each edge of the path together.

<img src="MatlabScripts/imgs/chainRule0.png"  width="100%">
<img src="MatlabScripts/imgs/chainRule1.png"  width="100%">
<img src="MatlabScripts/imgs/chainRule2.png"  width="100%">



## *Feed forward activations*

$$a^{l}_j = \sigma \bigg( \sum_k w^l _{jk} a^{l-1}_k + b^l_j \bigg)$$

## *Feed forward activations in vectorized form*

$$a^l = \sigma \bigg( W^la^{l-1}+b^l \bigg)$$


## *Cost function* $MSE$

$$C(w,b) = \frac{1}{n} \sum_x || y(x) - a^L||^2 $$


## *Gradient vector of the cost function*

$$TrainingInputs=x_1, x_2, \ldots,x_n$$

$$MiniBatches=[X_1, X_2, \ldots,X_m],[X_1, X_2, \ldots,X_m]...$$

$$\frac{1}{m} \sum_{j=1}^m \nabla C_{X_{j}} \approx \frac{1}{n}\sum_{x=1}^n\nabla C_x = \nabla C$$

$$\nabla C \approx \frac{1}{m} \sum_{j=1}^m \nabla C_{X_{j}}$$


## *Update W,b using Gradient Descent*

$$w_k \rightarrow w_k' = w_k-\eta \frac{\partial C}{\partial w_k}$$

$$b_l \rightarrow b_l' = b_l-\eta \frac{\partial C}{\partial b_l}$$


## *Update W,b using Stochastic Gradient Descent with Mini-Batches*

$$w_k \rightarrow w_k' = w_k-\frac{\eta}{m}\sum_j \frac{\partial C_{X_j}}{\partial w_k}$$

$$b_l \rightarrow b_l' = b_l-\frac{\eta}{m}\sum_j \frac{\partial C_{X_j}}{\partial b_l}$$

