# Tractable Function-Space Variational Inference in Bayesian Neural Networks (FSVI)

This repository will contain the official implementation for

**_Tractable Function-Space Variational Inference in Bayesian Neural Networks_**; Tim G. J. Rudner, Zonghao Chen, Yee Whye Teh, Yarin Gal. **NeurIPS 2022**.

**Abstract:** Reliable predictive uncertainty estimation plays an important role in enabling the deployment of neural networks to safety-critical settings. A popular approach for estimating the predictive uncertainty of neural networks is to define a prior distribution over the network parameters, infer an approximate posterior distribution, and use it to make stochastic predictions. However, explicit inference over neural network parameters makes it difficult to incorporate meaningful prior information about the data-generating process into the model. In this paper, we pursue an alternative approach. Recognizing that the primary object of interest in most settings is the distribution over functions induced by the posterior distribution over neural network parameters, we frame Bayesian inference in neural networks explicitly as inferring a posterior distribution over functions and propose a scalable function-space variational inference method that allows incorporating prior information and results in reliable predictive uncertainty estimates. We show that the proposed method leads to state-of-the-art uncertainty estimation and predictive performance on a range of prediction tasks and demonstrate that it performs well on a challenging safety-critical medical diagnosis task in which reliable uncertainty estimation is essential.

<p align="center">
  &#151; <a href="https://timrudner.com/fsvi"><b>View Paper</b></a> &#151;
</p>

**Code will be released soon!**
