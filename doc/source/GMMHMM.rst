We'll use Matrix Cookbook (https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for some useful equations.

General form of expectation :math:`Q(\theta, \theta^{old})` of any GMM model looks like this (Bishop, (13.17)):

.. math::
  Q(\theta, \theta^{old}) = \sum_{k=1}^{K} \gamma(z_{1k})\ln \pi_k + \sum_{n=2}^{N} \sum_{j=1}^{K} \sum_{k=1}^{K} \xi (z_{n-1,j}, z_{nk}) \ln A_{jk} + \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma(z_{nk}) \ln p(x_n | \phi_k)

In the case of GMMHMM PDF in the last term looks like this:

.. math::
  p(x_n | \phi_k) = \sum_{l=1}^{L} \pi_{kl} \mathcal{N}(x_n | \mu_{kl}, \Sigma_{kl})

Thus:

.. math::
  Q(\theta, \theta^{old}) = \sum_{k=1}^{K} \gamma(z_{1k})\ln \pi_k + \sum_{n=2}^{N} \sum_{j=1}^{K} \sum_{k=1}^{K} \xi (z_{n-1,j}, z_{nk}) \ln A_{jk} + \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma(z_{nk}) \sum_{l=1}^{L} \pi_{kl} \mathcal{N}(x_n | \mu_{kl}, \Sigma_{kl})

Priors for parameters :math:`\pi_{p}`:

.. math::
  p(\pi_p | \alpha_p) = \frac{1}{\text{B}(\alpha_p)} \prod_{l=1}^{L} \pi_{pl}^{\alpha_{pl} - 1}

Priors for parameters :math:`\mu_{pt}`:

.. math::
  p(\mu_{pt} | \mu_{pt0}, \lambda) = \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda} \Sigma_{pt})

Priors for parameters :math:`\Sigma_{kl}` in 'full' case:

.. math::
  p(\Sigma_{pt} | \Psi_{pt}, \nu_{pt}) = \frac{\left|\Psi_{pt}\right|^{\frac{\nu_{pt}}{2}}}{2 ^ \frac{\nu_{pt} D} {2} \Gamma_D (\frac{\nu_{pt}}{2})} \left|\Sigma_{pt}\right|^{-\frac{\nu_{pt} + D + 1}{2}} e^{-\frac{1}{2} \text{tr} (\Psi_{pt} \Sigma_{pt}^{-1})} = \text{IW}(\Sigma_{pt} | \Psi_{pt}, \nu_{pt})

Priors for parameters :math:`\Sigma_{kl}` in 'tied' case:

.. math::
  p(\Sigma_p | \Psi_p, \nu_p) = \text{IW}(\Sigma_p | \Psi_p, \nu_p)

Priors for parameters :math:`\sigma_{kld} ^ 2` in 'diag' case:

.. math::
  p(\sigma_{kld} ^ 2 | \alpha_{kld}, \beta_{kld}) = \frac { \beta_{kld} ^ {\alpha_{kld}} } { \Gamma (\alpha_{kld}) } (\sigma_{kld} ^ 2) ^ {-\alpha_{kld} - 1} \exp \Big( \frac {-\beta_{kld}} {\sigma_{kld}^2} \Big) = \Gamma^{-1}(\sigma_{kld} ^ 2 | \alpha_{kld}, \beta_{kld})


Priors for parameters :math:`\sigma_{kl} ^ 2` in 'spherical' case:

.. math::
  p(\sigma_{kl} ^ 2 | \alpha_{kl}, \beta_{kl}) = \Gamma^{-1}(\sigma_{kl} ^ 2 | \alpha_{kl}, \beta_{kl})


The whole prior log-distribution:

.. math::
  \ln p(\pi, \mu, \Sigma) = \sum_{p=1}^{P} \Big(\ln \frac{1}{\text{B}(\alpha_p)} + \sum_{l=1}^{L} (\alpha_{pl} - 1) \ln \pi_{pl}\Big) + \sum_{p=1}^{P} \sum_{l=1}^{L} \ln \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda} \Sigma_{pt}) + p(\Sigma)

where :math:`p(\Sigma)` is the appropriate sum of one of four priors for covariances above. In 'full' case it is:

.. math::
  p(\Sigma) = \sum_{p=1}^{P} \sum_{l=1}^{L} \ln \text{IW}(\Sigma_{pt} | \Psi_{pt}, \nu_{pt})

In 'tied' case it is:

.. math::
  p(\Sigma) = \sum_{p=1}^{P} \ln \text{IW}(\Sigma_{p} | \Psi_p, \nu_p)

In 'diag' case it is:

.. math::
  p(\Sigma) = \sum_{p=1}^{P} \sum_{l=1}^{L} \sum_{d=1}^{D} \Gamma^{-1}(\sigma_{kld} ^ 2 | \alpha_{kld}, \beta_{kld})

In 'spherical' case it is:

.. math::
  p(\Sigma) = \sum_{p=1}^{P} \sum_{l=1}^{L} \Gamma^{-1}(\sigma_{kl} ^ 2 | \alpha_{kl}, \beta_{kl})

Thus, in order to derive M-step for MAP-EM algorithm, we should maximize :math:`Q(\theta, \theta^{\text{ old }}) + \ln p(\theta)` w. r. t. :math:`\theta`.

Let's maximize :math:`Q(\theta, \theta^{\text{ old }}) + \ln p(\theta)` w. r. t. some :math:`\pi_{pt}`. These values should satisfy :math:`\sum_{l=1}^{L} \pi_{pl} = 1` :math:`\forall p`. Taking this into account, we maximize :math:`Q(\theta, \theta^{\text{ old }}) + \ln p(\theta)` using Lagrange multiplier and maximizing the following value:

.. math::
  Q(\theta, \theta^{\text{ old }}) + \ln p(\theta) + \sum_{p=1}^{P} \lambda_p (\sum_{l=1}^{L} \pi_{pl} - 1)

Deriving :math:`Q(\theta, \theta^{\text{ old }})` by :math:`\pi_{pt}`:

.. math::
  \frac{\partial Q(\theta, \theta^{\text{ old }})}{\partial \pi_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) \frac {\mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})}

Deriving :math:`\ln p(\theta)` by by :math:`\pi_{pt}`:

.. math::
  \frac{\partial \ln p(\theta)}{\partial \pi_{pt}} = \frac{\alpha_{pt} - 1}{\pi_{pt}}

Deriving :math:`\sum_{p=1}^{P} \lambda_p (\sum_{l=1}^{L} \pi_{pl} - 1)` by :math:`\pi_{pt}`:

.. math::
  \frac {\partial \sum_{p=1}^{P} \lambda_p (\sum_{l=1}^{L} \pi_{pl} - 1)} {\partial \pi_{pt}} = \lambda_p

Final result;

.. math::
  \frac {\partial (Q(\theta, \theta^{\text{ old }}) + \ln p(\theta) + \sum_{p=1}^{P} \lambda_p (\sum_{l=1}^{L} \pi_{pl} - 1))} {\partial \pi_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) \frac {\mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})} + \frac{\alpha_{pt} - 1}{\pi_{pt}} + \lambda_p = 0

Multiplying by :math:`\pi_{pt}` and summing over *t* we get:

.. math::
  \sum_{n=1}^{N} \gamma(z_{np}) \frac {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})} + \sum_{l=1}^{L} (\alpha_{pl} - 1) + \sum_{l=1}^{L} \pi_{pl} \lambda_p = 0

From which we get:

.. math::
  \sum_{n=1}^{N} \gamma(z_{np}) + \sum_{l=1}^{L} (\alpha_{pl} - 1) + \lambda_p = 0

  \lambda_p = -\sum_{n=1}^{N} \gamma(z_{np}) - \sum_{l=1}^{L} (\alpha_{pl} - 1)

Substituting the result for :math:`\lambda_p` into the original expression:

.. math::
  \sum_{n=1}^{N} \gamma(z_{np}) \frac {\mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})} + \frac{\alpha_{pt} - 1}{\pi_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) + \sum_{l=1}^{L} (\alpha_{pl} - 1)

  \sum_{n=1}^{N} \gamma(z_{np}) \frac {\pi_{pt} \mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})} + \alpha_{pt} - 1 = \pi_{pt} \Big(\sum_{n=1}^{N} \gamma(z_{np}) + \sum_{l=1}^{L} (\alpha_{pl} - 1)\Big)

  \frac{\sum_{n=1}^{N} \gamma(z_{np}) \frac {\pi_{pt} \mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})} + \alpha_{pt} - 1} { \sum_{n=1}^{N} \gamma(z_{np}) + \sum_{l=1}^{L} (\alpha_{pl} - 1)} = \pi_{pt}

Let's introduce a few notations:

.. math::
  \frac {\pi_{pt} \mathcal{N} (x_n | \mu_{ pt }, \Sigma_{pt})} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_{pl}, \Sigma_{pl})} = \gamma(\tilde{z}_{npt})

  \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) = N_{pt}

  \sum_{n=1}^{N} \gamma(z_{np}) = N_p

Then the expression for maximizing the :math:`\pi_{pt}` is as follows:

.. math::
  \pi_{pt} = \frac{N_{pt} + \alpha_{pt} - 1} {N_p + \sum_{l=1}^{L} (\alpha_{pl} - 1)}

Let's do the same with :math:`\mu_{pt}`. This time, there aren't any constraints, so the task of maximizing :math:`Q(\theta, \theta^{\text{ old }}) + \ln p(\theta)` reduces to finding partial derivative of this function w. r. t. :math:`\mu_{pt}` and equating it to zero. 

First, let's derivate :math:`\ln p(\theta)` using formula (85) from Matrix Cookbook:

.. math::
  \frac {\partial \ln p(\theta)} {\mu_{pt}} = \frac {\partial (\ln \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}))} {\partial \mu_{pt}} = \frac {\frac {\partial (\mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}))} {\partial \mu_{pt}}} {\mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt})}

  \frac {\partial (\mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}))} {\partial \mu_{pt}} = \frac {\partial \Big(\frac{1} {(2 \pi)^{D/2}} \frac {\sqrt{\lambda_{pt}}} {\left|\Sigma_{pt}\right|^{1/2}}\exp \left \{ -\frac {\lambda_{pt}} {2} (\mu_{pt} - \mu_{pt0})^T \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) \right \}\Big)} {\partial \mu_{pt}} =
 
  \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}) \frac {\partial (-\frac {\lambda_{pt}} {2} (\mu_{pt} - \mu_{pt0})^T \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}))} {\partial \mu_{pt}} = 
  
  \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}) \Big(-\frac {\lambda_{pt}} {2}\Big) 2 \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) = -\lambda_{pt} \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt})

  \frac {\partial \ln p(\theta)} {\mu_{pt}} = -\lambda_{pt} \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0})

Then, let's derivate :math:`Q(\theta, \theta^{\text{old}})` using formula (86):

.. math::
  \frac{\partial Q(\theta, \theta^{\text{ old }})}{\partial \mu_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) \frac {\pi_{pt} \mathcal{N} (x_n | \mu_pt, \Sigma_pt)} {\sum_l \pi_{pl} \mathcal{N} (x_n | \mu_pl, \Sigma_pl)} \Sigma_{pt}^{-1} (x_n - \mu_{pt}) = \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) \Sigma_{pt}^{-1} (x_n - \mu_{pt})
  
Then, let's derivate :math:`\ln p(\theta)`:

.. math::
  \frac {\partial \ln p(\theta)} {\mu_{pt}} = \frac {\partial (\ln \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}))} {\partial \mu_{pt}} = \frac {\frac {\partial (\mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}))} {\partial \mu_{pt}}} {\mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt})}

  \frac {\partial (\mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}))} {\partial \mu_{pt}} = \frac {\partial \Big(\frac{1} {(2 \pi)^{D/2}} \frac {\sqrt{\lambda_{pt}}} {\left|\Sigma_{pt}\right|^{1/2}}\exp \left \{ -\frac {\lambda_{pt}} {2} (\mu_{pt} - \mu_{pt0})^T \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) \right \}\Big)} {\partial \mu_{pt}} =
 
  \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}) \frac {\partial (-\frac {\lambda_{pt}} {2} (\mu_{pt} - \mu_{pt0})^T \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}))} {\partial \mu_{pt}} = 
  
  \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt}) \Big(-\frac {\lambda_{pt}} {2}\Big) 2 \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) = -\lambda_{pt} \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) \mathcal{N} (\mu_{pt} | \mu_{pt0}, \frac{1}{\lambda_{pt}} \Sigma_{pt})

  \frac {\partial \ln p(\theta)} {\mu_{pt}} = -\lambda_{pt} \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0})

Now, the result is:

.. math::
  \frac {\partial (Q(\theta, \theta^{\text{ old }}) + \ln p(\theta))} {\partial \mu_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) \Sigma_{pt}^{-1} (x_n - \mu_{pt}) - \lambda_{pt} \Sigma_{pt}^{-1} (\mu_{pt} - \mu_{pt0}) = 0

  \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) (x_n - \mu_{pt}) - \lambda_{pt} (\mu_{pt} - \mu_{pt0}) = 0

  \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) x_n - \mu_{pt}\sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) - \lambda_{pt} \mu_{pt} + \lambda_{pt}\mu_{pt0} = 0

  \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) x_n - \mu_{pt} N_{pt}  - \lambda_{pt} \mu_{pt} + \lambda_{pt}\mu_{pt0} = 0

  \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) x_n + \lambda_{pt}\mu_{pt0} = \mu_{pt} (N_{pt}  + \lambda_{pt})

  \mu_{pt} = \frac {\sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) x_n + \lambda_{pt}\mu_{pt0}} {N_{pt}  + \lambda_{pt}}


Basically all the same with :math:`\Sigma`, but with 4 different variants of it, for full, tied, diagonal and spherical covariance.

Let's start with 'full'. We're trying to find :math:`\Sigma_{pt}`. First, derivative of :math:`Q(\theta, \theta^{\text{ old }})`:

.. math::
  \frac {\partial Q(\theta, \theta^{\text{old}})} {\partial \Sigma_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) \frac {\frac {\partial \mathcal{N} (x_n | \mu_{pt}, \Sigma_{pt})} {\partial \Sigma_{pt}}} {\mathcal{N} (x_n | \mu_{pt}, \Sigma_{pt})}

  \frac {\partial \mathcal{N} (x | \mu, \Sigma)} {\partial \Sigma} = \frac {\partial \Big(\frac{1} {(2 \pi)^{D/2}} \frac {1} {\left|\Sigma\right|^{1/2}}\exp \left \{ -\frac {1} {2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right \}\Big)} {\partial \Sigma} = 

  = \frac{1} {(2 \pi)^{D/2}} \frac {\partial \left|\Sigma\right| ^ {-\frac{1}{2}}} {\partial \Sigma} \exp \{\cdots\} + \frac{1} {(2 \pi)^{D/2}} \frac {1} {\left|\Sigma\right|^{1/2}} \frac {\partial \exp \{\cdots\}} {\partial \Sigma}

Using the chain rule and formula (49) from Matrix Cookbook to find the derivative of determinant, for the first term we get:

.. math::
  \frac {\partial \left|\Sigma\right| ^ {-\frac{1}{2}}} {\partial \Sigma} = \frac {\partial \left|\Sigma\right| ^ {-\frac{1}{2}}} {\partial \left|\Sigma\right|} \frac {\partial \left|\Sigma\right|} {\partial \Sigma} = - \frac {1} {2} \left|\Sigma\right| ^ {-\frac{3}{2}} \left|\Sigma\right| \Sigma^{-T} = - \frac {1} {2} \frac {1} {\left|\Sigma\right| ^ {1/2}} \Sigma^{-T}

Using the chain rule and formula (61) from Matrix Cookbook to find the derivative of inverse, for the second term we get:

.. math::
  \frac {\partial \exp \{\cdots\}} {\partial \Sigma} = \frac {\partial \exp \{\cdots\}} {\{\cdots\}} \frac {\{\cdots\}} {\partial \Sigma} = \exp \{\cdots\} \Big(-\frac {1} {2} \Big) \frac {\partial \Big((x - \mu)^T \Sigma^{-1} (x - \mu) \Big)} {\partial \Sigma} = 
  
  = \exp \{\cdots\} \Big(-\frac {1} {2} \Big) (-\Sigma^{-T} (x - \mu) (x - \mu)^T \Sigma^{-T})

Combining the two:

.. math::
  \frac {\partial \mathcal{N} (x | \mu, \Sigma)} {\partial \Sigma} = \frac{1} {(2 \pi)^{D/2}} \frac {1} {\left|\Sigma\right| ^ {1/2}}  \exp \{\cdots\} \Big(-\frac {1} {2}\Big) \Sigma^{-T} + \frac{1} {(2 \pi)^{D/2}} \frac {1} {\left|\Sigma\right|^{1/2}} \exp \{\cdots\} \Big(-\frac {1} {2} \Big) (-\Sigma^{-T} (x - \mu) (x - \mu)^T \Sigma^{-T}) = 

  =  \mathcal{N} (x | \mu, \Sigma)\Big(-\frac {1} {2}\Big) \Sigma^{-T} + \mathcal{N} (x | \mu, \Sigma) \frac {1} {2} (\Sigma^{-T} (x - \mu) (x - \mu)^T \Sigma^{-T})

From which we finally get:

.. math::
  \frac {\partial Q(\theta, \theta^{\text{old}})} {\partial \Sigma_{pt}} = \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) \Big( \Big(-\frac {1} {2}\Big) \Sigma_{pt}^{-T} + \frac {1} {2} (\Sigma_{pt}^{-T} (x_n - \mu_{pt}) (x_n - \mu_{pt})^T \Sigma_{pt}^{-T}) \Big) = 
  
  = \Big(-\frac {1} {2}\Big) \Sigma_{pt}^{-T} \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) + \frac {1} {2} \Sigma_{pt}^{-T} (\sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) (x_n - \mu_{pt}) (x_n - \mu_{pt})^T) \Sigma_{pt}^{-T} = 
  
  = \Big(-\frac {1} {2}\Big) \Sigma_{pt}^{-T} N_{pt} + \frac {1} {2} \Sigma_{pt}^{-T} \big(\sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) (x_n - \mu_{pt}) (x_n - \mu_{pt})^T\big) \Sigma_{pt}^{-T} 


Now to :math:`\ln p(\theta)`:

.. math::
  \frac {\partial \ln p(\theta)} {\partial \Sigma_{pt}} = \frac {\frac {\partial \mathcal {N} (\mu_{pt} | \mu_{pt0}, \frac {1} {\lambda_pt} \Sigma_{pt})} {\partial \Sigma_{pt}}} {\mathcal {N} (\mu_{pt} | \mu_{pt0}, \frac {1} {\lambda_pt} \Sigma_{pt})} + \frac {\frac {\partial \text {IW} (\Sigma_{pt} | \Psi_{pt}, \nu_{pt})} {\partial \Sigma_{pt}}} {\text {IW} (\Sigma_{pt} | \Psi_{pt}, \nu_{pt})}

We can calculate the derivative of normal distribution in the equation above using previous results:

.. math::
  \frac {\partial \mathcal {N} (x | \mu, \frac {1} {\lambda} \Sigma)} {\partial \Sigma} = \frac {\partial \Big(\frac{1} {(2 \pi)^{D/2}} \frac {\sqrt{\lambda}} {\left|\Sigma\right|^{1/2}} \exp \left \{ -\frac {\lambda} {2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right \}\Big)} {\partial \Sigma} 

  = \frac{\sqrt{\lambda}} {(2 \pi)^{D/2}} \frac {\partial \left|\Sigma\right| ^ {-\frac{1}{2}}} {\partial \Sigma} \exp \{\cdots\} + \frac{1} {(2 \pi)^{D/2}} \frac {\sqrt{\lambda}} {\left|\Sigma\right|^{1/2}} \frac {\partial \exp \{\cdots\}} {\partial \Sigma}

  = \mathcal{N} (x | \mu, \frac {1} {\lambda} \Sigma)\Big(-\frac {1} {2}\Big) \Sigma^{-T} + \mathcal{N} (x | \mu, \frac {1} {\lambda} \Sigma) \frac {\lambda} {2} \Sigma^{-T} (x - \mu) (x - \mu)^T \Sigma^{-T}

Now to the derivative of inverse-Wishart distribution:

.. math::
  \frac {\partial \text {IW} (\Sigma | \Psi, \nu)} {\partial \Sigma} = \frac {\partial \Big( \frac{\left|\Psi\right|^{\frac{\nu}{2}}}{2 ^ \frac{\nu D} {2} \Gamma_D (\frac{\nu}{2})} \left|\Sigma\right|^{-\frac{\nu + D + 1}{2}} \exp \left \{-\frac{1}{2} \text{tr} (\Psi \Sigma^{-1}) \right \} \Big)} {\partial \Sigma} 
  
  = \frac{ \left| \Psi \right| ^ {\frac {\nu} {2} } } {2 ^ \frac{\nu D} {2} \Gamma_D (\frac{\nu}{2})} \frac {\partial  \left|\Sigma\right|^{-\frac{\nu + D + 1}{2}} } {\partial \Sigma} \exp \left  \{-\frac{1}{2} \text{tr} (\Psi \Sigma^{-1}) \right \} + \frac{\left|\Psi\right|^{\frac{\nu}{2}}}{2 ^ \frac{\nu D} {2} \Gamma_D (\frac{\nu}{2})} \left|\Sigma\right|^{-\frac{\nu + D + 1}{2}} \exp \left \{-\frac{1}{2} \text{tr} (\Psi \Sigma^{-1}) \right \} \Big( - \frac {1} {2} \Big) \frac {\partial \text{tr} (\Psi \Sigma^{-1})} {\partial \Sigma}

Using the same equation (49) from Matrix Cookbook, we get:

.. math::

  \frac {\partial  \left|\Sigma\right|^{-\frac{\nu + D + 1}{2}} } {\partial \Sigma} = \frac {\partial  \left|\Sigma\right|^{-\frac{\nu + D + 1}{2}} } {\partial \left| \Sigma \right|} \frac {\partial \left| \Sigma \right|} {\partial \Sigma} = - \frac {(\nu + D + 1)} {2} \left| \Sigma \right| ^ {-\frac{\nu + D + 1}{2}} \Sigma^{-1} \frac {\partial \left| \Sigma \right|} {\partial \Sigma} = -\frac {(\nu + D + 1)} {2} \left| \Sigma \right| ^ {-\frac{\nu + D + 1}{2}} \Sigma^{-1} \left| \Sigma \right| \Sigma^{-T} = 

   = -\frac {(\nu + D + 1)} {2} \Sigma^{-T} \left| \Sigma \right| ^ {-\frac{\nu + D + 1}{2}}

Using formula (63), for the derivative of a trace we get:

.. math::
  \frac {\partial \text{tr} (\Psi \Sigma^{-1})} {\partial \Sigma} = -\Sigma^{-T} \Psi^T \Sigma^{-T}

Combining the two, we get:

.. math::
  \frac {\partial \text {IW} (\Sigma | \Psi, \nu)} {\partial \Sigma} = -\frac {(\nu + D + 1)} {2} \Sigma^{-T} \text {IW} (\Sigma | \Psi, \nu) + \frac {1} {2} \Sigma^{-T} \Psi^T \Sigma^{-T} \text {IW} (\Sigma | \Psi, \nu)

Now, finally, we can get the whole derivative of prior distribution w. r. t. :math:`\Sigma_{pt}`:

.. math::
  \frac {\partial \ln p(\theta)} {\partial \Sigma_{pt}} = \Big(-\frac {1} {2}\Big) \Sigma_{pt}^{-T} + \frac {\lambda_{pt}} {2} \Sigma_{pt}^{-T} (\mu_{pt} - \mu_{pt0}) (\mu_{pt} - \mu_{pt0})^T \Sigma_{pt}^{-T} + \Big(-\frac {(\nu_{pt} + D + 1)} {2}\Big) \Sigma_{pt}^{-T} + \frac {1} {2} \Sigma_{pt}^{-T} \Psi_{pt}^T \Sigma_{pt}^{-T}  

Then, we can equate the derivative of :math:`Q(\theta, \theta ^ {\text{old}}) + \ln (\theta)` to 0:

.. math::
  \frac {\partial (Q(\theta, \theta ^ {\text{old}}) + \ln (\theta))} {\partial \Sigma_{pt}} = \Big(-\frac {1} {2}\Big) \Sigma_{pt}^{-T} N_{pt} + \frac {1} {2} \Sigma_{pt}^{-T} \big(\sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) (x_n - \mu_{pt}) (x_n - \mu_{pt})^T\big) \Sigma_{pt}^{-T} +
 
  + \Big(-\frac {1} {2}\Big) \Sigma_{pt}^{-T} + \frac {\lambda_{pt}} {2} \Sigma_{pt}^{-T} (\mu_{pt} - \mu_{pt0}) (\mu_{pt} - \mu_{pt0})^T \Sigma_{pt}^{-T} + \Big(-\frac {(\nu_{pt} + D + 1)} {2}\Big) \Sigma_{pt}^{-T} + \frac {1} {2} \Sigma_{pt}^{-T} \Psi_{pt}^T \Sigma_{pt}^{-T} = 0

Multiplying by :math:`2 \Sigma^{T}` from both sides, we get:

.. math::
  -\Sigma_{pt}^{T} N_{pt} + \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) (x_n - \mu_{pt}) (x_n - \mu_{pt})^T - \Sigma_{pt}^{T} + \lambda_{pt} (\mu_{pt} - \mu_{pt0}) (\mu_{pt} - \mu_{pt0})^T - (\nu_{pt} + D + 1) \Sigma_{pt}^{T} +  \Psi_{pt}^T = 0
  
Let's, once again, introduce a few notations:

.. math::
  C_{npt} = (x_n - \mu_{pt}) (x_n - \mu_{pt})^T

  C_{\mu_{pt}} = (\mu_{pt} - \mu_{pt0}) (\mu_{pt} - \mu_{pt0})^T

Let's rewrite the expression above using these notations:

.. math::
  -\Sigma_{pt}^{T} N_{pt} + \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) C_{npt} - \Sigma_{pt}^{T} + \lambda_{pt} C_{\mu_{pt}} - (\nu_{pt} + D + 1) \Sigma_{pt}^{T} +  \Psi_{pt}^T = 0

  \sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) C_{npt} + \lambda_{pt} C_{\mu_{pt}} + \Psi_{pt}^T = \Sigma_{pt}^{T} (N_{pt} + 1 + (\nu_{pt} + D + 1))

  \Sigma_{pt}^T = \Sigma_{pt} = \frac {\sum_{n=1}^{N} \gamma(z_{np}) \gamma(\tilde{z}_{npt}) C_{npt} + \lambda_{pt} C_{\mu_{pt}} + \Psi_{pt}^T} {N_{pt} + 1 + \nu_{pt} + D + 1}
