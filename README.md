## Strata
Mechanistically-informed model structures for transcriptomics perturbation effect prediction. 

Strata is a research framework for predicting transcriptional responses to genetic perturbations using single-cell RNA-seq data and curated molecular interaction priors.

**Core contribution** - A learnable, biologically guided perturbation-propagation model that uses curated gene regulatory/signaling networks as a prior to model genetic perturbations across the transcriptome and generalize to unseen gene combinations. Instead of treating a biological network as a fixed ground-truth graph which is the case with current state of the art approaches, Strata initializes a learnable graph from curated prior knowledge and allows perturbational data to refine the strength of these relationships during training.

