Implementation of "Generative Modeling for Protein Structures" (NeurIPS 2018), the first research paper on generative AI for protein design. I've found this paper particularly interesting for its use of a convex optimization algorithm to recover structure predictions, the math behind which has been a lot of fun to explore. There was no publicly available code for the paper before this implementation, as far as I know.

Replicating the following steps:
1. Download PDB train and test data
2. Generate pairwise distance datasets of protein fragments
3. Train a GAN on the dataset
4. Recover structure predictions via convex optimization
5. Evaluate the quality of the predictions

##

...


Q: will any convex optimization algorithm do the trick?
