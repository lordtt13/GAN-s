# Cycle GAN Implementation

CycleGAN is a cross-domain GAN. Like other GANs, it can be trained
in unsupervised manner.
CycleGAN is made of two generators (G & F) and two discriminators.
Each generator is a U-Network. The discriminator is a 
typical decoder network with the option to use PatchGAN structure.
There are 2 datasets: x = source, y = target. 
The forward-cycle solves x'= F(y') = F(G(x)) where y' is 
the predicted output in y-domain and x' is the reconstructed input.
The target discriminator determines if y' is fake/real. 
The objective of the forward-cycle generator G is to learn 
how to trick the target discriminator into believing that y'
is real.
The backward-cycle improves the performance of CycleGAN by doing 
the opposite of forward cycle. It learns how to solve
y' = G(x') = G(F(y)) where x' is the predicted output in the
x-domain. The source discriminator determines if x' is fake/real.
The objective of the backward-cycle generator F is to learn 
how to trick the target discriminator into believing that x' 
is real.
References:
[1]Zhu, Jun-Yan, et al. "Unpaired Image-to-Image Translation Using
Cycle-Consistent Adversarial Networks." 2017 IEEE International
Conference on Computer Vision (ICCV). IEEE, 2017.
[2]Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net:
Convolutional networks for biomedical image segmentation."
International Conference on Medical image computing and
computer-assisted intervention. Springer, Cham, 2015.