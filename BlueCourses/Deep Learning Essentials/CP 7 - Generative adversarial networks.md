### The generative adversarial architecture
- GAN are deep neural network architectures comprised of two networks:
- - The adversarial network, which generates new data instances
- - The discriminative network, which evaluates the generated instances
- GAN were introduced by Ian Goodfellow and his colleagues in 2014
- Discriminative algorithms try to classify input data
- Generative algorithms do the opposite: they generate new data instances
- GAN Lab: https://poloclub.github.io/ganlab/

### ☞ Generating digits with a GAN
- See notebook 'dle_gan_imagedigits.ipynb'

### Challenges
- Our simple 'fully connected' approach generates something that resembles written digits
- GAN are notoriously hard to train
- Nash equilibrium is hard to achieve
- Vanishing gradients are a common problem
- Mode collapse is another common problem
- No good evaluation metric to assess generator

### Best practices
- It's best to use tanh as the final activation function for the generator
- - Though this means that we need to preprocess the normal images to lie between -1 and 1 as well 
- A leaky ReLU works better than a ReLU in the discriminator
- For the latent samples (the noise), sampling from a gaussian distribution works better that uniformly
- Using Batch normalization helps as well
- When training the discrimination, don't mix real and fake instances in one batch
- - Construct different batches for real and fake instances
- Use label smoothing
- - For each incoming sample, if it real, replace the label with a random number between e.g 0.7 and 1.0. If it fake sample, replace with 0.0 and 0.3
- - Or even: make the labels the noisy for the discriminator: occasionally flip the labels
- Use the Adam optimizer
- - Though setting of parameters such as learning rate very important
- - Discriminator loss should have low variance and should not spike too much
- Don't try to apply tricks based on the generator or discriminator loss to decide when to stop training
- Other ideas: dropout, label smoothing, feature matching, minibatch discrimination, virtual batch normalization, ...

### ☞ Generating digits with a GAN, revisited
- See notebook 'dle_gan_imagedigitsrevisited.ipynb'

### Further aspects
-

### Variants
- Wasserstein GAN (2017)
- Use a better metric of distribution similarity
- - To measure divergence between generated and real distributions
- Newer approaches are getting much better see StyleGAN and StyleGAN2

### Use cases
- Text to image generation
- Image to image translation
- Increasing image resolution
- Predict next video frame
- Deepfake generation