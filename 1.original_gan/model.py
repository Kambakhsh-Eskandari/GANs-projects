import torch 
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=25, size=(1,28,28)):
    
    ## function for visualizing image tensors
    
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.show()
    
    
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


# def get_generator_block(input_dim, output_dim):
#     return nn.Sequential(
#         nn.Linear(input_dim, output_dim),
#         nn.BatchNorm1d(output_dim),
#         nn.ReLU(inplace=True)
#     )
    
    
class Generator(nn.Module):
    
    def __init__(self, z_dim, im_dim = 784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(hidden_dim, hidden_dim*2),
            self.get_generator_block(hidden_dim*2, hidden_dim*4),
            self.get_generator_block(hidden_dim*4, hidden_dim*8),
            self.nn.Linear(hidden_dim*8, im_dim),
            self.nn.Sigmoid()
        )
    def get_generator_block(self, input_dim, output_dim):
        return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )
        
     
    def forward(self, noise):
        return self.gen(noise)
    
    
# gen loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    
    noise = get_noise(num_images, z_dim, device=device)
    
    fake_image = gen(noise)
    disc_fake = disc(fake_image)
    
    gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))
    
    return gen_loss


# def get_discriminator_block(input_dim, output_dim):
    
#     return nn.Sequential(
#         nn.Linear(input_dim, output_dim),
#         nn.LeakyReLU(0.2)
#     )
    
    
class Discriminator(nn.Module):
    
    def __init__(self, im_dim=784, hidden_dim= 128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            self.get_discriminator_block(hidden_dim, 1)
        )
        
    def get_discriminator_block(self,input_dim, output_dim):
    
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, image):
        return self.disc(image)
    
    
    
## disc loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    ## real : batch of real images
    ## num_images : the number of images the generator should produce, 
    # which is also the length of the real images
    
    noise = get_noise(num_images, z_dim, device = device)
    
    fake_img = gen(noise)
    disc_pred_fake = disc(fake_img.detach())
    disc_loss_fake = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))
    
    disc_pred_real = disc(real)
    disc_loss_real = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2
    
    return disc_loss



criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

## load mnist dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

device = 'cuda'



## initialzing generator, discriminator and optimizer
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)



# Train
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False

for epoch in range(n_epochs):
    ## DataLoader returns the batches
    
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        
        ## Flatten the batch of real images from dataset
        real = real.view(cur_batch_size, -1).to(device)
        
        ## Update the discriminator ## 
        # zero out 
        disc_opt.zero_grad()
        
        # calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        
        # update gradients 
        disc_loss.backward(retain_graph=True)
        
        #update optimizer
        disc_opt.step()
        
        ## Update generator
        
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()
        
        ## Keep track of the average disc loss
        mean_discriminator_loss += disc_loss.item() / display_step
        
        ## Keep track of the average gen loss
        mean_generator_loss += gen_loss.item() / display_step
        
        ## Visualizing ## 
        if cur_step % display_step ==0 and cur_step > 0:
            print(f"Step {cur_step} : Generator loss: {mean_generator_loss}, Discriminator loss : {mean_discriminator_loss}, cur_batch_size : {cur_batch_size}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_discriminator_loss = 0
            mean_discriminator_loss = 0
        
        cur_step += 1