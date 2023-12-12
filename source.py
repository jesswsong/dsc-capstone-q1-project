import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

num_steps = 100
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

def create_dataset(num_points=10**4, noise=0.1):
    
    """
    First step: create and load data
    
    return:
    - data, data in a format to be visualized
    - dataset, the actual dataset that we will be performing on
    - s_color, the original color assignment of data
    """
    scale = 10
    dim = 2
    s_roll, s_color = make_swiss_roll(num_points, noise=noise) 
    s_roll = s_roll[:,[0, dim]] / scale

    data = s_roll.T

    fig = plt.figure(figsize = (5,5))
    plt.scatter(*data, edgecolor='white', c=s_color);
    
    plt.savefig('dataset.png')

    dataset = torch.Tensor(s_roll).float()
    
    return data, dataset, s_color


def q_x(x_0,t):
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)


def forward_q_x(dataset):
    
    """
    Second step: Create the forward process of x_t
    return a function that calculates the diffused distribution x at timestamp t
    """

    assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
    alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
    ==one_minus_alphas_bar_sqrt.shape, "forward process parameter shape mistmatch"
    
    return q_x(dataset, torch.tensor([num_steps-1]))


def visualize_forward(dataset, s_color, num_rows=2, num_cols=10, fig_size=(28,6)):
    """
    dependency: must have forward_function
    """
    num_shows = 20

    # Create a figure
    fig = plt.figure(figsize=fig_size)
    plt.rc('text', color='black')

    for i in range(num_shows):
        j, k = divmod(i, num_cols)  # Use divmod to calculate j and k
        t = i * num_steps // num_shows  # Calculate t based on i

        q_i = q_x(dataset, torch.tensor([i*num_steps//num_shows]))

        # Add a subplot in each iteration
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        # Scatter plot
        ax.scatter(q_i[:, 0], q_i[:, 1], c=s_color, edgecolor='white')

        # Customize the axis
        ax.set_axis_off()
        ax.set_title(f'$q(\\mathbf{{x}}_{{{t}}})$')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the entire figure as a single image (e.g., a PNG file)
    plt.savefig('forward_process_visualization.png')


class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(2,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
    def forward(self,x,t):
        #  x = x_0
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        
        return x


def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    batch_size = x_0.shape[0]

    t = torch.randint(0,n_steps,size=(batch_size//2,))
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1)
    
    a = alphas_bar_sqrt[t] # torch.Size([batchsize, 1])
    aml = one_minus_alphas_bar_sqrt[t] # torch.Size([batchsize, 1])
    e = torch.randn_like(x_0) # torch.Size([batchsize, 2])
    x = x_0*a+e*aml # torch.Size([batchsize, 2])
    
    output = model(x,t.squeeze(-1)) #t.squeeze(-1)为torch.Size([batchsize])
    # output:torch.Size([batchsize, 2])

    return (e - output).square().mean()


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, q_x, random_z):
    cur_x = q_x
    x_seq = [cur_x]
    if random_z:
        for i in reversed(range(n_steps)):
            cur_x = p_sample_z(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
    else:
        for i in reversed(range(n_steps)):
            cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
    return x_seq


def p_sample_z(model,x,t,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x,t)

    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x,t)

    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))

    return (mean)


def train(batch_size, num_epoch, dataset, data_colors, q_x, random_z=False):
    print('Training model...')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    plt.rc('text', color='blue')

    model = MLPDiffusion(num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a figure to hold the subplots
    fig = plt.figure(figsize=(28,30))

    for t in tqdm(range(num_epoch)):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        if t % 100 == 0:
            print(loss)
            x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt, q_x, random_z=random_z)

            for i in range(1, 11):
                ax = fig.add_subplot(num_epoch/100, 10, i+10*t/100)
                cur_x = x_seq[i * 10].detach()
                ax.scatter(cur_x[:, 0], cur_x[:, 1], c=data_colors, edgecolor='white')
                ax.set_axis_off()
                ax.set_title(f'$q(\\mathbf{{x}}_{{{i * 10}}})$')

            # Save the figure with a unique name
            fig.savefig(f'backward_visualization.png')
            
    return model

def visualize_output(model, data, dataset, q_x, s_color, random_z=False):
    x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt, q_x, random_z)
    fig,ax = plt.subplots(1,2,figsize = (20,10))
    ax[0].scatter(*data, edgecolor='white', c=s_color)
    ax[0].set_title('Original Dataset')

    cur_x = x_seq[100].detach()
    ax[1].scatter(cur_x[:,0], cur_x[:,1], c=s_color, edgecolor='white');
    ax[1].set_title('Diffusion Model Output')
    fig.savefig(f'final output.png')
